import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from llm_utils import parse_json_payload, read_text, resolve_workspace_target
from llm_utils.config_loader import load_api_key, load_base_url, load_model_name
from planner.policies import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME
from planner.prompt.loader import render_txt
from planner.state import (
    advance_step,
    append_evidence,
    append_open_question,
    append_tool_record,
    create_runtime_state,
    get_current_step,
    increment_replan_count,
    mark_step_completed,
    mark_step_failed,
    mark_step_running,
    record_failure_category,
)
from planner.types import EvidenceItem, PlanStep, PlannerRunResult, PlannerRuntimeState, ToolExecutionRecord
from planner.verifier import (
    build_replan_reason,
    build_step_outcome,
    should_trigger_replan,
    verify_step_result,
)
from planner.semantic_agent import SemanticAgent
from tools.description.description import TOOL_DESCRIPTIONS
from tools.core import WorkspaceGuard
from build.index_store import load_existing_function_index_any
from tools.semantic_diff_ts import DEFAULT_INCLUDE_GLOB, semantic_diff_with_description
from tools.tools import build_tools

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
SEMANTIC_MODE_ALIASES = {
    "audit": "code_audit",
    "code_scan": "code_audit",
    "static_audit": "code_audit",
    "static_scan": "code_audit",
    "summary": "analysis",
    "summarize": "analysis",
    "report": "analysis",
    "semantic_analysis": "semantic_diff",
    "semantic-index": "semantic_index",
    "semantic index": "semantic_index",
    "index": "semantic_index",
    "index_check": "semantic_index",
    "semantic_index_check": "semantic_index",
    "semantic_localization": "semantic_localize",
    "semantic-localize": "semantic_localize",
    "semantic localize": "semantic_localize",
    "semantic_locate": "semantic_localize",
    "semantic-location": "semantic_localize",
    "function_localization": "semantic_localize",
    "function_localize": "semantic_localize",
    "function_location": "semantic_localize",
    "function_locate": "semantic_localize",
    "locate_function": "semantic_localize",
    "symbol_localize": "semantic_localize",
    "symbol_location": "semantic_localize",
    "implementation_localize": "semantic_localize",
    "core_implementation_localize": "semantic_localize",
    "核心实现函数定位": "semantic_localize",
    "函数定位": "semantic_localize",
    "semantic-retrieve": "semantic_retrieve",
    "semantic-retrieval": "semantic_retrieve",
    "semantic retrieve": "semantic_retrieve",
    "function_retrieve": "semantic_retrieve",
    "symbol_retrieve": "semantic_retrieve",
    "semantic-validation": "semantic_validate",
    "semantic-validate": "semantic_validate",
    "semantic validate": "semantic_validate",
    "semantic_validation": "semantic_validate",
    "compliance_validate": "semantic_validate",
    "semanticdiff": "semantic_diff",
    "semantic diff": "semantic_diff",
    "semantic-diff": "semantic_diff",
}
BUILD_ONLY_TOOL_NAMES = {"semantic_index_functions", "build_hierarchical_code_index"}
VALID_STEP_MODES = {
    "code_audit",
    "gui_test",
    "analysis",
    "semantic_diff",
    "semantic_index",
    "semantic_localize",
    "semantic_retrieve",
    "semantic_validate",
}


def build_planner_model(
    model_name: str = "",
    base_url: str = "",
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or load_model_name(DEFAULT_MODEL_NAME),
        base_url=base_url or load_base_url() or DEFAULT_BASE_URL,
        api_key=load_api_key(),
    )


def build_executor_agent(model: ChatOpenAI | None = None, verbose: bool = False):
    active_model = model or build_planner_model()
    return create_agent(
        active_model,
        tools=build_tools(),
        verbose=verbose,
    )


def _extract_text(payload: Any) -> str:
    return read_text(payload)


def _parse_json_payload(text: str) -> dict[str, Any]:
    return parse_json_payload(text, strict=True)


def _tool_description_text() -> str:
    blocks: list[str] = []
    for name, meta in TOOL_DESCRIPTIONS.items():
        if name in BUILD_ONLY_TOOL_NAMES:
            continue
        category = meta.get("category", "general")
        description = meta.get("description", "")
        blocks.append(f"{name} [{category}]\n{description}")
    return "\n\n".join(blocks)


def _extract_evidence_items(step: PlanStep, text: str) -> list[EvidenceItem]:
    normalized = text.strip()
    if not normalized:
        return []
    confidence = 0.6
    lowered = normalized.lower()
    if "line " in lowered or "file" in lowered or "evidence" in lowered:
        confidence = 0.8
    return [
        EvidenceItem(
            id=f"ev-{step.id}-{step.attempt_count}",
            step_id=step.id,
            source_type="executor_text",
            summary=normalized[:280],
            confidence=confidence,
        )
    ]


def _render_prompt(template_name: str, **kwargs: object) -> str:
    return render_txt(PROMPT_DIR / template_name, **kwargs)


def _invoke_tool_callable(tool_obj: Any, **kwargs: Any) -> str:
    func = getattr(tool_obj, "func", None)
    if callable(func):
        return read_text(func(**kwargs))
    invoke = getattr(tool_obj, "invoke", None)
    if callable(invoke):
        return read_text(invoke(kwargs))
    if callable(tool_obj):
        return read_text(tool_obj(**kwargs))
    raise TypeError(f"unsupported tool object: {type(tool_obj).__name__}")


def _canonical_mode_token(value: str) -> str:
    token = (value or "").strip().lower()
    token = re.sub(r"[\s\-]+", "_", token)
    return re.sub(r"_+", "_", token).strip("_")


def _infer_step_mode(raw_mode: str, title: str, objective: str) -> str:
    raw = read_text(raw_mode).strip()
    canonical = _canonical_mode_token(raw)
    if canonical in VALID_STEP_MODES:
        return canonical
    if raw.lower() in SEMANTIC_MODE_ALIASES:
        return SEMANTIC_MODE_ALIASES[raw.lower()]
    if canonical in SEMANTIC_MODE_ALIASES:
        return SEMANTIC_MODE_ALIASES[canonical]

    text = f"{raw} {title} {objective}".lower()
    if any(token in text for token in ("gui", "ui", "界面", "浏览器", "点击")):
        return "gui_test"
    if any(token in text for token in ("diff", "compare", "对比", "差异")):
        return "semantic_diff"
    if any(token in text for token in ("index", "索引")):
        return "semantic_index"
    if any(token in text for token in ("retrieve", "retrieval", "caller", "callee", "call graph", "调用", "上下文", "检索")):
        return "semantic_retrieve"
    if any(token in text for token in ("validate", "validation", "verify", "coverage", "合规", "验证", "覆盖率")):
        return "semantic_validate"
    if any(token in text for token in ("localize", "locate", "location", "symbol", "function", "implementation", "函数", "符号", "定位", "实现", "核心")):
        return "semantic_localize"
    if any(token in text for token in ("audit", "scan", "grep", "read", "lsp", "semgrep", "审计", "扫描", "读取", "搜索", "静态")):
        return "code_audit"
    if any(token in text for token in ("analysis", "summarize", "summary", "report", "判断", "分析", "汇总", "总结", "报告")):
        return "analysis"
    return "analysis"


class PlanAndExecutePlanner:
    def __init__(
        self,
        planner_model: ChatOpenAI | None = None,
        executor_agent: Any | None = None,
        gui_executor: Any | None = None,
        verbose: bool = False,
        max_replans: int = 1,
        gui_max_steps: int = 8,
    ) -> None:
        self.planner_model = planner_model or build_planner_model()
        self.executor_agent = executor_agent or build_executor_agent(self.planner_model, verbose=verbose)
        self.gui_executor = gui_executor
        self.semantic_agent = SemanticAgent()
        self.max_replans = max(0, int(max_replans))
        self.gui_max_steps = max(1, int(gui_max_steps))

    def invoke(self, inputs: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(inputs, str):
            result = self.execute(inputs)
            return result.to_dict()
        objective = (
            inputs.get("objective")
            or inputs.get("task")
            or inputs.get("input")
            or inputs.get("query")
            or ""
        )
        context = read_text(inputs.get("context", ""))
        max_steps = inputs.get("max_steps")
        workspace_path = read_text(inputs.get("workspace_path", "")).strip()
        semantic_required = bool(inputs.get("semantic_required", False))
        semantic_target_hint = read_text(inputs.get("semantic_target_hint", "")).strip()
        result = self.execute(
            objective=objective,
            context=context,
            max_steps=max_steps,
            workspace_path=workspace_path,
            semantic_required=semantic_required,
            semantic_target_hint=semantic_target_hint,
        )
        return result.to_dict()

    def create_plan(self, objective: str, context: str = "") -> tuple[str, list[PlanStep]]:
        prompt = _render_prompt(
            "create_plan.txt",
            tool_descriptions=_tool_description_text(),
            objective=objective,
            context=context or "N/A",
        )
        raw = _extract_text(self.planner_model.invoke(prompt))
        try:
            payload = _parse_json_payload(raw)
            summary = read_text(payload.get("summary", "")).strip() or "执行任务规划"
            steps_data = payload.get("steps", [])
            steps = self._normalize_steps(steps_data)
            if not steps:
                raise ValueError("planner returned empty steps")
            return summary, steps
        except Exception:
            fallback = [
                PlanStep(
                    id="step-1",
                    title="执行任务",
                    mode="code_audit",
                    objective=objective,
                    expected_output="返回与目标直接相关的执行结果",
                )
            ]
            return "执行任务规划", fallback

    def execute(
        self,
        objective: str,
        context: str = "",
        max_steps: int | None = None,
        workspace_path: str = "",
        semantic_required: bool = False,
        semantic_target_hint: str = "",
    ) -> PlannerRunResult:
        if not objective or not objective.strip():
            empty_state = create_runtime_state(objective="", context=context, summary="", steps=[])
            empty_state.status = "failed"
            return PlannerRunResult(
                objective="",
                summary="",
                steps=[],
                final_response="objective 不能为空",
                status="failed",
                runtime_state=empty_state.to_dict(),
            )

        summary, steps = self.create_plan(objective=objective.strip(), context=context)
        steps = self._enforce_semantic_steps(
            steps=steps,
            semantic_required=semantic_required,
            target_hint=semantic_target_hint,
        )
        if max_steps is not None:
            limit = max(1, int(max_steps))
            steps = steps[:limit]

        runtime_state = create_runtime_state(
            objective=objective.strip(),
            context=context,
            summary=summary,
            steps=steps,
            semantic_required=semantic_required,
            semantic_target_hint=semantic_target_hint,
        )

        while True:
            step = get_current_step(runtime_state)
            if step is None:
                runtime_state.status = "completed"
                break

            if step.mode == "invalid":
                message = f"invalid_mode: unsupported step mode '{step.raw_mode or 'unknown'}' at {step.id}"
                mark_step_failed(runtime_state, step.id, message, failure_reason="invalid_mode")
                append_open_question(runtime_state, message)
                record_failure_category(runtime_state, "invalid_mode")
                runtime_state.status = "failed"
                break

            mark_step_running(runtime_state, step.id)
            result_text, tool_record = self.execute_step(
                objective=runtime_state.objective,
                context=runtime_state.context,
                step=step,
                history=runtime_state.steps[: runtime_state.current_step_index],
                runtime_state=runtime_state,
                workspace_path=workspace_path,
            )
            append_tool_record(runtime_state, tool_record)
            verified, reason = verify_step_result(step.mode, result_text, tool_record.structured_output)
            evidence_items = _extract_evidence_items(step, result_text)
            for evidence in evidence_items:
                append_evidence(runtime_state, evidence)
            outcome = build_step_outcome(
                step=step,
                text=result_text,
                verified=verified,
                reason=reason,
                evidence_ids=[item.id for item in evidence_items],
            )

            if outcome.status == "completed":
                mark_step_completed(runtime_state, step.id, result_text, notes=outcome.summary)
                advance_step(runtime_state)
                continue

            mark_step_failed(runtime_state, step.id, result_text, failure_reason=outcome.replan_reason or reason)
            append_open_question(runtime_state, f"{step.id} failed: {outcome.replan_reason or reason}")
            failure_category = self._map_failure_category(outcome.replan_reason or reason)
            if failure_category:
                record_failure_category(runtime_state, failure_category)
            if should_trigger_replan(outcome, runtime_state, max_replans=self.max_replans):
                increment_replan_count(runtime_state)
                remaining, replan_parse_failed = self.replan(
                    state=runtime_state,
                    failed_step=step,
                    reason=outcome.replan_reason or reason,
                )
                if replan_parse_failed:
                    record_failure_category(runtime_state, "replan_parse_failed")
                    append_open_question(runtime_state, "replan_parse_failed: fallback to deterministic semantic step")
                    if runtime_state.semantic_required:
                        fallback_target = step.target_path or runtime_state.semantic_target_hint
                        runtime_state.steps = runtime_state.steps[: runtime_state.current_step_index] + [
                            self._build_forced_semantic_pipeline_step(mode="semantic_index", target_hint=fallback_target),
                            self._build_forced_semantic_pipeline_step(mode="semantic_localize", target_hint=fallback_target),
                            self._build_forced_semantic_pipeline_step(mode="semantic_retrieve", target_hint=fallback_target),
                            self._build_forced_semantic_pipeline_step(mode="semantic_validate", target_hint=fallback_target),
                        ]
                        continue
                if remaining:
                    remaining = self._enforce_semantic_steps(
                        steps=remaining,
                        semantic_required=runtime_state.semantic_required,
                        target_hint=runtime_state.semantic_target_hint,
                    )
                    runtime_state.steps = runtime_state.steps[: runtime_state.current_step_index] + remaining
                    continue

            runtime_state.status = "failed"
            break

        final_response = self.summarize_execution(
            objective=runtime_state.objective,
            summary=runtime_state.plan_summary,
            steps=runtime_state.steps,
        )
        return PlannerRunResult(
            objective=runtime_state.objective,
            summary=runtime_state.plan_summary,
            steps=runtime_state.steps,
            final_response=final_response,
            status=runtime_state.status,
            runtime_state=runtime_state.to_dict(),
        )

    def execute_step(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
        runtime_state: PlannerRuntimeState,
        workspace_path: str = "",
    ) -> tuple[str, ToolExecutionRecord]:
        tool_id = f"tool-{len(runtime_state.tool_history or []) + 1}"
        if step.mode == "gui_test":
            text = self._execute_gui_step(
                objective=objective,
                context=context,
                step=step,
                history=history,
            )
            lowered = text.lower()
            return text, ToolExecutionRecord(
                id=tool_id,
                step_id=step.id,
                tool_name="executor_agent(gui)",
                input_summary=f"{step.mode}:{step.objective[:120]}",
                output_summary=text[:200],
                success="failed" not in lowered and "timeout" not in lowered and "error" not in lowered,
            )

        if step.mode == "semantic_diff":
            text, structured_output = self._execute_semantic_diff_step(
                objective=objective,
                context=context,
                step=step,
                history=history,
                workspace_path=workspace_path,
            )
            lowered = text.lower()
            error_stage = read_text(structured_output.get("error_stage", "")).strip().lower()
            return text, ToolExecutionRecord(
                id=tool_id,
                step_id=step.id,
                tool_name="executor_agent(semantic_diff)",
                input_summary=f"{step.mode}:{step.objective[:120]}",
                output_summary=text[:200],
                success=(
                    "failed" not in lowered
                    and "timeout" not in lowered
                    and "error" not in lowered
                    and error_stage not in {"index", "diff", "scope_invalid"}
                ),
                structured_output=structured_output,
            )

        if step.mode in {"semantic_index", "semantic_localize", "semantic_retrieve", "semantic_validate"}:
            text, structured_output = self._execute_semantic_pipeline_step(
                objective=objective,
                context=context,
                step=step,
                runtime_state=runtime_state,
                workspace_path=workspace_path,
            )
            lowered = text.lower()
            return text, ToolExecutionRecord(
                id=tool_id,
                step_id=step.id,
                tool_name=f"executor_agent({step.mode})",
                input_summary=f"{step.mode}:{step.objective[:120]}",
                output_summary=text[:200],
                success=("failed" not in lowered and "error" not in lowered),
                structured_output=structured_output,
            )

        prompt = self._build_execution_prompt(
            objective=objective,
            context=context,
            step=step,
            history=history,
        )
        try:
            response = self.executor_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                }
            )
            text = _extract_text(response).strip()
            return text, ToolExecutionRecord(
                id=tool_id,
                step_id=step.id,
                tool_name="executor_agent",
                input_summary=f"{step.mode}:{step.objective[:120]}",
                output_summary=text[:200],
                success=True,
            )
        except Exception as exc:
            text = str(exc)
            return text, ToolExecutionRecord(
                id=tool_id,
                step_id=step.id,
                tool_name="executor_agent",
                input_summary=f"{step.mode}:{step.objective[:120]}",
                output_summary=text[:200],
                success=False,
                error_message=text,
            )

    def replan(
        self,
        state: PlannerRuntimeState,
        failed_step: PlanStep,
        reason: str,
    ) -> tuple[list[PlanStep], bool]:
        completed_steps = [step for step in state.steps if step.status == "completed"]
        completed_text = self._history_text(completed_steps)
        failure_reason = build_replan_reason(
            runtime_state=state,
            failed_step=failed_step,
            reason=reason,
        )
        failed_text = json.dumps(asdict(failed_step), ensure_ascii=False, indent=2)
        prompt = _render_prompt(
            "replan.txt",
            objective=state.objective,
            plan_summary=state.plan_summary,
            context=state.context or "N/A",
            completed_text=completed_text or "N/A",
            failure_reason=failure_reason,
            failed_step=failed_text,
        )
        try:
            raw = _extract_text(self.planner_model.invoke(prompt))
            payload = _parse_json_payload(raw)
            return self._normalize_steps(payload.get("steps", [])), False
        except Exception:
            return [], True

    def summarize_execution(self, objective: str, summary: str, steps: list[PlanStep]) -> str:
        lightweight_steps: list[dict[str, Any]] = []
        for step in steps:
            result_text = read_text(step.result).strip()
            lightweight_steps.append(
                {
                    "id": step.id,
                    "title": step.title,
                    "mode": step.mode,
                    "status": step.status,
                    "objective": step.objective[:240],
                    "target_path": step.target_path,
                    "notes": step.notes[:400],
                    "failure_reason": step.failure_reason,
                    "result_summary": (result_text[:800] + ("..." if len(result_text) > 800 else "")),
                }
            )
        execution_text = json.dumps(lightweight_steps, ensure_ascii=False, indent=2)
        prompt = _render_prompt(
            "summarize_execution.txt",
            objective=objective,
            summary=summary,
            execution_text=execution_text or "N/A",
        )
        try:
            return _extract_text(self.planner_model.invoke(prompt)).strip()
        except Exception:
            return execution_text

    def _normalize_steps(self, steps_data: Any) -> list[PlanStep]:
        steps: list[PlanStep] = []
        if not isinstance(steps_data, list):
            return steps
        for idx, item in enumerate(steps_data, start=1):
            if not isinstance(item, dict):
                continue
            raw_mode = read_text(item.get("mode", "code_audit")).strip() or "code_audit"
            title = read_text(item.get("title", f"Step {idx}")).strip() or f"Step {idx}"
            objective = read_text(item.get("objective", "")).strip() or title
            normalized_mode = _infer_step_mode(raw_mode=raw_mode, title=title, objective=objective)
            expected_output = read_text(item.get("expected_output", "")).strip() or "Return the step execution result"
            target_path = read_text(item.get("target_path", "")).strip()
            steps.append(
                PlanStep(
                    id=f"step-{idx}",
                    title=title,
                    mode=normalized_mode,  # type: ignore[arg-type]
                    objective=objective,
                    expected_output=expected_output,
                    target_path=target_path,
                    raw_mode=raw_mode,
                )
            )
        return steps

    def _enforce_semantic_steps(
        self,
        steps: list[PlanStep],
        semantic_required: bool,
        target_hint: str = "",
    ) -> list[PlanStep]:
        if not semantic_required:
            return self._reindex_steps(steps)
        required_modes = ["semantic_index", "semantic_localize", "semantic_retrieve", "semantic_validate"]
        existing_modes = [step.mode for step in steps]
        for mode in reversed(required_modes):
            if mode in existing_modes:
                continue
            steps.insert(0, self._build_forced_semantic_pipeline_step(mode=mode, target_hint=target_hint))
        first_semantic = next(
            (idx for idx, step in enumerate(steps) if step.mode in set(required_modes) | {"semantic_diff"}),
            0,
        )
        if first_semantic > 1:
            semantic_block = [step for step in steps if step.mode in set(required_modes) | {"semantic_diff"}]
            non_semantic = [step for step in steps if step.mode not in set(required_modes) | {"semantic_diff"}]
            steps = non_semantic[:1] + semantic_block + non_semantic[1:]
        return self._reindex_steps(steps)

    def _reindex_steps(self, steps: list[PlanStep]) -> list[PlanStep]:
        for idx, step in enumerate(steps, start=1):
            step.id = f"step-{idx}"
        return steps

    def _build_forced_semantic_step(self, target_hint: str = "") -> PlanStep:
        return PlanStep(
            id="step-forced-semantic",
            title="Run deterministic semantic index and diff",
            mode="semantic_diff",
            objective="Build semantic index and run requirement-to-implementation semantic diff.",
            expected_output=(
                "Must include index_result, diff_result, coverage summary, and evidence file/path entries."
            ),
            target_path=target_hint.strip(),
            raw_mode="semantic_diff",
        )

    def _build_forced_semantic_pipeline_step(self, mode: str, target_hint: str = "") -> PlanStep:
        title_map = {
            "semantic_index": "Check existing semantic index",
            "semantic_localize": "Localize requirement with hierarchical index",
            "semantic_retrieve": "Retrieve symbol/graph context for unknown findings",
            "semantic_validate": "Validate semantic findings and classify confidence",
        }
        objective_map = {
            "semantic_index": "Check that a prebuilt repository -> directory -> file -> function semantic index is available.",
            "semantic_localize": "Localize requirement to top directories/files/functions with score and reasons.",
            "semantic_retrieve": "When evidence is insufficient, retrieve definition/caller/callee context and re-detect.",
            "semantic_validate": "Validate preliminary semantic findings and output structured decision and confidence.",
        }
        return PlanStep(
            id=f"step-forced-{mode}",
            title=title_map.get(mode, mode),
            mode=mode,  # type: ignore[arg-type]
            objective=objective_map.get(mode, mode),
            expected_output=(
                "Must include requirement, localized_candidates, retrieved_contexts, decision, confidence, evidence."
            ),
            target_path=target_hint.strip(),
            raw_mode=mode,
        )

    def _has_completed_semantic_step(self, state: PlannerRuntimeState) -> bool:
        semantic_modes = {
            "semantic_diff",
            "semantic_index",
            "semantic_localize",
            "semantic_retrieve",
            "semantic_validate",
        }
        return any(step.mode in semantic_modes and step.status == "completed" for step in state.steps)

    def _map_failure_category(self, reason: str) -> str:
        token = (reason or "").strip().lower()
        mapping = {
            "tool_failed": "tool_failed",
            "no_evidence": "verifier_no_evidence",
            "low_confidence": "verifier_low_confidence",
            "invalid_mode": "invalid_mode",
            "scope_invalid": "scope_invalid",
            "replan_parse_failed": "replan_parse_failed",
        }
        return mapping.get(token, token)

    def _build_execution_prompt(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
    ) -> str:
        return _render_prompt(
            "execute_step.txt",
            objective=objective,
            context=context or "N/A",
            history=self._history_text(history) or "N/A",
            step_title=step.title,
            step_mode=step.mode,
            step_objective=step.objective,
            expected_output=step.expected_output,
        )

    def _history_text(self, history: list[PlanStep]) -> str:
        if not history:
            return ""
        return "\n\n".join(
            f"{step.id} | {step.title} | {step.status}\n目标: {step.objective}\n结果: {step.result}"
            for step in history
        )

    def _execute_gui_step(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
    ) -> str:
        prompt = _render_prompt(
            "execute_gui_step.txt",
            objective=objective,
            context=context or "N/A",
            history=self._history_text(history) or "N/A",
            step_title=step.title,
            step_objective=step.objective,
            expected_output=step.expected_output,
            gui_max_steps=self.gui_max_steps,
        )
        response = self.executor_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        return _extract_text(response).strip()

    def _execute_semantic_diff_step(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
        workspace_path: str,
    ) -> tuple[str, dict[str, Any]]:
        resolved_workspace = workspace_path.strip() or self._extract_workspace_path_from_context(context)
        if not resolved_workspace:
            return (
                "semantic_diff workflow failed: workspace path is empty",
                {
                    "index_success": False,
                    "diff_success": False,
                    "index_cache_path": "",
                    "matched_count": 0,
                    "missing_count": 0,
                    "evidence_paths": [],
                    "error_stage": "scope_invalid",
                    "index_result_file": "",
                    "diff_result_file": "",
                    "run_directory": "",
                },
            )

        target_path, scope_error = self._resolve_semantic_target_path(
            workspace_path=resolved_workspace,
            target_path=step.target_path,
            objective=step.objective,
            context=context,
        )
        if scope_error:
            return (
                f"semantic_diff workflow failed: {scope_error}",
                {
                    "index_success": False,
                    "diff_success": False,
                    "index_cache_path": "",
                    "matched_count": 0,
                    "missing_count": 0,
                    "evidence_paths": [],
                    "error_stage": "scope_invalid",
                    "index_result_file": "",
                    "diff_result_file": "",
                    "run_directory": "",
                },
            )

        description = self._build_semantic_description(
            objective=objective,
            context=context,
            step=step,
            history=history,
        )
        try:
            existing_index, index_cache_path, index_error = load_existing_function_index_any(
                path=target_path,
                include_glob=DEFAULT_INCLUDE_GLOB,
                max_files=2000,
            )
            if existing_index is None:
                raise RuntimeError(index_error)
            index_text = json.dumps(
                {
                    "status": "ok",
                    "root": existing_index.root,
                    "resolved_path": existing_index.resolved_path,
                    "target_path": existing_index.target_path,
                    "scope_type": existing_index.scope_type,
                    "file_count": existing_index.file_count,
                    "function_count": existing_index.function_count,
                    "summary_mode": existing_index.summary_mode,
                    "summary_model": existing_index.summary_model,
                    "indexed_targets": existing_index.target_files,
                    "cache_path": index_cache_path,
                },
                ensure_ascii=False,
            )
            diff_result = _invoke_tool_callable(
                semantic_diff_with_description,
                path=target_path,
                description=description,
                rebuild=False,
            )
            diff_text = read_text(diff_result).strip()
            index_data = self._safe_parse_json(index_text)
            diff_data = self._safe_parse_json(diff_text)
            artifacts = self._persist_semantic_run(
                workspace_path=resolved_workspace,
                step=step,
                target_path=target_path,
                description=description,
                index_raw=index_text,
                diff_raw=diff_text,
                index_data=index_data,
                diff_data=diff_data,
            )
            parsed = self._build_semantic_structured_output(
                index_data=index_data,
                diff_data=diff_data,
                artifacts=artifacts,
            )
            summary_text = self._format_semantic_summary(parsed)
            return summary_text, parsed
        except Exception as exc:
            return (
                f"semantic_diff workflow failed: {exc}",
                {
                    "index_success": False,
                    "diff_success": False,
                    "index_cache_path": "",
                    "matched_count": 0,
                    "missing_count": 0,
                    "evidence_paths": [],
                    "error_stage": "tool_failed",
                    "index_result_file": "",
                    "diff_result_file": "",
                    "run_directory": "",
                },
            )

    def _execute_semantic_pipeline_step(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        runtime_state: PlannerRuntimeState,
        workspace_path: str,
    ) -> tuple[str, dict[str, Any]]:
        resolved_workspace = workspace_path.strip() or self._extract_workspace_path_from_context(context)
        if not resolved_workspace:
            return (
                f"{step.mode} workflow failed: workspace path is empty",
                {
                    "stage": step.mode,
                    "error_stage": "scope_invalid",
                    "decision": "unknown",
                    "confidence": 0.0,
                    "requirement": step.objective,
                    "localized_candidates": {},
                    "retrieved_contexts": [],
                    "evidence": [],
                    "missing_requirements": [],
                    "covered_requirements": [],
                    "partial_requirements": [],
                },
            )

        target_path, scope_error = self._resolve_semantic_target_path(
            workspace_path=resolved_workspace,
            target_path=step.target_path,
            objective=step.objective,
            context=context,
        )
        if scope_error:
            return (
                f"{step.mode} workflow failed: {scope_error}",
                {
                    "stage": step.mode,
                    "error_stage": "scope_invalid",
                    "decision": "unknown",
                    "confidence": 0.0,
                    "requirement": step.objective,
                    "localized_candidates": {},
                    "retrieved_contexts": [],
                    "evidence": [],
                    "missing_requirements": [],
                    "covered_requirements": [],
                    "partial_requirements": [],
                },
            )
        try:
            return self.semantic_agent.execute_step(
                mode=step.mode,
                objective=objective,
                step=step,
                runtime_state=runtime_state,
                workspace_path=resolved_workspace,
                target_path=target_path,
            )
        except Exception as exc:
            return (
                f"{step.mode} workflow failed: {exc}",
                {
                    "stage": step.mode,
                    "error_stage": "tool_failed",
                    "decision": "unknown",
                    "confidence": 0.0,
                    "requirement": step.objective,
                    "localized_candidates": {},
                    "retrieved_contexts": [],
                    "evidence": [],
                    "missing_requirements": [],
                    "covered_requirements": [],
                    "partial_requirements": [],
                },
            )

    def _resolve_semantic_target_path(
        self,
        workspace_path: str,
        target_path: str,
        objective: str,
        context: str,
    ) -> tuple[str, str]:
        workspace = str(Path(workspace_path).resolve())
        candidate = target_path.strip()
        if not candidate:
            candidate = self._guess_target_path_from_text(objective) or self._guess_target_path_from_text(context)
        if not candidate:
            return workspace, ""
        resolved_str, error = resolve_workspace_target(workspace, candidate)
        if error:
            return "", error
        try:
            WorkspaceGuard.ensure_under_workspace(workspace, resolved_str)
        except Exception:
            return "", f"scope_invalid: target path '{candidate}' is outside workspace"
        return resolved_str, ""

    def _guess_target_path_from_text(self, text: str) -> str:
        source = (text or "").strip()
        if not source:
            return ""
        path_token = (
            r"(?:"
            r"(?:[A-Za-z]:)?(?:[\\/][A-Za-z0-9_.\-]+)+(?:[\\/])?"
            r"|"
            r"[A-Za-z0-9_.\-]+(?:[\\/][A-Za-z0-9_.\-]+)+(?:[\\/])?"
            r")"
        )
        patterns = (
            rf"({path_token})",
            rf"(?:in|under)\s+({path_token})",
        )
        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if not match:
                continue
            token = match.group(1).strip().strip("`'\"").rstrip(".,;:)")
            if token and any(sep in token for sep in ("/", "\\")):
                return token
        return ""

    def _build_semantic_structured_output(
        self,
        index_data: dict[str, Any],
        diff_data: dict[str, Any],
        artifacts: dict[str, str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index_success": True,
            "diff_success": True,
            "index_cache_path": "",
            "matched_count": 0,
            "missing_count": 0,
            "evidence_paths": [],
            "error_stage": "",
            "index_result_file": artifacts.get("index_result_file", ""),
            "diff_result_file": artifacts.get("diff_result_file", ""),
            "run_directory": artifacts.get("run_directory", ""),
            "missing_preview": [],
            "partial_preview": [],
            "top_matches_preview": [],
            "missing_total": 0,
            "partial_total": 0,
            "top_matches_total": 0,
        }

        cache_keys = ("index_cache_path", "cache_path", "index_path", "artifact_path")
        for key in cache_keys:
            value = read_text(index_data.get(key, "")).strip() if isinstance(index_data, dict) else ""
            if value:
                payload["index_cache_path"] = value
                break

        matched = diff_data.get("requirement_matches") if isinstance(diff_data, dict) else None
        missing = diff_data.get("missing_requirements") if isinstance(diff_data, dict) else None
        partial = diff_data.get("partial_matches") if isinstance(diff_data, dict) else None
        top_matches = diff_data.get("top_matches") if isinstance(diff_data, dict) else None
        if isinstance(matched, list):
            payload["matched_count"] = len(matched)
        else:
            payload["matched_count"] = int(diff_data.get("matched_count", 0)) if isinstance(diff_data, dict) else 0
        if isinstance(missing, list):
            payload["missing_count"] = len(missing)
            payload["missing_total"] = len(missing)
            payload["missing_preview"] = self._clip_collection(missing, limit=10)
        else:
            payload["missing_count"] = int(diff_data.get("missing_count", 0)) if isinstance(diff_data, dict) else 0
            payload["missing_total"] = payload["missing_count"]
        if isinstance(partial, list):
            payload["partial_total"] = len(partial)
            payload["partial_preview"] = self._clip_collection(partial, limit=10)
        if isinstance(top_matches, list):
            payload["top_matches_total"] = len(top_matches)
            payload["top_matches_preview"] = self._clip_collection(top_matches, limit=10)

        evidence_paths: list[str] = []
        if isinstance(diff_data, dict):
            candidates = diff_data.get("evidence_paths", [])
            if isinstance(candidates, list):
                evidence_paths.extend([read_text(item).strip() for item in candidates if read_text(item).strip()])
            if isinstance(matched, list):
                for item in matched:
                    if not isinstance(item, dict):
                        continue
                    value = read_text(item.get("path") or item.get("file") or "").strip()
                    if value:
                        evidence_paths.append(value)
        payload["evidence_paths"] = sorted({value for value in evidence_paths if value})
        return payload

    def _safe_parse_json(self, raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {}
        try:
            payload = _parse_json_payload(text)
            return payload if isinstance(payload, dict) else {"data": payload}
        except Exception:
            return {"raw_text": text}

    def _persist_semantic_run(
        self,
        workspace_path: str,
        step: PlanStep,
        target_path: str,
        description: str,
        index_raw: str,
        diff_raw: str,
        index_data: dict[str, Any],
        diff_data: dict[str, Any],
    ) -> dict[str, str]:
        base_dir = Path(workspace_path) / ".llmautotest" / "semantic_runs"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = base_dir / f"{stamp}_{step.id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        index_file = run_dir / "index_result.json"
        diff_file = run_dir / "diff_result.json"
        meta_file = run_dir / "run_meta.json"

        index_payload = index_data if isinstance(index_data, dict) else {"raw_text": index_raw}
        diff_payload = diff_data if isinstance(diff_data, dict) else {"raw_text": diff_raw}
        index_file.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        diff_file.write_text(json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_file.write_text(
            json.dumps(
                {
                    "step_id": step.id,
                    "step_title": step.title,
                    "workspace_path": workspace_path,
                    "target_path": target_path,
                    "description": description,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "run_directory": str(run_dir),
            "index_result_file": str(index_file),
            "diff_result_file": str(diff_file),
            "meta_file": str(meta_file),
        }

    def _clip_collection(self, items: list[Any], limit: int = 10) -> list[Any]:
        if limit <= 0:
            return []
        clipped: list[Any] = []
        for item in items[:limit]:
            if isinstance(item, dict):
                mini: dict[str, Any] = {}
                for key in ("id", "title", "requirement", "summary", "path", "file", "symbol", "status"):
                    value = item.get(key)
                    if value is not None and read_text(value).strip():
                        mini[key] = read_text(value).strip()[:240]
                clipped.append(mini if mini else {"raw": read_text(item)[:240]})
            else:
                clipped.append(read_text(item)[:240])
        return clipped

    def _format_semantic_summary(self, structured: dict[str, Any]) -> str:
        evidence_paths = structured.get("evidence_paths", [])
        evidence_count = len(evidence_paths) if isinstance(evidence_paths, list) else 0
        missing_preview = structured.get("missing_preview", [])
        partial_preview = structured.get("partial_preview", [])
        top_matches_preview = structured.get("top_matches_preview", [])
        lines = [
            "semantic_diff workflow summary",
            f"- matched_count: {structured.get('matched_count', 0)}",
            f"- missing_count: {structured.get('missing_count', 0)}",
            f"- evidence_paths: {evidence_count}",
            f"- index_cache_path: {structured.get('index_cache_path', '') or 'N/A'}",
            f"- run_directory: {structured.get('run_directory', '') or 'N/A'}",
            f"- index_result_file: {structured.get('index_result_file', '') or 'N/A'}",
            f"- diff_result_file: {structured.get('diff_result_file', '') or 'N/A'}",
            f"- missing_preview_count: {len(missing_preview) if isinstance(missing_preview, list) else 0}",
            f"- partial_preview_count: {len(partial_preview) if isinstance(partial_preview, list) else 0}",
            f"- top_matches_preview_count: {len(top_matches_preview) if isinstance(top_matches_preview, list) else 0}",
        ]
        return "\n".join(lines)

    def _extract_workspace_path_from_context(self, context: str) -> str:
        text = context or ""
        patterns = (
            r"^Workspace path:\s*(.+)$",
            r"^workspace_path:\s*(.+)$",
            r"^工作区路径[:：]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return ""

    def _build_semantic_description(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
    ) -> str:
        _ = history  # intentionally not used to avoid injecting planner meta instructions
        sections = self._extract_rfc_sections(f"{objective}\n{step.objective}\n{context}")
        section_text = ", ".join(sections) if sections else "N/A"
        target_scope = step.target_path.strip() or self._guess_target_path_from_text(step.objective)
        return (
            "Audit objective:\n"
            f"{objective.strip()}\n\n"
            "Semantic step objective:\n"
            f"{step.objective.strip()}\n\n"
            "RFC sections in scope:\n"
            f"{section_text}\n\n"
            "Target path scope:\n"
            f"{target_scope or 'workspace root'}\n"
        )

    def _extract_rfc_sections(self, text: str) -> list[str]:
        source = (text or "").strip()
        if not source:
            return []
        pattern = r"\b\d+\.\d+\b(?:\s+[A-Za-z][A-Za-z0-9 _-]*)?"
        matches = [item.strip() for item in re.findall(pattern, source)]
        seen: set[str] = set()
        ordered: list[str] = []
        for item in matches:
            norm = item.lower()
            if norm in seen:
                continue
            seen.add(norm)
            ordered.append(item)
        return ordered[:20]


def build_plan_and_execute_planner(
    planner_model: ChatOpenAI | None = None,
    executor_agent: Any | None = None,
    gui_executor: Any | None = None,
    verbose: bool = False,
    max_replans: int = 1,
    gui_max_steps: int = 8,
) -> PlanAndExecutePlanner:
    return PlanAndExecutePlanner(
        planner_model=planner_model,
        executor_agent=executor_agent,
        gui_executor=gui_executor,
        verbose=verbose,
        max_replans=max_replans,
        gui_max_steps=gui_max_steps,
    )
