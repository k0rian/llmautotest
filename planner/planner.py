import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from llm_utils.config_loader import load_api_key
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
    read_text,
)
from planner.types import EvidenceItem, PlanStep, PlannerRunResult, PlannerRuntimeState, ToolExecutionRecord
from planner.verifier import (
    build_replan_reason,
    build_step_outcome,
    should_trigger_replan,
    verify_step_result,
)
from tools.description.description import TOOL_DESCRIPTIONS
from tools.semantic_diff_ts import semantic_diff_with_description, semantic_index_functions
from tools.tools import build_tools

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"


def build_planner_model(
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
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
    raw = text.strip()
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
        if match:
            raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _tool_description_text() -> str:
    blocks: list[str] = []
    for name, meta in TOOL_DESCRIPTIONS.items():
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
        result = self.execute(
            objective=objective,
            context=context,
            max_steps=max_steps,
            workspace_path=workspace_path,
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
        if max_steps is not None:
            limit = max(1, int(max_steps))
            steps = steps[:limit]

        runtime_state = create_runtime_state(
            objective=objective.strip(),
            context=context,
            summary=summary,
            steps=steps,
        )

        while True:
            step = get_current_step(runtime_state)
            if step is None:
                runtime_state.status = "completed"
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
            verified, reason = verify_step_result(step.mode, result_text)
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
            if should_trigger_replan(outcome, runtime_state, max_replans=self.max_replans):
                increment_replan_count(runtime_state)
                remaining = self.replan(
                    state=runtime_state,
                    failed_step=step,
                    reason=outcome.replan_reason or reason,
                )
                if remaining:
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
            text = self._execute_semantic_diff_step(
                objective=objective,
                context=context,
                step=step,
                history=history,
                workspace_path=workspace_path,
            )
            lowered = text.lower()
            return text, ToolExecutionRecord(
                id=tool_id,
                step_id=step.id,
                tool_name="executor_agent(semantic_diff)",
                input_summary=f"{step.mode}:{step.objective[:120]}",
                output_summary=text[:200],
                success="failed" not in lowered and "timeout" not in lowered and "error" not in lowered,
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
    ) -> list[PlanStep]:
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
            return self._normalize_steps(payload.get("steps", []))
        except Exception:
            return []

    def summarize_execution(self, objective: str, summary: str, steps: list[PlanStep]) -> str:
        execution_text = "\n\n".join(
            json.dumps(asdict(step), ensure_ascii=False, indent=2) for step in steps
        )
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
        valid_modes = {"code_audit", "gui_test", "analysis", "semantic_diff"}
        for idx, item in enumerate(steps_data, start=1):
            if not isinstance(item, dict):
                continue
            mode = read_text(item.get("mode", "code_audit")).strip() or "code_audit"
            if mode not in valid_modes:
                mode = "code_audit"
            title = read_text(item.get("title", f"步骤 {idx}")).strip() or f"步骤 {idx}"
            objective = read_text(item.get("objective", "")).strip() or title
            expected_output = read_text(item.get("expected_output", "")).strip() or "返回本步骤结果"
            steps.append(
                PlanStep(
                    id=f"step-{idx}",
                    title=title,
                    mode=mode,
                    objective=objective,
                    expected_output=expected_output,
                )
            )
        return steps

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
    ) -> str:
        resolved_workspace = workspace_path.strip() or self._extract_workspace_path_from_context(context)
        if not resolved_workspace:
            return "semantic_diff workflow failed: workspace path is empty"

        description = self._build_semantic_description(
            objective=objective,
            context=context,
            step=step,
            history=history,
        )
        try:
            index_result = _invoke_tool_callable(
                semantic_index_functions,
                path=resolved_workspace,
                rebuild=False,
            )
            diff_result = _invoke_tool_callable(
                semantic_diff_with_description,
                path=resolved_workspace,
                description=description,
                rebuild=False,
            )
            return (
                "semantic_diff workflow result\n"
                f"workspace: {resolved_workspace}\n\n"
                f"index_result:\n{read_text(index_result).strip()}\n\n"
                f"diff_result:\n{read_text(diff_result).strip()}"
            )
        except Exception as exc:
            return f"semantic_diff workflow failed: {exc}"

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
        return _render_prompt(
            "semantic_diff_description.txt",
            objective=objective,
            step_title=step.title,
            step_objective=step.objective,
            expected_output=step.expected_output,
            context=context or "N/A",
            history=self._history_text(history) or "N/A",
        )


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
