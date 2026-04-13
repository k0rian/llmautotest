from dataclasses import asdict

from planner.policies import ALLOW_REPLAN_REASONS, DEFAULT_MAX_REPLANS, MIN_EVIDENCE_PER_STEP
from planner.types import PlanStep, PlannerRuntimeState, StepMode, StepOutcome


def verify_step_result(
    step_mode: StepMode,
    result_text: str,
    structured_output: dict | None = None,
) -> tuple[bool, str]:
    text = (result_text or "").strip()
    if not text:
        return False, "no_evidence"
    lowered = text.lower()

    if step_mode == "invalid":
        return False, "invalid_mode"

    if step_mode in {"semantic_diff", "semantic_index", "semantic_localize", "semantic_retrieve", "semantic_validate"} and isinstance(structured_output, dict):
        error_stage = str(structured_output.get("error_stage", "")).strip().lower()
        if error_stage == "scope_invalid":
            return False, "scope_invalid"
        if error_stage in {"index", "diff", "tool_failed"}:
            return False, "tool_failed"

        if step_mode == "semantic_diff":
            index_success = bool(structured_output.get("index_success", False))
            diff_success = bool(structured_output.get("diff_success", False))
            if not (index_success and diff_success):
                return False, "tool_failed"

            matched_count = int(structured_output.get("matched_count", 0) or 0)
            missing_count = int(structured_output.get("missing_count", 0) or 0)
            evidence_paths = structured_output.get("evidence_paths", [])
            evidence_count = len(evidence_paths) if isinstance(evidence_paths, list) else 0
            if matched_count + missing_count <= 0 and evidence_count <= 0:
                return False, "no_evidence"
        else:
            stage = str(structured_output.get("stage", "")).strip().lower()
            if not stage:
                return False, "no_evidence"
            evidence = structured_output.get("evidence", [])
            localized = structured_output.get("localized_candidates", {})
            retrieved = structured_output.get("retrieved_contexts", [])
            has_localized = isinstance(localized, dict) and any(
                isinstance(localized.get(key), list) and localized.get(key)
                for key in ("directories", "files", "functions")
            )
            has_evidence = isinstance(evidence, list) and len(evidence) > 0
            has_retrieved = isinstance(retrieved, list) and len(retrieved) > 0
            if step_mode in {"semantic_retrieve", "semantic_validate"} and not (has_evidence or has_retrieved):
                return False, "no_evidence"
            if step_mode == "semantic_localize" and not has_localized:
                return False, "no_evidence"
        return True, "ok"

    failed_markers = (
        "tool error",
        "execution failed",
        '"status": "failed"',
        '"status":"failed"',
        '"status": "timeout"',
        '"status":"timeout"',
    )
    if any(marker in lowered for marker in failed_markers):
        return False, "tool_failed"
    if step_mode == "gui_test" and "gui tool error" in lowered:
        return False, "tool_failed"
    if step_mode == "semantic_diff" and "semantic_diff workflow failed" in lowered:
        return False, "tool_failed"

    weak_markers = ("可能", "猜测", "无法确认", "not sure", "uncertain")
    if any(marker in lowered for marker in weak_markers):
        return False, "low_confidence"

    if step_mode == "semantic_diff":
        semantic_markers = (
            "diff_result",
            "requirement_matches",
            "missing_requirements",
            "summary",
            "coverage",
            "evidence",
        )
        marker_hits = sum(1 for marker in semantic_markers if marker in lowered)
        if marker_hits < 2:
            return False, "no_evidence"

    evidence_markers = ("证据", "evidence", "line ", "文件", "file", "路径", "path")
    evidence_hits = sum(1 for marker in evidence_markers if marker in lowered)
    if evidence_hits < MIN_EVIDENCE_PER_STEP:
        return False, "no_evidence"
    return True, "ok"


def build_step_outcome(
    step: PlanStep,
    text: str,
    verified: bool,
    reason: str,
    evidence_ids: list[str] | None = None,
) -> StepOutcome:
    if verified:
        return StepOutcome(
            step_id=step.id,
            status="completed",
            goal_satisfied=True,
            summary=text[:600],
            new_evidence_ids=list(evidence_ids or []),
            replan_reason="",
        )
    return StepOutcome(
        step_id=step.id,
        status="needs_replan" if reason in ALLOW_REPLAN_REASONS else "failed",
        goal_satisfied=False,
        summary=text[:600],
        new_evidence_ids=list(evidence_ids or []),
        replan_reason=reason,
    )


def should_trigger_replan(
    outcome: StepOutcome,
    state: PlannerRuntimeState,
    max_replans: int | None = None,
) -> bool:
    limit = DEFAULT_MAX_REPLANS if max_replans is None else max(0, int(max_replans))
    if outcome.status != "needs_replan":
        return False
    if outcome.replan_reason not in ALLOW_REPLAN_REASONS:
        return False
    return state.replan_count < limit


def build_replan_reason(
    runtime_state: PlannerRuntimeState,
    failed_step: PlanStep,
    reason: str,
) -> str:
    completed_steps = [step for step in runtime_state.steps if step.status == "completed"]
    completed = "\n\n".join(
        f"{step.id} | {step.title} | {step.status}\n目标: {step.objective}\n结果: {step.result}"
        for step in completed_steps
    ) or "N/A"
    failed_text = asdict(failed_step)
    return (
        f"总目标:\n{runtime_state.objective}\n\n"
        f"规划摘要:\n{runtime_state.plan_summary}\n\n"
        f"重规划次数:\n{runtime_state.replan_count}\n\n"
        f"已完成步骤:\n{completed}\n\n"
        f"失败原因:\n{reason}\n\n"
        f"失败步骤:\n{failed_text}"
    )
