import json
from typing import Any

from planner.types import (
    AuditState,
    EvidenceItem,
    PlanStep,
    PlannerRuntimeState,
    ToolExecutionRecord,
)


def read_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "messages" in value and value["messages"]:
            return read_text(value["messages"][-1])
        if "output" in value:
            return read_text(value["output"])
        return json.dumps(value, ensure_ascii=False, indent=2)
    content = getattr(value, "content", None)
    if content is not None:
        return read_text(content)
    return str(value)


def format_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        title = read_text(step.get("title", f"Step {index}")).strip() or f"Step {index}"
        mode = read_text(step.get("mode", "code_audit")).strip() or "code_audit"
        objective = read_text(step.get("objective", "")).strip()
        lines.append(f"{index}. [{mode}] {title}")
        if objective:
            lines.append(f"   - {objective}")
    return "\n".join(lines)


def build_perception_blocked_state(message: str) -> AuditState:
    return {
        "planner_summary": "Stopped in perception stage",
        "plan": "Execution stage was skipped",
        "audit_output": f"Perception gate blocked execution: {message}",
    }


def create_runtime_state(
    objective: str,
    context: str,
    summary: str,
    steps: list[PlanStep],
    semantic_required: bool = False,
    semantic_target_hint: str = "",
) -> PlannerRuntimeState:
    return PlannerRuntimeState(
        objective=objective,
        context=context,
        plan_summary=summary,
        steps=steps,
        current_step_index=0,
        replan_count=0,
        tool_history=[],
        evidence=[],
        open_questions=[],
        failure_counts={},
        latest_failure_category="",
        semantic_required=semantic_required,
        semantic_target_hint=semantic_target_hint,
        semantic_context={},
        status="running",
    )


def get_current_step(state: PlannerRuntimeState) -> PlanStep | None:
    idx = state.current_step_index
    if idx < 0 or idx >= len(state.steps):
        return None
    return state.steps[idx]


def _find_step(state: PlannerRuntimeState, step_id: str) -> PlanStep | None:
    for step in state.steps:
        if step.id == step_id:
            return step
    return None


def mark_step_running(state: PlannerRuntimeState, step_id: str) -> None:
    step = _find_step(state, step_id)
    if step is None:
        return
    step.status = "running"
    step.attempt_count += 1


def mark_step_completed(state: PlannerRuntimeState, step_id: str, result: str, notes: str = "") -> None:
    step = _find_step(state, step_id)
    if step is None:
        return
    step.status = "completed"
    step.result = result
    step.notes = notes
    step.failure_reason = ""


def mark_step_failed(state: PlannerRuntimeState, step_id: str, result: str, failure_reason: str = "") -> None:
    step = _find_step(state, step_id)
    if step is None:
        return
    step.status = "failed"
    step.result = result
    step.failure_reason = failure_reason


def advance_step(state: PlannerRuntimeState) -> None:
    state.current_step_index += 1


def increment_replan_count(state: PlannerRuntimeState) -> None:
    state.replan_count += 1


def append_tool_record(state: PlannerRuntimeState, record: ToolExecutionRecord) -> None:
    if state.tool_history is None:
        state.tool_history = []
    state.tool_history.append(record)


def append_evidence(state: PlannerRuntimeState, evidence: EvidenceItem) -> None:
    if state.evidence is None:
        state.evidence = []
    state.evidence.append(evidence)


def append_open_question(state: PlannerRuntimeState, question: str) -> None:
    text = question.strip()
    if not text:
        return
    if state.open_questions is None:
        state.open_questions = []
    state.open_questions.append(text)


def record_failure_category(state: PlannerRuntimeState, category: str) -> None:
    value = category.strip()
    if not value:
        return
    if state.failure_counts is None:
        state.failure_counts = {}
    state.failure_counts[value] = int(state.failure_counts.get(value, 0)) + 1
    state.latest_failure_category = value
