from dataclasses import asdict, dataclass
from typing import Any, Literal, TypedDict


StepMode = Literal["code_audit", "gui_test", "analysis", "semantic_diff"]


class AuditState(TypedDict, total=False):
    user_request: str
    workspace_path: str
    skill_prompt: str
    perception_summary: str
    perception_meta: dict[str, Any]
    lsp_ready: bool
    lsp_error: str
    plan: str
    planner_summary: str
    audit_output: str
    planner_runtime: dict[str, Any]
    final_output: str


@dataclass
class PlanStep:
    id: str
    title: str
    mode: StepMode
    objective: str
    expected_output: str
    status: str = "pending"
    result: str = ""
    notes: str = ""
    failure_reason: str = ""
    attempt_count: int = 0


@dataclass
class ToolExecutionRecord:
    id: str
    step_id: str
    tool_name: str
    input_summary: str
    output_summary: str
    success: bool
    error_message: str = ""


@dataclass
class EvidenceItem:
    id: str
    step_id: str
    source_type: str
    summary: str
    file_path: str = ""
    symbol_name: str = ""
    confidence: float = 0.0


StepOutcomeStatus = Literal["completed", "failed", "needs_replan"]


@dataclass
class StepOutcome:
    step_id: str
    status: StepOutcomeStatus
    goal_satisfied: bool
    summary: str
    new_evidence_ids: list[str]
    replan_reason: str = ""


RuntimeStatus = Literal["running", "completed", "failed"]


@dataclass
class PlannerRuntimeState:
    objective: str
    context: str
    plan_summary: str
    steps: list[PlanStep]
    current_step_index: int = 0
    replan_count: int = 0
    tool_history: list[ToolExecutionRecord] | None = None
    evidence: list[EvidenceItem] | None = None
    open_questions: list[str] | None = None
    status: RuntimeStatus = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "context": self.context,
            "plan_summary": self.plan_summary,
            "steps": [asdict(step) for step in self.steps],
            "current_step_index": self.current_step_index,
            "replan_count": self.replan_count,
            "tool_history": [asdict(item) for item in (self.tool_history or [])],
            "evidence": [asdict(item) for item in (self.evidence or [])],
            "open_questions": list(self.open_questions or []),
            "status": self.status,
        }


@dataclass
class PlannerRunResult:
    objective: str
    summary: str
    steps: list[PlanStep]
    final_response: str
    status: str
    runtime_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "summary": self.summary,
            "steps": [asdict(step) for step in self.steps],
            "final_response": self.final_response,
            "status": self.status,
            "runtime_state": self.runtime_state,
        }
