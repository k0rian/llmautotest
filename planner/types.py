from dataclasses import asdict, dataclass
from typing import Any, Literal, TypedDict


StepMode = Literal["code_audit", "gui_test", "analysis"]


class AuditState(TypedDict, total=False):
    user_request: str
    workspace_path: str
    skill_prompt: str
    perception_summary: str
    lsp_ready: bool
    lsp_error: str
    plan: str
    planner_summary: str
    audit_output: str
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


@dataclass
class PlannerRunResult:
    objective: str
    summary: str
    steps: list[PlanStep]
    final_response: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "summary": self.summary,
            "steps": [asdict(step) for step in self.steps],
            "final_response": self.final_response,
            "status": self.status,
        }
