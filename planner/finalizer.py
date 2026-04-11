from planner.policies import DEFAULT_AUDIT_REQUEST
from planner.types import AuditState


def finalizer_node(state: AuditState) -> AuditState:
    request = state.get("user_request", "").strip() or DEFAULT_AUDIT_REQUEST
    workspace_path = state.get("workspace_path", "").strip()
    planner_summary = state.get("planner_summary", "").strip()
    plan = state.get("plan", "").strip()
    perception_summary = state.get("perception_summary", "").strip()
    audit_output = state.get("audit_output", "").strip()
    final_output = (
        f"Audit Task: {request}\n"
        f"Workspace: {workspace_path}\n\n"
        f"Perception:\n{perception_summary or 'N/A'}\n\n"
        f"Plan Summary:\n{planner_summary or 'N/A'}\n\n"
        f"Execution Plan:\n{plan}\n\n"
        f"Audit Conclusion:\n{audit_output}"
    )
    return {"final_output": final_output}
