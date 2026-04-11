from planner.policies import DEFAULT_AUDIT_REQUEST
from planner.types import AuditState


def finalizer_node(state: AuditState) -> AuditState:
    request = state.get("user_request", "").strip() or DEFAULT_AUDIT_REQUEST
    workspace_path = state.get("workspace_path", "").strip()
    planner_summary = state.get("planner_summary", "").strip()
    plan = state.get("plan", "").strip()
    perception_summary = state.get("perception_summary", "").strip()
    audit_output = state.get("audit_output", "").strip()
    runtime = state.get("planner_runtime", {})
    runtime_summary = ""
    if isinstance(runtime, dict) and runtime:
        steps = runtime.get("steps", []) if isinstance(runtime.get("steps"), list) else []
        completed = sum(1 for item in steps if isinstance(item, dict) and item.get("status") == "completed")
        failed = sum(1 for item in steps if isinstance(item, dict) and item.get("status") == "failed")
        replan_count = runtime.get("replan_count", 0)
        evidence = runtime.get("evidence", []) if isinstance(runtime.get("evidence"), list) else []
        top_evidence_lines: list[str] = []
        for idx, item in enumerate(evidence[:3], start=1):
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "")).strip()
            if summary:
                top_evidence_lines.append(f"{idx}. {summary[:120]}")
        runtime_summary = (
            "Runtime Summary:\n"
            f"- Step completed: {completed}\n"
            f"- Step failed: {failed}\n"
            f"- Replan count: {replan_count}\n"
            f"- Top evidence:\n{chr(10).join(top_evidence_lines) if top_evidence_lines else 'N/A'}\n\n"
        )
    final_output = (
        f"Audit Task: {request}\n"
        f"Workspace: {workspace_path}\n\n"
        f"Perception:\n{perception_summary or 'N/A'}\n\n"
        f"Plan Summary:\n{planner_summary or 'N/A'}\n\n"
        f"Execution Plan:\n{plan}\n\n"
        f"{runtime_summary}"
        f"Audit Conclusion:\n{audit_output}"
    )
    return {"final_output": final_output}
