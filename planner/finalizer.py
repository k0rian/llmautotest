from pathlib import Path

from langchain_community.tools.file_management import WriteFileTool

from planner.policies import DEFAULT_AUDIT_REQUEST
from planner.types import AuditState


def _build_markdown_report(
    request: str,
    workspace_path: str,
    perception_summary: str,
    planner_summary: str,
    plan: str,
    runtime_summary: str,
    audit_output: str,
) -> str:
    return (
        "# Audit Conclusion Report\n\n"
        f"## Audit Task\n{request}\n\n"
        f"## Workspace\n`{workspace_path}`\n\n"
        f"## Perception\n{perception_summary or 'N/A'}\n\n"
        f"## Plan Summary\n{planner_summary or 'N/A'}\n\n"
        f"## Execution Plan\n{plan or 'N/A'}\n\n"
        f"## Runtime Summary\n{runtime_summary or 'N/A'}\n\n"
        f"## Audit Conclusion\n{audit_output or 'N/A'}\n"
    )


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
            f"- Step completed: {completed}\n"
            f"- Step failed: {failed}\n"
            f"- Replan count: {replan_count}\n"
            f"- Top evidence:\n{chr(10).join(top_evidence_lines) if top_evidence_lines else 'N/A'}"
        )
    final_output = (
        f"Audit Task: {request}\n"
        f"Workspace: {workspace_path}\n\n"
        f"Perception:\n{perception_summary or 'N/A'}\n\n"
        f"Plan Summary:\n{planner_summary or 'N/A'}\n\n"
        f"Execution Plan:\n{plan}\n\n"
        f"Runtime Summary:\n{runtime_summary or 'N/A'}\n\n"
        f"Audit Conclusion:\n{audit_output}"
    )
    markdown_report = _build_markdown_report(
        request=request,
        workspace_path=workspace_path,
        perception_summary=perception_summary,
        planner_summary=planner_summary,
        plan=plan,
        runtime_summary=runtime_summary,
        audit_output=audit_output,
    )
    output_file = ""
    write_error = ""
    try:
        resolved_workspace = str(Path(workspace_path).resolve()) if workspace_path else ""
        if resolved_workspace:
            output_file = str(Path(resolved_workspace) / "audit_conclusion.md")
            writer = WriteFileTool(root_dir=resolved_workspace)
            writer.invoke(
                {
                    "file_path": "audit_conclusion.md",
                    "text": markdown_report,
                    "append": False,
                }
            )
    except Exception as exc:
        write_error = str(exc)

    if output_file:
        final_output = f"{final_output}\n\nReport File:\n{output_file}"
    if write_error:
        final_output = f"{final_output}\n\nReport Write Error:\n{write_error}"

    result: AuditState = {"final_output": final_output}
    if output_file:
        result["final_output_file"] = output_file
    if write_error:
        result["final_output_write_error"] = write_error
    return result
