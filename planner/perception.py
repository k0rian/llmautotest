import json
import logging

from router.router import build_perception_result

from planner.policies import PROMPT_FILE
from planner.types import AuditState

LOGGER = logging.getLogger(__name__)


def load_static_audit_prompt(workspace_path: str) -> str:
    perception = build_perception_result(workspace_path=workspace_path, base_prompt_file=PROMPT_FILE)
    prompt_text = str(perception.get("system_prompt", "")).strip()
    if not prompt_text:
        LOGGER.error("Prompt file is empty or skill routing failed: %s", PROMPT_FILE)
        raise RuntimeError(f"Prompt file is empty or skill routing failed: {PROMPT_FILE}")
    return prompt_text


def perception_node(state: AuditState) -> AuditState:
    workspace_path = state.get("workspace_path", "").strip()
    if not workspace_path:
        return {
            "lsp_ready": False,
            "lsp_error": "Workspace path is empty",
            "perception_summary": "Perception failed: missing workspace path",
            "perception_meta": {
                "workspace_root": workspace_path,
                "lsp_ready": False,
                "lsp_error": "Workspace path is empty",
                "project_hints": {},
            },
            "audit_output": "Perception failed before planning: workspace path is required",
        }

    try:
        perception = build_perception_result(workspace_path=workspace_path, base_prompt_file=PROMPT_FILE)
    except Exception as exc:
        return {
            "lsp_ready": False,
            "lsp_error": str(exc),
            "perception_summary": f"Perception error: {exc}",
            "perception_meta": {
                "workspace_root": workspace_path,
                "lsp_ready": False,
                "lsp_error": str(exc),
                "project_hints": {},
            },
            "audit_output": f"Perception stage failed: {exc}",
        }

    prompt_text = str(perception.get("system_prompt", "")).strip()
    validation = perception.get("lsp_validation", {})
    checks = validation.get("checks", []) if isinstance(validation, dict) else []
    ready = bool(validation.get("ready", False)) if isinstance(validation, dict) else False
    errors = validation.get("errors", []) if isinstance(validation, dict) else []

    error_text = ""
    if errors:
        parts: list[str] = []
        for item in errors:
            language = str(item.get("language", "unknown"))
            server = str(item.get("server", ""))
            reason = str(item.get("reason", "unknown error"))
            parts.append(f"{language}/{server}: {reason}")
        error_text = "; ".join(parts)

    scores = perception.get("language_scores", {})
    skill_files = perception.get("skill_files", [])
    summary = (
        f"Language detection: {json.dumps(scores, ensure_ascii=False)}; "
        f"Skill files: {json.dumps(skill_files, ensure_ascii=False)}; "
        f"LSP checks: {json.dumps(checks, ensure_ascii=False)}"
    )

    result: AuditState = {
        "skill_prompt": prompt_text,
        "perception_summary": summary,
        "lsp_ready": ready,
        "perception_meta": {
            "workspace_root": workspace_path,
            "lsp_ready": ready,
            "lsp_error": error_text,
            "project_hints": {
                "language_scores": scores,
                "skill_files": skill_files,
                "lsp_checks": checks,
            },
        },
    }
    if not ready:
        result["lsp_error"] = error_text or "LSP service is unavailable"
        result["audit_output"] = f"Perception stage blocked execution: {result['lsp_error']}"
    return result
