import json
import logging
from pathlib import Path

from router.router import build_perception_result

from planner.policies import PROMPT_FILE
from planner.types import AuditState
from llm_utils import read_markdown

LOGGER = logging.getLogger(__name__)


def _infer_semantic_intent(request: str) -> bool:
    text = (request or "").strip().lower()
    if not text:
        return False
    keywords = (
        "semantic",
        "index",
        "diff",
        "coverage",
        "requirement",
        "prd",
        "spec",
        "语义",
        "索引",
        "比对",
        "对比",
        "覆盖率",
        "需求",
        "设计文档",
        "功能对齐",
    )
    return any(token in text for token in keywords)


def load_static_audit_prompt(workspace_path: str) -> str:
    _ = workspace_path
    prompt_text = read_markdown(Path(PROMPT_FILE))
    if not prompt_text:
        LOGGER.error("Prompt file is empty or skill routing failed: %s", PROMPT_FILE)
        raise RuntimeError(f"Prompt file is empty or skill routing failed: {PROMPT_FILE}")
    return prompt_text


def perception_node(state: AuditState) -> AuditState:
    workspace_path = state.get("workspace_path", "").strip()
    user_request = state.get("user_request", "").strip()
    semantic_intent = _infer_semantic_intent(user_request)
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
        perception = build_perception_result(workspace_path=workspace_path)
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
        f"LSP checks: {json.dumps(checks, ensure_ascii=False)}; "
        f"Semantic intent: {semantic_intent}; "
        f"LSP fallback active: {not ready}"
    )

    result: AuditState = {
        "skill_prompt": load_static_audit_prompt(workspace_path),
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
                "system_prompt_injected": False,
                "semantic_intent": semantic_intent,
                "semantic_fallback": not ready,
                "execution_mode": "semantic_fallback" if not ready else "standard",
            },
        },
    }
    if not ready:
        result["lsp_error"] = error_text or "LSP service is unavailable"
        result["perception_summary"] = (
            f"{summary}; "
            "Switching to semantic-index + file-search workflow because LSP is unavailable."
        )
    return result
