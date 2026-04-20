import json
from pathlib import Path
from typing import Any

from tools.code_index import build_hierarchical_code_index
from tools.semantic_diff_ts import DEFAULT_INCLUDE_GLOB, semantic_index_functions


def _resolve_build_target(workspace_path: str, target_path: str) -> str:
    workspace = Path(workspace_path).resolve()
    target = Path(target_path or ".")
    if not target.is_absolute():
        target = workspace / target
    return str(target.resolve())


def _parse_tool_json(raw: str, tool_name: str) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"{tool_name} returned non-json result: {text}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{tool_name} returned unexpected payload: {text}")
    if payload.get("status") != "ok":
        raise RuntimeError(f"{tool_name} failed: {text}")
    return payload


def build_semantic_indexes(
    workspace_path: str,
    target_path: str,
    *,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> dict[str, Any]:
    resolved_target = _resolve_build_target(workspace_path=workspace_path, target_path=target_path)
    function_raw = semantic_index_functions.func(
        path=resolved_target,
        include_glob=include_glob,
        max_files=max(1, int(max_files)),
        rebuild=bool(rebuild),
        use_llm_summary=bool(use_llm_summary),
        summary_model_name=summary_model_name,
    )
    function_index = _parse_tool_json(function_raw, "semantic_index_functions")

    hierarchical_raw = build_hierarchical_code_index.func(
        path=resolved_target,
        rebuild=bool(rebuild),
        use_llm=bool(use_llm_summary),
        summary_model_name=summary_model_name,
        include_glob=include_glob,
        max_files=max(1, int(max_files)),
    )
    hierarchical_index = _parse_tool_json(hierarchical_raw, "build_hierarchical_code_index")

    return {
        "status": "ok",
        "target_path": resolved_target,
        "summary_mode": "llm" if use_llm_summary else "deterministic",
        "summary_model": summary_model_name if use_llm_summary else "",
        "function_index": function_index,
        "hierarchical_index": hierarchical_index,
    }
