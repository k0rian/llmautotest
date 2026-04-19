import json
import traceback
from pathlib import Path
from typing import Any

from llm_utils.config_loader import load_api_key, load_base_url, load_llm_summary_enabled, load_model_name
from planner.finalizer import finalizer_node
from planner.perception import perception_node
from planner.planner import build_plan_and_execute_planner
from planner.prompt.loader import render_txt
from planner.policies import (
    DEFAULT_AUDIT_REQUEST,
    DEFAULT_BASE_URL,
    DEFAULT_GUI_MAX_STEPS,
    DEFAULT_MAX_REPLANS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_NAME,
)
from planner.state import format_steps, read_text
from planner.types import AuditState
from tools.core import ProjectContext

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"

RUNTIME_IMPORT_ERROR: Exception | None = None
try:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from tools.tools import build_tools
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc


def _ensure_runtime_dependencies() -> None:
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependencies. Please install requirements.txt before running CLI agent."
        ) from RUNTIME_IMPORT_ERROR


def build_model(model_name: str = "", base_url: str = "") -> Any:
    _ensure_runtime_dependencies()
    return ChatOpenAI(
        model=model_name or load_model_name(DEFAULT_MODEL_NAME),
        base_url=base_url or load_base_url() or DEFAULT_BASE_URL,
        api_key=load_api_key(),
        streaming=True,
    )


def build_static_audit_tools() -> list[Any]:
    _ensure_runtime_dependencies()
    return build_tools()


def build_executor(model: Any | None = None):
    _ensure_runtime_dependencies()
    active_model = model or build_model()
    return create_agent(active_model, tools=build_static_audit_tools())


def build_audit_planner(model: Any | None = None, executor: Any | None = None):
    _ensure_runtime_dependencies()
    active_model = model or build_model()
    active_executor = executor or build_executor(active_model)
    return build_plan_and_execute_planner(
        planner_model=active_model,
        executor_agent=active_executor,
        gui_executor=None,
        verbose=True,
        max_replans=DEFAULT_MAX_REPLANS,
        gui_max_steps=DEFAULT_GUI_MAX_STEPS,
    )


def planner_execute_node_factory(planner: Any):
    def planner_execute_node(state: AuditState) -> AuditState:
        request = state.get("user_request", "").strip() or DEFAULT_AUDIT_REQUEST
        workspace_path = state.get("workspace_path", "").strip()
        if not workspace_path:
            return {
                "planner_summary": "Stopped before planning",
                "plan": "Execution stage was skipped",
                "audit_output": "Workspace path is empty",
                "planner_runtime": {},
            }
        ProjectContext.directory = Path(workspace_path).resolve()

        skill_prompt = state.get("skill_prompt", "")
        perception_meta = state.get("perception_meta", {})
        lsp_ready = bool(state.get("lsp_ready", False))
        lsp_error = state.get("lsp_error", "").strip()

        semantic_intent = False
        semantic_fallback = not lsp_ready
        semantic_target_hint = ""
        hint_text = ""
        fallback_guidance = ""
        semantic_intent_guidance = ""
        if isinstance(perception_meta, dict):
            hints = perception_meta.get("project_hints", {})
            if isinstance(hints, dict):
                semantic_intent = bool(hints.get("semantic_intent", False))
                semantic_fallback = bool(hints.get("semantic_fallback", semantic_fallback))
                semantic_target_hint = read_text(hints.get("semantic_target_hint", "")).strip()
            if hints:
                hint_text = f"\n\nPerception hints:\n{json.dumps(hints, ensure_ascii=False)}"
        if not semantic_target_hint:
            semantic_target_hint = _extract_target_hint(request)

        if semantic_fallback:
            fallback_reason = lsp_error or "LSP service is unavailable"
            fallback_guidance = (
                "\n\nExecution fallback:\n"
                "- LSP is unavailable, do not rely on LSP tools.\n"
                "- Must include semantic pipeline steps: semantic_index -> semantic_localize -> semantic_retrieve -> semantic_validate.\n"
                "- Must include at least one code_audit step focused on file_search/grep_search/read_file evidence collection.\n"
                f"- Fallback reason: {fallback_reason}"
            )
        if semantic_intent:
            semantic_intent_guidance = (
                "\n\nSemantic-intent directive:\n"
                "- The request appears to require semantic index based analysis.\n"
                "- Plan must include semantic pipeline modes early and use them as primary evidence sources."
            )

        effective_request = request
        if semantic_fallback or semantic_intent:
            effective_request = (
                f"{request}\n\n"
                "[Planner directives]\n"
                f"{'1) Must include semantic_index + semantic_localize + semantic_retrieve + semantic_validate steps.\n' if (semantic_fallback or semantic_intent) else ''}"
                f"{'2) Must include file search based code_audit step.\n' if semantic_fallback else ''}"
            )

        context = render_txt(
            PROMPT_DIR / "workflow_context.txt",
            workspace_path=workspace_path,
            skill_prompt=skill_prompt,
            hint_text=hint_text,
            fallback_guidance=fallback_guidance,
            semantic_intent_guidance=semantic_intent_guidance,
        )

        result = planner.invoke(
            {
                "objective": effective_request,
                "context": context,
                "max_steps": DEFAULT_MAX_STEPS,
                "workspace_path": workspace_path,
                "semantic_required": bool(semantic_fallback or semantic_intent),
                "semantic_target_hint": semantic_target_hint,
                "semantic_use_llm_summary": bool(state.get("semantic_use_llm_summary", load_llm_summary_enabled(False))),
                "semantic_summary_model": read_text(state.get("semantic_summary_model", "")).strip(),
            }
        )
        summary = read_text(result.get("summary", "")).strip()
        steps_text = format_steps(result.get("steps", []))
        final_response = read_text(result.get("final_response", "")).strip()
        runtime_state = result.get("runtime_state")
        return {
            "planner_summary": summary,
            "plan": steps_text,
            "audit_output": final_response,
            "planner_runtime": runtime_state if isinstance(runtime_state, dict) else {},
        }

    return planner_execute_node


def build_cli_graph(model: Any | None = None, executor: Any | None = None):
    _ensure_runtime_dependencies()
    active_model = model or build_model()
    active_executor = executor or build_executor(active_model)
    planner = build_audit_planner(model=active_model, executor=active_executor)
    graph = StateGraph(AuditState)
    graph.add_node("perceive", perception_node)
    graph.add_node("plan_execute", planner_execute_node_factory(planner))
    graph.add_node("finalize", finalizer_node)
    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "plan_execute")
    graph.add_edge("plan_execute", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_audit_cli_async(
    user_request: str,
    workspace_path: str,
    model: Any | None = None,
    semantic_use_llm_summary: bool | None = None,
    semantic_summary_model: str = "",
) -> str:
    app = build_cli_graph(model=model)
    use_llm_summary = load_llm_summary_enabled(False) if semantic_use_llm_summary is None else bool(semantic_use_llm_summary)
    payload = {
        "user_request": user_request.strip() or DEFAULT_AUDIT_REQUEST,
        "workspace_path": str(Path(workspace_path).resolve()),
        "semantic_use_llm_summary": use_llm_summary,
        "semantic_summary_model": semantic_summary_model.strip(),
    }
    result: dict[str, Any] = await app.ainvoke(payload)
    return read_text(result.get("final_output", "")).strip()


def run_audit_cli(
    user_request: str,
    workspace_path: str,
    model: Any | None = None,
    semantic_use_llm_summary: bool | None = None,
    semantic_summary_model: str = "",
) -> str:
    import asyncio

    return asyncio.run(
        run_audit_cli_async(
            user_request,
            workspace_path,
            model=model,
            semantic_use_llm_summary=semantic_use_llm_summary,
            semantic_summary_model=semantic_summary_model,
        )
    )


def runtime_import_traceback() -> str:
    if RUNTIME_IMPORT_ERROR is None:
        return ""
    return traceback.format_exc()


def _extract_target_hint(request: str) -> str:
    text = (request or "").strip()
    if not text:
        return ""
    import re

    match = re.search(r"([A-Za-z0-9_\-./\\]+[/\\])", text)
    if not match:
        return ""
    token = match.group(1).strip().strip("`'\"")
    return token
