import json
import traceback
from pathlib import Path
from typing import Any

from llm_utils.config_loader import load_api_key
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


def build_model(model_name: str = DEFAULT_MODEL_NAME, base_url: str = DEFAULT_BASE_URL) -> Any:
    _ensure_runtime_dependencies()
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
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
        hint_text = ""
        fallback_guidance = ""
        semantic_intent_guidance = ""
        if isinstance(perception_meta, dict):
            hints = perception_meta.get("project_hints", {})
            if isinstance(hints, dict):
                semantic_intent = bool(hints.get("semantic_intent", False))
                semantic_fallback = bool(hints.get("semantic_fallback", semantic_fallback))
            if hints:
                hint_text = f"\n\nPerception hints:\n{json.dumps(hints, ensure_ascii=False)}"

        if semantic_fallback:
            fallback_reason = lsp_error or "LSP service is unavailable"
            fallback_guidance = (
                "\n\nExecution fallback:\n"
                "- LSP is unavailable, do not rely on LSP tools.\n"
                "- Must include a semantic_diff step that runs semantic_index_functions + semantic_diff_with_description.\n"
                "- Must include at least one code_audit step focused on file_search/grep_search/read_file evidence collection.\n"
                f"- Fallback reason: {fallback_reason}"
            )
        if semantic_intent:
            semantic_intent_guidance = (
                "\n\nSemantic-intent directive:\n"
                "- The request appears to require semantic index based analysis.\n"
                "- Plan must include semantic_diff mode early and use it as a primary evidence source."
            )

        effective_request = request
        if semantic_fallback or semantic_intent:
            effective_request = (
                f"{request}\n\n"
                "[Planner directives]\n"
                f"{'1) Must include semantic_diff step.\n' if (semantic_fallback or semantic_intent) else ''}"
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
) -> str:
    app = build_cli_graph(model=model)
    payload = {
        "user_request": user_request.strip() or DEFAULT_AUDIT_REQUEST,
        "workspace_path": str(Path(workspace_path).resolve()),
    }
    result: dict[str, Any] = await app.ainvoke(payload)
    return read_text(result.get("final_output", "")).strip()


def run_audit_cli(user_request: str, workspace_path: str, model: Any | None = None) -> str:
    import asyncio

    return asyncio.run(run_audit_cli_async(user_request, workspace_path, model=model))


def runtime_import_traceback() -> str:
    if RUNTIME_IMPORT_ERROR is None:
        return ""
    return traceback.format_exc()
