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
from planner.state import build_perception_blocked_state, format_steps, read_text
from planner.types import AuditState

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
        if not bool(state.get("lsp_ready", False)):
            message = state.get("lsp_error", "").strip() or "LSP service is unavailable"
            return build_perception_blocked_state(message)

        request = state.get("user_request", "").strip() or DEFAULT_AUDIT_REQUEST
        workspace_path = state.get("workspace_path", "").strip()
        skill_prompt = state.get("skill_prompt", "")
        perception_meta = state.get("perception_meta", {})
        hint_text = ""
        if isinstance(perception_meta, dict):
            hints = perception_meta.get("project_hints", {})
            if hints:
                hint_text = f"\n\nPerception hints:\n{json.dumps(hints, ensure_ascii=False)}"
        context = render_txt(
            PROMPT_DIR / "workflow_context.txt",
            workspace_path=workspace_path,
            skill_prompt=skill_prompt,
            hint_text=hint_text,
        )

        result = planner.invoke(
            {
                "objective": request,
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
