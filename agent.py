import argparse
import json
from pathlib import Path
from typing import Any, TypedDict
import traceback
import logging
from llm_utils.config_loader import load_api_key
from router.router import build_perception_result
import sys
import os
import asyncio
from langchain_core.callbacks import BaseCallbackHandler

RUNTIME_IMPORT_ERROR: Exception | None = None

try:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from tools.tools import build_tools
except ModuleNotFoundError as exc:
    print(traceback.format_exc())
    print(exc)
    sys.exit(-1)

os.environ["PYTHONUTF8"] = "1"
DEFAULT_MODEL_NAME = "doubao-seed-1-6-lite-251015"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_AUDIT_REQUEST = "请对当前工作区执行静态代码审计，重点关注 bug、性能问题和可优化点。"
PROMPT_FILE = Path(__file__).resolve().parent / "PROMPT.md"
TYPEWRITER_DELAY = 0.01
LOGGER = logging.getLogger(__name__)


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


class TypewriterStreamHandler(BaseCallbackHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, delay: float = TYPEWRITER_DELAY):
        super().__init__()
        self._loop = loop
        self._delay = max(0.0, delay)
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        await self._worker
        self._worker = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        text = token or ""
        if not text:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    async def _run(self) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            for char in chunk:
                print(char, end="", flush=True)
                if self._delay > 0:
                    await asyncio.sleep(self._delay)


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
) -> Any:
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


def _read_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "messages" in value and value["messages"]:
            return _read_text(value["messages"][-1])
        if "output" in value:
            return _read_text(value["output"])
        return json.dumps(value, ensure_ascii=False, indent=2)
    content = getattr(value, "content", None)
    if content is not None:
        return _read_text(content)
    return str(value)



def load_static_audit_prompt(workspace_path: str) -> str:
    perception = build_perception_result(workspace_path=workspace_path, base_prompt_file=PROMPT_FILE)
    prompt_text = str(perception.get("system_prompt", "")).strip()
    if not prompt_text:
        LOGGER.error("Prompt 文件为空或技能路由失败: %s", PROMPT_FILE)
        raise RuntimeError(f"Prompt 文件为空或技能路由失败: {PROMPT_FILE}")
    return prompt_text


def _ensure_runtime_dependencies() -> None:
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "缺少运行依赖，请先安装 requirements.txt 中的依赖后再运行 CLI agent"
        ) from RUNTIME_IMPORT_ERROR


def build_executor(model: Any | None = None):
    _ensure_runtime_dependencies()
    active_model = model or build_model()
    return create_agent(
        active_model,
        tools=build_static_audit_tools(),
        # verbose=True,
    )


def build_audit_planner(model: Any | None = None, executor: Any | None = None):
    _ensure_runtime_dependencies()
    from planner.planner import build_plan_and_execute_planner

    active_model = model or build_model()
    active_executor = executor or build_executor(active_model)
    return build_plan_and_execute_planner(
        planner_model=active_model,
        executor_agent=active_executor,
        gui_executor=None,
        verbose=True,
        max_replans=1,
        gui_max_steps=0,
    )


def _format_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        title = _read_text(step.get("title", f"步骤 {index}")).strip() or f"步骤 {index}"
        mode = _read_text(step.get("mode", "code_audit")).strip() or "code_audit"
        objective = _read_text(step.get("objective", "")).strip()
        lines.append(f"{index}. [{mode}] {title}")
        if objective:
            lines.append(f"   - {objective}")
    return "\n".join(lines)


def perception_node(state: AuditState) -> AuditState:
    workspace_path = state.get("workspace_path", "").strip()
    if not workspace_path:
        return {
            "lsp_ready": False,
            "lsp_error": "工作区路径为空，无法执行感知阶段",
            "perception_summary": "感知失败：缺少工作区路径",
            "audit_output": "感知阶段失败：工作区路径为空，无法校验技能与 LSP 服务",
        }
    try:
        perception = build_perception_result(workspace_path=workspace_path, base_prompt_file=PROMPT_FILE)
    except Exception as exc:
        return {
            "lsp_ready": False,
            "lsp_error": str(exc),
            "perception_summary": f"感知阶段异常：{str(exc)}",
            "audit_output": f"感知阶段失败：{str(exc)}",
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
        error_text = "；".join(parts)
    scores = perception.get("language_scores", {})
    skill_files = perception.get("skill_files", [])
    summary = (
        f"语言识别：{json.dumps(scores, ensure_ascii=False)}；"
        f"技能文件：{json.dumps(skill_files, ensure_ascii=False)}；"
        f"LSP校验：{json.dumps(checks, ensure_ascii=False)}"
    )
    result: AuditState = {
        "skill_prompt": prompt_text,
        "perception_summary": summary,
        "lsp_ready": ready,
    }
    if not ready:
        result["lsp_error"] = error_text or "LSP 服务不可用"
        result["audit_output"] = f"感知阶段失败：{result['lsp_error']}"
    return result


def planner_execute_node_factory(planner: Any):
    def planner_execute_node(state: AuditState) -> AuditState:
        if not bool(state.get("lsp_ready", False)):
            message = state.get("lsp_error", "").strip() or "LSP 服务不可用，终止审计执行"
            return {
                "planner_summary": "已在感知阶段终止",
                "plan": "未进入审计执行阶段",
                "audit_output": f"感知阶段拦截：{message}",
            }
        request = state.get("user_request", "").strip() or DEFAULT_AUDIT_REQUEST
        workspace_path = state.get("workspace_path", "").strip()
        skill_prompt = state.get("skill_prompt", "")
        context = (
            f"工作区路径：{workspace_path}\n"
            "当前 agent 仅执行静态代码审计，不包含 GUI 测试。\n"
            "请仅使用静态审计相关工具完成任务。\n\n"
            f"执行提示词：\n{skill_prompt}"
        )
        result = planner.invoke(
            {
                "objective": request,
                "context": context,
                "max_steps": 5,
            }
        )
        summary = _read_text(result.get("summary", "")).strip()
        steps_text = _format_steps(result.get("steps", []))
        final_response = _read_text(result.get("final_response", "")).strip()
        return {
            "planner_summary": summary,
            "plan": steps_text,
            "audit_output": final_response,
        }

    return planner_execute_node


def finalizer_node(state: AuditState) -> AuditState:
    request = state.get("user_request", "").strip() or DEFAULT_AUDIT_REQUEST
    workspace_path = state.get("workspace_path", "").strip()
    planner_summary = state.get("planner_summary", "").strip()
    plan = state.get("plan", "").strip()
    perception_summary = state.get("perception_summary", "").strip()
    audit_output = state.get("audit_output", "").strip()
    final_output = (
        f"审计任务: {request}\n"
        f"工作区: {workspace_path}\n\n"
        f"感知结果:\n{perception_summary or '无'}\n\n"
        f"规划摘要:\n{planner_summary or '无'}\n\n"
        f"执行计划:\n{plan}\n\n"
        f"审计结论:\n{audit_output}"
    )
    return {"final_output": final_output}


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


async def _typewriter_print(text: str, delay: float = TYPEWRITER_DELAY) -> None:
    for char in text:
        print(char, end="", flush=True)
        if delay > 0:
            await asyncio.sleep(delay)


async def _run_audit_cli_async(
    user_request: str,
    workspace_path: str,
    model: Any | None = None,
) -> str:
    app = build_cli_graph(model=model)
    loop = asyncio.get_running_loop()
    stream_handler = TypewriterStreamHandler(loop=loop, delay=TYPEWRITER_DELAY)
    payload = {
        "user_request": user_request.strip() or DEFAULT_AUDIT_REQUEST,
        "workspace_path": str(Path(workspace_path).resolve()),
    }
    await stream_handler.start()
    result: dict[str, Any] = {}
    try:
        result = await app.ainvoke(payload, config={"callbacks": [stream_handler]})
    finally:
        await stream_handler.stop()
    final_output = _read_text(result.get("final_output", "")).strip()
    if final_output:
        print()
        await _typewriter_print(f"{final_output}\n", delay=TYPEWRITER_DELAY)
    return final_output


def run_audit_cli(
    user_request: str,
    workspace_path: str,
    model: Any | None = None,
) -> str:
    return asyncio.run(_run_audit_cli_async(user_request, workspace_path, model=model))


def _interactive_loop(workspace_path: str) -> None:
    while True:
        raw = input("\n请输入审计任务，直接回车使用默认任务，输入 exit 退出:\n> ").strip()
        if raw.lower() in {"exit", "quit"}:
            break
        request = raw or DEFAULT_AUDIT_REQUEST
        print()
        try:
            run_audit_cli(request, workspace_path)
        except RuntimeError as exc:
            print(f"CLI 运行失败: {exc}")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="静态代码审计 CLI Agent")
    parser.add_argument("task", nargs="?", default=DEFAULT_AUDIT_REQUEST, help="审计任务描述")
    parser.add_argument("--path", default=".", help="要审计的工作区路径")
    parser.add_argument("--interactive", action="store_true", help="启动交互式命令行模式")
    args = parser.parse_args()

    workspace_path = str(Path(args.path).resolve())
    if args.interactive:
        _interactive_loop(workspace_path)
        return
    try:
        run_audit_cli(args.task, workspace_path)
    except RuntimeError as exc:
        print(f"CLI 运行失败: {exc}")


if __name__ == "__main__":
    main()





