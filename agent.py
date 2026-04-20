import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from planner.policies import DEFAULT_AUDIT_REQUEST
from planner.state import read_text
from planner.workflow import build_cli_graph
from llm_utils.config_loader import load_llm_summary_enabled
from build.index_builder import build_semantic_indexes
from tools.semantic_diff_ts import DEFAULT_INCLUDE_GLOB

os.environ["PYTHONUTF8"] = "1"
TYPEWRITER_DELAY = 0.01


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
        if text:
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

    final_output = read_text(result.get("final_output", "")).strip()
    if final_output:
        print()
        await _typewriter_print(f"{final_output}\n", delay=TYPEWRITER_DELAY)
    return final_output


def run_audit_cli(
    user_request: str,
    workspace_path: str,
    model: Any | None = None,
) -> str:
    return asyncio.run(
        _run_audit_cli_async(
            user_request,
            workspace_path,
            model=model,
        )
    )


def _print_cli_header(workspace_path: str) -> None:
    print()
    print("=" * 68)
    print("LLM AutoTest CLI  |  Static Audit + Tool Orchestration")
    print("-" * 68)
    print(f"Workspace : {workspace_path}")
    print("=" * 68)


def _interactive_loop(workspace_path: str) -> None:
    while True:
        raw = input("\nTask (press Enter to use default, type exit to quit)\n> ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("\nSession ended.")
            break
        request = raw or DEFAULT_AUDIT_REQUEST
        print("\n[Running] Executing audit workflow...\n")
        try:
            run_audit_cli(
                request,
                workspace_path,
            )
        except RuntimeError as exc:
            print(f"[Error] CLI execution failed: {exc}")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Static code audit CLI agent")
    parser.add_argument("task", nargs="?", default=DEFAULT_AUDIT_REQUEST, help="Audit task description")
    parser.add_argument("--path", default=".", help="Workspace path")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument(
        "--build-index",
        default="",
        help="Build semantic indexes for the specified file or directory, then exit without running audit",
    )
    parser.add_argument(
        "--include-glob",
        default="",
        help="Optional comma-separated include glob for --build-index",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=2000,
        help="Maximum files to scan when using --build-index",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild when using --build-index",
    )
    parser.add_argument(
        "--llm-summary",
        dest="semantic_use_llm_summary",
        action="store_true",
        default=None,
        help="Use LLM-generated function summaries when building semantic indexes with --build-index",
    )
    parser.add_argument(
        "--no-llm-summary",
        dest="semantic_use_llm_summary",
        action="store_false",
        help="Disable LLM-generated function summaries for --build-index even if enabled in config.yml",
    )
    parser.add_argument(
        "--summary-model",
        default="",
        help="Override LLM.model_name for semantic summary generation when using --build-index",
    )
    args = parser.parse_args()

    workspace_path = str(Path(args.path).resolve())
    _print_cli_header(workspace_path)
    if args.build_index.strip():
        use_llm_summary = (
            load_llm_summary_enabled(False)
            if args.semantic_use_llm_summary is None
            else bool(args.semantic_use_llm_summary)
        )
        print("\n[Build Index] Building semantic indexes...\n")
        try:
            payload = build_semantic_indexes(
                workspace_path=workspace_path,
                target_path=args.build_index.strip(),
                include_glob=args.include_glob.strip() or DEFAULT_INCLUDE_GLOB,
                max_files=max(1, int(args.max_files)),
                rebuild=bool(args.rebuild_index),
                use_llm_summary=use_llm_summary,
                summary_model_name=args.summary_model.strip(),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[Error] Index build failed: {exc}")
        return

    if args.interactive:
        if args.semantic_use_llm_summary is not None or args.summary_model.strip():
            print("\n[Info] --llm-summary/--summary-model are only used with --build-index and will be ignored for audit.")
        _interactive_loop(workspace_path)
        return

    print("\n[Running] Executing audit workflow...\n")
    if args.semantic_use_llm_summary is not None or args.summary_model.strip():
        print("[Info] --llm-summary/--summary-model are only used with --build-index and will be ignored for audit.\n")
    try:
        run_audit_cli(
            args.task,
            workspace_path,
        )
    except RuntimeError as exc:
        print(f"[Error] CLI execution failed: {exc}")


if __name__ == "__main__":
    main()
