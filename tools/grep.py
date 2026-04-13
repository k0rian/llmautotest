import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator

from file.ripgrep import Ripgrep
from tools.core import AsyncProcess, Filesystem, ProjectContext, WorkspaceGuard


class GrepToolParams(BaseModel):
    """Validated params for grep search."""

    pattern: str = Field(description="The regex pattern to search for in file contents")
    path: Optional[str] = Field(
        default=None,
        description="The directory to search in. Defaults to the current workspace directory.",
    )
    include: Optional[str] = Field(
        default=None,
        description='File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")',
    )

    @field_validator("pattern")
    def validate_pattern(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pattern is required (cannot be empty)")
        return value


class GrepTool:
    """Ripgrep-backed grep tool for agent usage."""

    NAME = "grep"
    DESCRIPTION = """Search file contents using ripgrep (rg) - a fast regex search tool.
Supports regex patterns, directory scoping, and file pattern inclusion.
Exit codes: 0 = matches found, 1 = no matches, 2 = partial errors (e.g. inaccessible paths)."""
    MAX_LINE_LENGTH = 2000
    RESULT_LIMIT = 100

    @classmethod
    async def execute(cls, params: GrepToolParams, ctx: Any = None) -> Dict[str, Any]:
        await cls._request_permission(ctx, params)

        workspace_dir = cls._resolve_workspace_dir(ctx)
        search_path = cls._resolve_search_path(params.path, workspace_dir)
        WorkspaceGuard.ensure_under_workspace(workspace_dir, search_path)

        rg_path = await Ripgrep.filepath()
        rg_args = cls._build_rg_args(params, search_path)

        proc, exit_future = await AsyncProcess.spawn(
            [rg_path] + rg_args,
            cwd=workspace_dir,
            opts={
                "stdout": "pipe",
                "stderr": "pipe",
                "abort": getattr(ctx, "abort", None),
            },
        )

        stdout = await AsyncProcess.read_stream(proc.stdout)
        stderr = await AsyncProcess.read_stream(proc.stderr)
        exit_code = await exit_future

        cls._handle_exit_code(exit_code, stdout, stderr)

        matches = cls._parse_rg_output(stdout)
        return cls._format_result(matches, params.pattern, exit_code == 2)

    @classmethod
    async def _request_permission(cls, ctx: Any, params: GrepToolParams) -> None:
        if not ctx:
            return
        if hasattr(ctx, "ask"):
            await ctx.ask(
                {
                    "permission": "grep",
                    "patterns": [params.pattern],
                    "always": ["*"],
                    "metadata": {
                        "pattern": params.pattern,
                        "path": params.path,
                        "include": params.include,
                    },
                }
            )

    @classmethod
    def _resolve_workspace_dir(cls, ctx: Any) -> Path:
        if ctx is not None:
            for attr in ("workspace_path", "workspace", "directory", "cwd"):
                value = getattr(ctx, attr, None)
                if isinstance(value, str) and value.strip():
                    return Path(value).resolve()
            if isinstance(ctx, dict):
                for key in ("workspace_path", "workspace", "directory", "cwd"):
                    value = ctx.get(key)
                    if isinstance(value, str) and value.strip():
                        return Path(value).resolve()
        return Path(ProjectContext.directory).resolve()

    @classmethod
    def _resolve_search_path(cls, path: Optional[str], workspace_dir: Path) -> Path:
        if path and path.strip():
            value = Path(path)
            if not value.is_absolute():
                value = workspace_dir / value
            return value.resolve()
        return workspace_dir.resolve()

    @classmethod
    def _build_rg_args(cls, params: GrepToolParams, search_path: Path) -> List[str]:
        args = [
            "-nH",
            "--hidden",
            "--no-messages",
            "--field-match-separator=|",
            "--regexp",
            params.pattern,
        ]
        if params.include:
            args.extend(["--glob", params.include])
        args.append(str(search_path))
        return args

    @classmethod
    def _handle_exit_code(cls, exit_code: int, stdout: str, stderr: str) -> None:
        if exit_code == 1 or (exit_code == 2 and not stdout.strip()):
            return
        if exit_code not in (0, 1, 2):
            raise RuntimeError(f"ripgrep failed (exit code {exit_code}): {stderr}")

    @classmethod
    def _parse_rg_output(cls, stdout: str) -> List[Dict[str, Any]]:
        if not stdout:
            return []

        matches: List[Dict[str, Any]] = []
        lines = re.split(r"\r?\n", stdout.strip())
        for line in lines:
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue

            file_path, line_num_str, line_text = parts
            try:
                line_num = int(line_num_str)
            except ValueError:
                continue

            stats = Filesystem.stat(file_path)
            if not stats:
                continue

            matches.append(
                {
                    "path": file_path,
                    "modTime": stats.st_mtime * 1000,
                    "lineNum": line_num,
                    "lineText": line_text,
                }
            )

        matches.sort(key=lambda item: item["modTime"], reverse=True)
        return matches

    @classmethod
    def _format_result(cls, matches: List[Dict[str, Any]], pattern: str, has_errors: bool) -> Dict[str, Any]:
        if not matches:
            return {
                "title": pattern,
                "metadata": {"matches": 0, "truncated": False},
                "output": "No files found",
            }

        total_matches = len(matches)
        truncated = total_matches > cls.RESULT_LIMIT
        final_matches = matches[: cls.RESULT_LIMIT] if truncated else matches

        output_lines = [
            f"Found {total_matches} matches{f' (showing first {cls.RESULT_LIMIT})' if truncated else ''}"
        ]

        current_file = ""
        for match in final_matches:
            if match["path"] != current_file:
                if current_file:
                    output_lines.append("")
                current_file = match["path"]
                output_lines.append(f"{current_file}:")

            line_text = match["lineText"]
            if len(line_text) > cls.MAX_LINE_LENGTH:
                line_text = line_text[: cls.MAX_LINE_LENGTH] + "..."
            output_lines.append(f"  Line {match['lineNum']}: {line_text}")

        if truncated:
            output_lines.append("")
            output_lines.append(
                f"(Results truncated: showing {cls.RESULT_LIMIT} of {total_matches} matches. "
                "Consider using a more specific path or pattern.)"
            )

        if has_errors:
            output_lines.append("")
            output_lines.append("(Some paths were inaccessible and skipped)")

        return {
            "title": pattern,
            "metadata": {"matches": total_matches, "truncated": truncated},
            "output": "\n".join(output_lines),
        }


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


@tool
def grep_search(pattern: str, path: Optional[str] = None, include: Optional[str] = None) -> str:
    """Search file contents using ripgrep and return formatted results."""
    try:
        params = GrepToolParams(pattern=pattern, path=path, include=include)
        result = _run_async(GrepTool.execute(params=params, ctx=None))
        return result["output"]
    except Exception as exc:
        return f"Grep error: {exc}"


async def agent_grep_demo() -> None:
    """Simple local demo for manual grep tool testing."""

    class AgentContext:
        abort = asyncio.Event()

        async def ask(self, permission: Dict[str, Any]):
            print(f"[Agent Permission Check] {permission}")
            return True

    ctx = AgentContext()

    params = GrepToolParams(
        pattern=r"def\s+\w+",
        path="./",
        include="*.py",
    )

    try:
        result = await GrepTool.execute(params, ctx)
        print(f"Title: {result['title']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Output:\n{result['output']}")
    except Exception as exc:
        print(f"Grep failed: {exc}")
