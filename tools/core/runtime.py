import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ProjectContext:
    directory: Path = Path.cwd()

    @classmethod
    def resolve_path(cls, path: Optional[str]) -> Path:
        base = path or str(cls.directory)
        value = Path(base)
        if not value.is_absolute():
            value = cls.directory / value
        return value.resolve()


class Filesystem:
    @staticmethod
    def stat(file_path: str | Path) -> Optional[os.stat_result]:
        try:
            return os.stat(file_path)
        except OSError:
            return None


class WorkspaceGuard:
    @staticmethod
    def is_under_workspace(workspace_path: str | Path, target_path: str | Path) -> bool:
        workspace = os.path.abspath(str(workspace_path))
        target = os.path.abspath(str(target_path))
        try:
            return os.path.commonpath([workspace, target]) == workspace
        except Exception:
            return False

    @staticmethod
    def ensure_under_workspace(workspace_path: str | Path, target_path: str | Path) -> str:
        target = os.path.abspath(str(target_path))
        if not WorkspaceGuard.is_under_workspace(workspace_path, target):
            raise PermissionError(f"path '{target_path}' is outside workspace")
        return target


class AsyncProcess:
    @staticmethod
    async def spawn(
        cmd: List[str],
        cwd: Path | str,
        opts: Dict[str, Any] | None = None,
    ) -> Tuple[asyncio.subprocess.Process, asyncio.Future[int]]:
        options = opts or {}
        stdout = asyncio.subprocess.PIPE if options.get("stdout") == "pipe" else None
        stderr = asyncio.subprocess.PIPE if options.get("stderr") == "pipe" else None
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=stdout,
            stderr=stderr,
            cwd=str(cwd),
            env=os.environ.copy(),
        )
        abort_event = options.get("abort")
        if abort_event and isinstance(abort_event, asyncio.Event):
            async def watch_abort():
                await abort_event.wait()
                proc.terminate()
            asyncio.create_task(watch_abort())

        async def wait_exit():
            return await proc.wait()
        return proc, asyncio.ensure_future(wait_exit())

    @staticmethod
    async def read_stream(stream: asyncio.StreamReader) -> str:
        if not stream:
            return ""
        try:
            data = await stream.read()
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
