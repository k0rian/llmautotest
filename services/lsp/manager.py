import os
import threading
from pathlib import Path
from typing import Any
from tools.core import WorkspaceGuard
from .lsp import LSPSession


_SESSIONS: dict[str, LSPSession] = {}
_SESSIONS_LOCK = threading.Lock()


def to_uri(file_path: str) -> str:
    return Path(os.path.abspath(file_path)).as_uri()


def get_session(session_id: str) -> LSPSession:
    key = session_id.strip()
    if not key:
        raise ValueError("session_id cannot be empty")
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            raise ValueError(f"LSP session not found: {key}")
        return _SESSIONS[key]


def ensure_workspace_file(sess: LSPSession, file_path: str) -> str:
    absolute = os.path.abspath(file_path)
    if not os.path.isfile(absolute):
        raise ValueError(f"file not found '{file_path}'")
    return WorkspaceGuard.ensure_under_workspace(sess.workspace_path, absolute)


def start_session(
    session_id: str,
    command: str,
    workspace_path: str,
    initialization_options: dict[str, Any] | None = None,
    trace: str = "off",
) -> dict[str, Any]:
    key = session_id.strip()
    if not key:
        raise ValueError("session_id cannot be empty")
    if not command.strip():
        raise ValueError("command cannot be empty")
    if not os.path.isdir(workspace_path):
        raise ValueError(f"invalid workspace path '{workspace_path}'")
    with _SESSIONS_LOCK:
        if key in _SESSIONS and _SESSIONS[key].is_alive():
            return {"session_id": key, "status": "already_running"}
        if key in _SESSIONS:
            try:
                _SESSIONS[key].stop()
            except Exception:
                pass
        sess = LSPSession(key, command, workspace_path)
        _SESSIONS[key] = sess
    sess.start()
    init_options = initialization_options if isinstance(initialization_options, dict) else {}
    result = sess.initialize(initialization_options=init_options, trace=trace)
    return {
        "session_id": key,
        "status": "started",
        "workspace_path": os.path.abspath(workspace_path),
        "initialize_result": result,
    }


def stop_session(session_id: str) -> dict[str, Any]:
    key = session_id.strip()
    with _SESSIONS_LOCK:
        sess = _SESSIONS.pop(key, None)
    if not sess:
        return {"session_id": key, "status": "not_found"}
    sess.stop()
    return {"session_id": key, "status": "stopped"}


def list_sessions() -> list[dict[str, Any]]:
    with _SESSIONS_LOCK:
        return [
            {
                "session_id": sid,
                "alive": sess.is_alive(),
                "workspace_path": sess.workspace_path,
                "command": sess.command,
            }
            for sid, sess in _SESSIONS.items()
        ]
