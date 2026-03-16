import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from langchain.tools import tool


LANGUAGE_ID_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".toml": "toml",
    ".md": "markdown",
}


def _to_uri(file_path: str) -> str:
    return Path(os.path.abspath(file_path)).as_uri()


def _language_id(file_path: str, language_id: str) -> str:
    if language_id and language_id.strip():
        return language_id.strip()
    suffix = Path(file_path).suffix.lower()
    return LANGUAGE_ID_MAP.get(suffix, "plaintext")


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_json(value: str, fallback: Any) -> Any:
    if not value or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


class LSPSession:
    def __init__(self, session_id: str, command: str, workspace_path: str):
        self.session_id = session_id
        self.command = command
        self.workspace_path = os.path.abspath(workspace_path)
        self.process: subprocess.Popen | None = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_request_id = 1
        self._pending: dict[int, dict[str, Any]] = {}
        self._notifications: list[dict[str, Any]] = []
        self._diagnostics: dict[str, Any] = {}
        self._stderr_lines: list[str] = []
        self._doc_versions: dict[str, int] = {}
        self._running = False
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        cmd_parts = shlex.split(self.command, posix=False)
        self.process = subprocess.Popen(
            cmd_parts,
            cwd=self.workspace_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._running = True
        self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self) -> None:
        if not self.process:
            return
        if self._running:
            try:
                self.request("shutdown", {}, timeout_seconds=5)
            except Exception:
                pass
            try:
                self.notify("exit", {})
            except Exception:
                pass
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self._running = False

    def is_alive(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    def initialize(self, initialization_options: Any | None = None, trace: str = "off") -> Any:
        options = initialization_options if isinstance(initialization_options, dict) else {}
        result = self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "clientInfo": {"name": "llmautotest-agent", "version": "1.0"},
                "rootUri": Path(self.workspace_path).as_uri(),
                "workspaceFolders": [
                    {
                        "name": Path(self.workspace_path).name,
                        "uri": Path(self.workspace_path).as_uri(),
                    }
                ],
                "capabilities": {
                    "workspace": {
                        "applyEdit": True,
                        "workspaceEdit": {"documentChanges": True},
                        "didChangeConfiguration": {"dynamicRegistration": True},
                        "didChangeWatchedFiles": {"dynamicRegistration": True},
                        "symbol": {"dynamicRegistration": True},
                        "executeCommand": {"dynamicRegistration": True},
                    },
                    "textDocument": {
                        "synchronization": {
                            "didSave": True,
                            "didClose": True,
                            "willSave": False,
                            "willSaveWaitUntil": False,
                        },
                        "hover": {"dynamicRegistration": True},
                        "definition": {"dynamicRegistration": True},
                        "references": {"dynamicRegistration": True},
                        "rename": {"dynamicRegistration": True, "prepareSupport": True},
                        "codeAction": {
                            "dynamicRegistration": True,
                            "codeActionLiteralSupport": {
                                "codeActionKind": {
                                    "valueSet": [
                                        "",
                                        "quickfix",
                                        "refactor",
                                        "refactor.extract",
                                        "refactor.inline",
                                        "refactor.rewrite",
                                        "source",
                                        "source.organizeImports",
                                    ]
                                }
                            },
                        },
                        "documentSymbol": {
                            "dynamicRegistration": True,
                            "hierarchicalDocumentSymbolSupport": True,
                        },
                        "formatting": {"dynamicRegistration": True},
                        "publishDiagnostics": {"relatedInformation": True},
                    },
                },
                "trace": trace,
                "initializationOptions": options,
            },
            timeout_seconds=20,
        )
        self.notify("initialized", {})
        return result

    def request(self, method: str, params: dict[str, Any], timeout_seconds: int = 20) -> Any:
        with self._state_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            waiter = {"event": threading.Event(), "response": None}
            self._pending[request_id] = waiter
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        ok = waiter["event"].wait(timeout_seconds)
        if not ok:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"LSP request timeout: {method}")
        response = waiter["response"]
        if isinstance(response, dict) and "error" in response:
            return {"error": response["error"]}
        return response.get("result") if isinstance(response, dict) else response

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def did_open(self, file_path: str, language_id: str = "", text: str | None = None) -> dict[str, Any]:
        absolute = os.path.abspath(file_path)
        version = self._doc_versions.get(absolute, 0) + 1
        self._doc_versions[absolute] = version
        content = text
        if content is None:
            with open(absolute, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        uri = _to_uri(absolute)
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": _language_id(absolute, language_id),
                    "version": version,
                    "text": content,
                }
            },
        )
        return {"uri": uri, "version": version}

    def did_change(self, file_path: str, text: str) -> dict[str, Any]:
        absolute = os.path.abspath(file_path)
        version = self._doc_versions.get(absolute, 0) + 1
        self._doc_versions[absolute] = version
        uri = _to_uri(absolute)
        self.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )
        return {"uri": uri, "version": version}

    def did_close(self, file_path: str) -> dict[str, Any]:
        absolute = os.path.abspath(file_path)
        uri = _to_uri(absolute)
        self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        self._doc_versions.pop(absolute, None)
        return {"uri": uri}

    def diagnostics(self, file_path: str = "") -> Any:
        if file_path and file_path.strip():
            return self._diagnostics.get(_to_uri(file_path), [])
        return self._diagnostics

    def notifications(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return self._notifications[-limit:]

    def stderr_output(self, limit: int = 50) -> list[str]:
        if limit <= 0:
            return []
        return self._stderr_lines[-limit:]

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("LSP process is not running")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            self.process.stdin.write(header)
            self.process.stdin.write(body)
            self.process.stdin.flush()

    def _stdout_loop(self) -> None:
        if not self.process or not self.process.stdout:
            return
        stream = self.process.stdout
        while True:
            try:
                headers = {}
                while True:
                    line = stream.readline()
                    if not line:
                        return
                    if line in (b"\r\n", b"\n"):
                        break
                    text = line.decode("ascii", errors="ignore").strip()
                    if ":" in text:
                        key, value = text.split(":", 1)
                        headers[key.strip().lower()] = value.strip()
                content_length = int(headers.get("content-length", "0"))
                if content_length <= 0:
                    continue
                payload = stream.read(content_length)
                if not payload:
                    return
                message = json.loads(payload.decode("utf-8", errors="ignore"))
                self._handle_message(message)
            except Exception:
                return

    def _stderr_loop(self) -> None:
        if not self.process or not self.process.stderr:
            return
        for raw in iter(self.process.stderr.readline, b""):
            line = raw.decode("utf-8", errors="ignore").rstrip("\n")
            with self._state_lock:
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > 200:
                    self._stderr_lines = self._stderr_lines[-200:]

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            response_id = message.get("id")
            with self._state_lock:
                waiter = self._pending.pop(response_id, None)
            if waiter:
                waiter["response"] = message
                waiter["event"].set()
            return
        method = message.get("method")
        if method == "textDocument/publishDiagnostics":
            params = message.get("params", {})
            uri = params.get("uri")
            diagnostics = params.get("diagnostics", [])
            if uri:
                with self._state_lock:
                    self._diagnostics[uri] = diagnostics
        with self._state_lock:
            self._notifications.append(message)
            if len(self._notifications) > 500:
                self._notifications = self._notifications[-500:]


_SESSIONS: dict[str, LSPSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _session(session_id: str) -> LSPSession:
    key = session_id.strip()
    if not key:
        raise ValueError("session_id cannot be empty")
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            raise ValueError(f"LSP session not found: {key}")
        return _SESSIONS[key]


@tool
def lsp_start_session(
    session_id: str,
    command: str,
    workspace_path: str,
    initialization_options_json: str = "{}",
    trace: str = "off",
) -> str:
    """
    启动并初始化一个 LSP 会话，供后续工具复用。
    """
    try:
        key = session_id.strip()
        if not key:
            return "LSP error: session_id cannot be empty"
        if not command.strip():
            return "LSP error: command cannot be empty"
        if not os.path.isdir(workspace_path):
            return f"LSP error: invalid workspace path '{workspace_path}'"
        with _SESSIONS_LOCK:
            if key in _SESSIONS and _SESSIONS[key].is_alive():
                return _json_dump({"session_id": key, "status": "already_running"})
            if key in _SESSIONS:
                try:
                    _SESSIONS[key].stop()
                except Exception:
                    pass
            sess = LSPSession(key, command, workspace_path)
            _SESSIONS[key] = sess
        sess.start()
        init_options = _parse_json(initialization_options_json, {})
        result = sess.initialize(initialization_options=init_options, trace=trace)
        return _json_dump(
            {
                "session_id": key,
                "status": "started",
                "workspace_path": os.path.abspath(workspace_path),
                "initialize_result": result,
            }
        )
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_stop_session(session_id: str) -> str:
    """
    关闭一个 LSP 会话并释放资源。
    """
    try:
        key = session_id.strip()
        with _SESSIONS_LOCK:
            sess = _SESSIONS.pop(key, None)
        if not sess:
            return _json_dump({"session_id": key, "status": "not_found"})
        sess.stop()
        return _json_dump({"session_id": key, "status": "stopped"})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_list_sessions() -> str:
    """
    列出当前可用的 LSP 会话状态。
    """
    try:
        with _SESSIONS_LOCK:
            items = [
                {
                    "session_id": sid,
                    "alive": sess.is_alive(),
                    "workspace_path": sess.workspace_path,
                    "command": sess.command,
                }
                for sid, sess in _SESSIONS.items()
            ]
        return _json_dump({"count": len(items), "sessions": items})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_open_document(session_id: str, file_path: str, language_id: str = "") -> str:
    """
    将文件同步给 LSP，开始跟踪该文档版本与诊断。
    """
    try:
        sess = _session(session_id)
        if not os.path.isfile(file_path):
            return f"LSP error: file not found '{file_path}'"
        result = sess.did_open(file_path=file_path, language_id=language_id)
        return _json_dump({"session_id": session_id, "status": "opened", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_change_document(session_id: str, file_path: str, new_text: str) -> str:
    """
    发送全文变更到 LSP，触发增量分析与诊断更新。
    """
    try:
        sess = _session(session_id)
        result = sess.did_change(file_path=file_path, text=new_text)
        return _json_dump({"session_id": session_id, "status": "changed", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_close_document(session_id: str, file_path: str) -> str:
    """
    通知 LSP 文档关闭。
    """
    try:
        sess = _session(session_id)
        result = sess.did_close(file_path=file_path)
        return _json_dump({"session_id": session_id, "status": "closed", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_hover(session_id: str, file_path: str, line: int, character: int) -> str:
    """
    查询指定位置的悬浮信息。
    """
    try:
        sess = _session(session_id)
        result = sess.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": _to_uri(file_path)},
                "position": {"line": int(line), "character": int(character)},
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_definition(session_id: str, file_path: str, line: int, character: int) -> str:
    """
    查询定义跳转位置。
    """
    try:
        sess = _session(session_id)
        result = sess.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": _to_uri(file_path)},
                "position": {"line": int(line), "character": int(character)},
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_references(
    session_id: str,
    file_path: str,
    line: int,
    character: int,
    include_declaration: bool = True,
) -> str:
    """
    查询引用位置列表。
    """
    try:
        sess = _session(session_id)
        result = sess.request(
            "textDocument/references",
            {
                "textDocument": {"uri": _to_uri(file_path)},
                "position": {"line": int(line), "character": int(character)},
                "context": {"includeDeclaration": include_declaration},
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_document_symbols(session_id: str, file_path: str) -> str:
    """
    获取文档符号树。
    """
    try:
        sess = _session(session_id)
        result = sess.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": _to_uri(file_path)}},
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_workspace_symbols(session_id: str, query: str) -> str:
    """
    按关键字搜索工作区符号。
    """
    try:
        sess = _session(session_id)
        result = sess.request("workspace/symbol", {"query": query})
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_rename(
    session_id: str,
    file_path: str,
    line: int,
    character: int,
    new_name: str,
) -> str:
    """
    触发符号重命名并返回 WorkspaceEdit。
    """
    try:
        sess = _session(session_id)
        result = sess.request(
            "textDocument/rename",
            {
                "textDocument": {"uri": _to_uri(file_path)},
                "position": {"line": int(line), "character": int(character)},
                "newName": new_name,
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_code_actions(
    session_id: str,
    file_path: str,
    start_line: int,
    start_character: int,
    end_line: int,
    end_character: int,
    diagnostics_json: str = "[]",
) -> str:
    """
    获取某个范围可执行的 Code Action。
    """
    try:
        sess = _session(session_id)
        diagnostics = _parse_json(diagnostics_json, [])
        result = sess.request(
            "textDocument/codeAction",
            {
                "textDocument": {"uri": _to_uri(file_path)},
                "range": {
                    "start": {"line": int(start_line), "character": int(start_character)},
                    "end": {"line": int(end_line), "character": int(end_character)},
                },
                "context": {"diagnostics": diagnostics},
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_format_document(
    session_id: str,
    file_path: str,
    tab_size: int = 4,
    insert_spaces: bool = True,
) -> str:
    """
    请求文档格式化并返回 TextEdit 列表。
    """
    try:
        sess = _session(session_id)
        result = sess.request(
            "textDocument/formatting",
            {
                "textDocument": {"uri": _to_uri(file_path)},
                "options": {"tabSize": int(tab_size), "insertSpaces": bool(insert_spaces)},
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_get_diagnostics(session_id: str, file_path: str = "") -> str:
    """
    获取缓存的诊断信息，可按文件过滤。
    """
    try:
        sess = _session(session_id)
        result = sess.diagnostics(file_path=file_path)
        return _json_dump({"session_id": session_id, "diagnostics": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_get_notifications(session_id: str, limit: int = 20) -> str:
    """
    获取最近通知，便于 LangGraph 在步骤间追踪上下文。
    """
    try:
        sess = _session(session_id)
        result = sess.notifications(limit=max(1, min(int(limit), 100)))
        return _json_dump({"session_id": session_id, "notifications": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_get_server_logs(session_id: str, limit: int = 50) -> str:
    """
    获取最近的 LSP stderr 输出，便于故障排查。
    """
    try:
        sess = _session(session_id)
        logs = sess.stderr_output(limit=max(1, min(int(limit), 200)))
        return _json_dump({"session_id": session_id, "stderr": logs})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_wait(seconds: float = 0.5) -> str:
    """
    在图执行步骤间短暂等待，便于异步诊断到达。
    """
    try:
        delay = max(0.0, min(float(seconds), 5.0))
        time.sleep(delay)
        return _json_dump({"slept_seconds": delay})
    except Exception as e:
        return f"LSP error: {str(e)}"
