import json
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any
from .language import get_language_id


def _to_uri(file_path: str) -> str:
    return Path(os.path.abspath(file_path)).as_uri()


def _normalize_command(command: str | list[str]) -> tuple[str, list[str]]:
    if isinstance(command, list):
        parts = [str(item).strip() for item in command if str(item).strip()]
    else:
        parts = shlex.split(str(command), posix=False)
    if not parts:
        raise ValueError("command cannot be empty")
    executable = parts[0]
    resolved = shutil.which(executable)
    if resolved:
        parts[0] = resolved
    display = " ".join(parts)
    return display, parts


class LSPSession:
    def __init__(
        self,
        session_id: str,
        command: str | list[str],
        workspace_path: str,
        env: dict[str, str] | None = None,
    ):
        self.session_id = session_id
        command_text, command_parts = _normalize_command(command)
        self.command = command_text
        self.command_parts = command_parts
        self.workspace_path = os.path.abspath(workspace_path)
        self.env = dict(env) if isinstance(env, dict) else {}
        self.process: subprocess.Popen | None = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_request_id = 1
        self._pending: dict[int, dict[str, Any]] = {}
        self._notifications: list[dict[str, Any]] = []
        self._diagnostics: dict[str, Any] = {}
        self._stderr_lines: list[str] = []
        self._doc_versions: dict[str, int] = {}
        self._diagnostics_version = 0
        self._initialize_result: dict[str, Any] = {}
        self._running = False
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        process_env = os.environ.copy()
        process_env.update(self.env)
        try:
            self.process = subprocess.Popen(
                self.command_parts,
                cwd=self.workspace_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=process_env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"LSP command not found: {self.command}") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to start LSP process: {self.command}") from exc
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
        alive = bool(self.process and self.process.poll() is None)
        if not alive:
            self._running = False
        return alive

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
        self._initialize_result = result if isinstance(result, dict) else {}
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
                    "languageId": get_language_id(absolute, language_id),
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

    def did_save(self, file_path: str, text: str = "") -> dict[str, Any]:
        absolute = os.path.abspath(file_path)
        uri = _to_uri(absolute)
        payload = {"textDocument": {"uri": uri}}
        if text:
            payload["text"] = text
        self.notify("textDocument/didSave", payload)
        return {"uri": uri}

    def diagnostics(self, file_path: str = "") -> Any:
        with self._state_lock:
            if file_path and file_path.strip():
                return list(self._diagnostics.get(_to_uri(file_path), []))
            return dict(self._diagnostics)

    def diagnostics_version(self) -> int:
        with self._state_lock:
            return self._diagnostics_version

    def notifications(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._state_lock:
            return list(self._notifications[-limit:])

    def stderr_output(self, limit: int = 50) -> list[str]:
        if limit <= 0:
            return []
        with self._state_lock:
            return list(self._stderr_lines[-limit:])

    def initialize_result(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._initialize_result)

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
                with self._state_lock:
                    self._stderr_lines.append("LSP stdout loop terminated due to parser/read error")
                    if len(self._stderr_lines) > 200:
                        self._stderr_lines = self._stderr_lines[-200:]
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
        if "id" in message and "method" in message:
            request_id = message.get("id")
            method = message.get("method")
            if method == "workspace/configuration":
                self._send_response(request_id, [])
            elif method in {
                "window/workDoneProgress/create",
                "client/registerCapability",
                "client/unregisterCapability",
                "workspace/didChangeWorkspaceFolders",
            }:
                self._send_response(request_id, None)
            elif method == "workspace/applyEdit":
                self._send_response(request_id, {"applied": False})
            else:
                self._send_error(request_id, -32601, f"Method not implemented: {method}")
            with self._state_lock:
                self._notifications.append(message)
                if len(self._notifications) > 500:
                    self._notifications = self._notifications[-500:]
            return
        method = message.get("method")
        if method == "textDocument/publishDiagnostics":
            params = message.get("params", {})
            uri = params.get("uri")
            diagnostics = params.get("diagnostics", [])
            if uri:
                with self._state_lock:
                    self._diagnostics[uri] = diagnostics
                    self._diagnostics_version += 1
        with self._state_lock:
            self._notifications.append(message)
            if len(self._notifications) > 500:
                self._notifications = self._notifications[-500:]

    def _send_response(self, request_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})
