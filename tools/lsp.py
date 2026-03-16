import json
import time
from typing import Any
from langchain.tools import tool
from services.lsp.manager import (
    ensure_workspace_file,
    get_session,
    list_sessions,
    start_session,
    stop_session,
    to_uri,
)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_json(value: str, fallback: Any) -> Any:
    if not value or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


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
        init_options = _parse_json(initialization_options_json, {})
        result = start_session(
            session_id=session_id,
            command=command,
            workspace_path=workspace_path,
            initialization_options=init_options,
            trace=trace,
        )
        return _json_dump(result)
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_stop_session(session_id: str) -> str:
    """
    关闭一个 LSP 会话并释放资源。
    """
    try:
        return _json_dump(stop_session(session_id))
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_list_sessions() -> str:
    """
    列出当前可用的 LSP 会话状态。
    """
    try:
        items = list_sessions()
        return _json_dump({"count": len(items), "sessions": items})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_open_document(session_id: str, file_path: str, language_id: str = "") -> str:
    """
    将文件同步给 LSP，开始跟踪该文档版本与诊断。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.did_open(file_path=absolute, language_id=language_id)
        return _json_dump({"session_id": session_id, "status": "opened", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_change_document(session_id: str, file_path: str, new_text: str) -> str:
    """
    发送全文变更到 LSP，触发增量分析与诊断更新。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.did_change(file_path=absolute, text=new_text)
        return _json_dump({"session_id": session_id, "status": "changed", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_close_document(session_id: str, file_path: str) -> str:
    """
    通知 LSP 文档关闭。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.did_close(file_path=absolute)
        return _json_dump({"session_id": session_id, "status": "closed", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_save_document(session_id: str, file_path: str, include_text: bool = False) -> str:
    """
    通知 LSP 文档已保存，触发保存后诊断。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        text = ""
        if include_text:
            with open(absolute, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        result = sess.did_save(file_path=absolute, text=text)
        return _json_dump({"session_id": session_id, "status": "saved", **result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_hover(session_id: str, file_path: str, line: int, character: int) -> str:
    """
    查询指定位置的悬浮信息。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": to_uri(absolute)},
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
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": to_uri(absolute)},
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
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.request(
            "textDocument/references",
            {
                "textDocument": {"uri": to_uri(absolute)},
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
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": to_uri(absolute)}},
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
        sess = get_session(session_id)
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
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.request(
            "textDocument/rename",
            {
                "textDocument": {"uri": to_uri(absolute)},
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
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        diagnostics = _parse_json(diagnostics_json, [])
        result = sess.request(
            "textDocument/codeAction",
            {
                "textDocument": {"uri": to_uri(absolute)},
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
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        result = sess.request(
            "textDocument/formatting",
            {
                "textDocument": {"uri": to_uri(absolute)},
                "options": {"tabSize": int(tab_size), "insertSpaces": bool(insert_spaces)},
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_completion(
    session_id: str,
    file_path: str,
    line: int,
    character: int,
    trigger_character: str = "",
) -> str:
    """
    获取指定位置的自动补全候选。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        context = {"triggerKind": 1}
        if trigger_character:
            context = {"triggerKind": 2, "triggerCharacter": trigger_character}
        result = sess.request(
            "textDocument/completion",
            {
                "textDocument": {"uri": to_uri(absolute)},
                "position": {"line": int(line), "character": int(character)},
                "context": context,
            },
        )
        return _json_dump({"session_id": session_id, "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_raw_request(session_id: str, method: str, params_json: str = "{}", timeout_seconds: int = 20) -> str:
    """
    发送任意 LSP request，便于扩展未内置的方法。
    """
    try:
        if not method.strip():
            return "LSP error: method cannot be empty"
        sess = get_session(session_id)
        params = _parse_json(params_json, {})
        if not isinstance(params, dict):
            return "LSP error: params_json must be a JSON object"
        result = sess.request(method.strip(), params, timeout_seconds=max(1, min(int(timeout_seconds), 120)))
        return _json_dump({"session_id": session_id, "method": method.strip(), "result": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_get_diagnostics(session_id: str, file_path: str = "") -> str:
    """
    获取缓存的诊断信息，可按文件过滤。
    """
    try:
        sess = get_session(session_id)
        absolute = ""
        if file_path and file_path.strip():
            absolute = ensure_workspace_file(sess, file_path)
        result = sess.diagnostics(file_path=absolute)
        return _json_dump({"session_id": session_id, "diagnostics": result})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_wait_for_diagnostics(session_id: str, file_path: str, timeout_seconds: float = 5.0) -> str:
    """
    等待某文件诊断更新或超时返回。
    """
    try:
        sess = get_session(session_id)
        absolute = ensure_workspace_file(sess, file_path)
        baseline = sess.diagnostics_version()
        timeout = max(0.1, min(float(timeout_seconds), 30.0))
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = sess.diagnostics_version()
            if current > baseline:
                break
            time.sleep(0.1)
        diagnostics = sess.diagnostics(file_path=absolute)
        return _json_dump(
            {
                "session_id": session_id,
                "file": absolute,
                "diagnostics_version": sess.diagnostics_version(),
                "diagnostics_count": len(diagnostics),
                "diagnostics": diagnostics,
            }
        )
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_get_notifications(session_id: str, limit: int = 20) -> str:
    """
    获取最近通知，便于 LangGraph 在步骤间追踪上下文。
    """
    try:
        sess = get_session(session_id)
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
        sess = get_session(session_id)
        logs = sess.stderr_output(limit=max(1, min(int(limit), 200)))
        return _json_dump({"session_id": session_id, "stderr": logs})
    except Exception as e:
        return f"LSP error: {str(e)}"


@tool
def lsp_get_session_info(session_id: str) -> str:
    """
    查看会话状态、初始化结果与诊断版本。
    """
    try:
        sess = get_session(session_id)
        return _json_dump(
            {
                "session_id": session_id,
                "alive": sess.is_alive(),
                "workspace_path": sess.workspace_path,
                "command": sess.command,
                "diagnostics_version": sess.diagnostics_version(),
                "initialize_result": sess.initialize_result(),
            }
        )
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
