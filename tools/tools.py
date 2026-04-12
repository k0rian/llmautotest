from typing import Any

from langchain_community.tools.file_management import (
    FileSearchTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)

from tools.description.description import TOOL_DESCRIPTIONS
from tools.grep import grep_search
from tools.gui_ipc import gui_agent_run
from tools.shell import run_shell_command
from tools.semantic_diff_ts import (
    semantic_diff_with_description,
    semantic_index_functions,
    semantic_search_functions,
)
from tools.lsp import (
    lsp_change_document,
    lsp_close_document,
    lsp_code_actions,
    lsp_completion,
    lsp_definition,
    lsp_document_symbols,
    lsp_format_document,
    lsp_get_diagnostics,
    lsp_get_notifications,
    lsp_get_server_logs,
    lsp_get_session_info,
    lsp_hover,
    lsp_list_sessions,
    lsp_open_document,
    lsp_raw_request,
    lsp_references,
    lsp_rename,
    lsp_save_document,
    lsp_start_session,
    lsp_stop_session,
    lsp_wait,
    lsp_wait_for_diagnostics,
    lsp_workspace_symbols,
)
from tools.static_analysis import audit_codebase, list_audit_rules, semgrep_scan

def _tool_registry() -> dict[str, Any]:
    return {
        "read_file": ReadFileTool(),
        "write_file": WriteFileTool(),
        "list_directory": ListDirectoryTool(),
        "file_search": FileSearchTool(),
        "grep_search": grep_search,
        "semgrep_scan": semgrep_scan,
        "list_audit_rules": list_audit_rules,
        "audit_codebase": audit_codebase,
        "gui_agent_run": gui_agent_run,
        "run_shell_command": run_shell_command,
        "semantic_index_functions": semantic_index_functions,
        "semantic_search_functions": semantic_search_functions,
        "semantic_diff_with_description": semantic_diff_with_description,
        "lsp_start_session": lsp_start_session,
        "lsp_stop_session": lsp_stop_session,
        "lsp_list_sessions": lsp_list_sessions,
        "lsp_get_session_info": lsp_get_session_info,
        "lsp_open_document": lsp_open_document,
        "lsp_change_document": lsp_change_document,
        "lsp_save_document": lsp_save_document,
        "lsp_close_document": lsp_close_document,
        "lsp_hover": lsp_hover,
        "lsp_definition": lsp_definition,
        "lsp_references": lsp_references,
        "lsp_document_symbols": lsp_document_symbols,
        "lsp_workspace_symbols": lsp_workspace_symbols,
        "lsp_rename": lsp_rename,
        "lsp_completion": lsp_completion,
        "lsp_code_actions": lsp_code_actions,
        "lsp_raw_request": lsp_raw_request,
        "lsp_format_document": lsp_format_document,
        "lsp_get_diagnostics": lsp_get_diagnostics,
        "lsp_wait_for_diagnostics": lsp_wait_for_diagnostics,
        "lsp_get_notifications": lsp_get_notifications,
        "lsp_get_server_logs": lsp_get_server_logs,
        "lsp_wait": lsp_wait,
    }


def build_tools() -> list[Any]:
    registry = _tool_registry()
    return [registry[name] for name in TOOL_DESCRIPTIONS if name in registry]


def build_tool_description_map() -> dict[str, dict[str, str]]:
    return TOOL_DESCRIPTIONS.copy()
