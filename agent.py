from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from llm_utils.config_loader import get_api_key
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    FileSearchTool,
)
from tools.lsp import (
    lsp_start_session,
    lsp_stop_session,
    lsp_list_sessions,
    lsp_get_session_info,
    lsp_open_document,
    lsp_change_document,
    lsp_save_document,
    lsp_close_document,
    lsp_hover,
    lsp_definition,
    lsp_references,
    lsp_document_symbols,
    lsp_workspace_symbols,
    lsp_rename,
    lsp_completion,
    lsp_code_actions,
    lsp_raw_request,
    lsp_format_document,
    lsp_get_diagnostics,
    lsp_wait_for_diagnostics,
    lsp_get_notifications,
    lsp_get_server_logs,
    lsp_wait,
)
from tools.grep import grep_search
from tools.static_analysis import semgrep_scan, list_audit_rules, audit_codebase


tools =[
    ReadFileTool(),
    WriteFileTool(),
    ListDirectoryTool(),
    FileSearchTool(),
    grep_search,
    semgrep_scan,
    list_audit_rules,
    audit_codebase,
    lsp_start_session,
    lsp_stop_session,
    lsp_list_sessions,
    lsp_get_session_info,
    lsp_open_document,
    lsp_change_document,
    lsp_save_document,
    lsp_close_document,
    lsp_hover,
    lsp_definition,
    lsp_references,
    lsp_document_symbols,
    lsp_workspace_symbols,
    lsp_rename,
    lsp_completion,
    lsp_code_actions,
    lsp_raw_request,
    lsp_format_document,
    lsp_get_diagnostics,
    lsp_wait_for_diagnostics,
    lsp_get_notifications,
    lsp_get_server_logs,
    lsp_wait,
]

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=get_api_key(),
)

agent = create_agent(
    model,
    tools=tools,
    verbose=True,
)





