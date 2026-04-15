TOOL_DESCRIPTIONS = {
    "read_file": {
        "category": "file",
        "description": "- Read file contents from local workspace\n- Returns plain text output for downstream analysis",
    },
    "write_file": {
        "category": "file",
        "description": "- Write or overwrite file contents in local workspace\n- Use this tool for direct file content updates",
    },
    "list_directory": {
        "category": "file",
        "description": "- List files and folders under a target directory\n- Use this tool to quickly inspect workspace structure",
    },
    "file_search": {
        "category": "file",
        "description": "- Search files by filename patterns in workspace\n- Use this tool when you know naming clues but not full path",
    },
    "grep_search": {
        "category": "search",
        "description": "- Fast content search tool that works with any codebase size\n- Searches file contents using regular expressions\n- Supports full regex syntax (eg. \"log.*Error\", \"function\\s+\\w+\", etc.)\n- Filter files by pattern with the include parameter (eg. \"*.js\", \"*.{ts,tsx}\")\n- Returns file paths and line numbers with at least one match sorted by modification time\n- Use this tool when you need to find files containing specific patterns\n- If you need to identify/count the number of matches within files, use ripgrep directly in command execution, not grep\n- When doing open-ended multi-round exploration, combine with task planning for iterative narrowing",
    },
    "semgrep_scan": {
        "category": "analysis",
        "description": "- Run semgrep-based static analysis on a directory\n- Returns findings in structured JSON text",
    },
    "list_audit_rules": {
        "category": "analysis",
        "description": "- List built-in security audit regex rules\n- Use this tool to understand current local audit coverage",
    },
    "audit_codebase": {
        "category": "analysis",
        "description": "- Run combined audit using semgrep and built-in pattern rules\n- Returns merged findings with severity summary",
    },
    "gui_agent_run": {
        "category": "gui",
        "description": "- Execute GUI interaction task through gui_agent\n- Supports step limit and history reset\n- Returns structured execution status/details JSON text",
    },
    "run_shell_command": {
        "category": "shell",
        "description": "- Execute CLI commands in current environment\n- Commands may require explicit user confirmation before running\n- Returns JSON payload with return code, stdout, stderr and cwd",
    },
    "semantic_index_functions": {
        "category": "semantic",
        "description": "- Build function-level semantic index for codebase\n- Extracts Python/JS/TS/C/C++ functions and vectorizes signatures/docs/source\n- Supports .py/.js/.jsx/.ts/.tsx/.c/.h/.cc/.cpp/.hpp\n- Use this before large-scale semantic retrieval tasks",
    },
    "semantic_search_functions": {
        "category": "semantic",
        "description": "- Semantic retrieve functions by natural language query\n- Returns most relevant functions with score and location\n- Suitable for mapping requirement text to implementation points",
    },
    "semantic_lookup_function_name": {
        "category": "semantic",
        "description": "- Lookup indexed functions by exact or fuzzy function name\n- Uses the function-name index built during semantic indexing\n- Returns file, line range, signature, and language for matched names",
    },
    "semantic_diff_with_description": {
        "category": "semantic",
        "description": "- Compare implementation with user description (PRD/API docs)\n- Generates coverage summary: covered/partial/missing requirements\n- Also lists potentially undocumented functions",
    },
    "build_hierarchical_code_index": {
        "category": "semantic",
        "description": "- Build hierarchical semantic index from function/file/directory/repository levels\n- Supports cache reuse and incremental update by file content hash\n- Returns index stats and cache path for downstream localization/retrieval",
    },
    "semantic_localize_requirement": {
        "category": "semantic",
        "description": "- Localize requirement in top-down order: repository -> directory -> file -> function\n- Returns scored candidates with reasons for each hit",
    },
    "query_symbol_definition": {
        "category": "semantic",
        "description": "- Query symbol definition candidates from semantic index\n- Returns file path, line range, signature, and source summary evidence",
    },
    "query_callee_functions": {
        "category": "semantic",
        "description": "- Query direct callee functions of a target function\n- Returns caller-callee evidence with file paths and line ranges",
    },
    "query_caller_functions": {
        "category": "semantic",
        "description": "- Query direct caller functions of a target function\n- Returns caller evidence with file paths and line ranges",
    },
    "validate_semantic_finding": {
        "category": "semantic",
        "description": "- Validate preliminary semantic finding using evidence bundle\n- Distinguishes true missing/partial coverage/naming misleading/insufficient evidence",
    },
    "lsp_start_session": {"category": "lsp", "description": "- Start and initialize an LSP session for a workspace"},
    "lsp_stop_session": {"category": "lsp", "description": "- Stop an LSP session and release resources"},
    "lsp_list_sessions": {"category": "lsp", "description": "- List current LSP sessions and status"},
    "lsp_get_session_info": {"category": "lsp", "description": "- Get runtime information for an LSP session"},
    "lsp_open_document": {"category": "lsp", "description": "- Open and sync a document into LSP state"},
    "lsp_change_document": {"category": "lsp", "description": "- Send full-text document change to LSP"},
    "lsp_save_document": {"category": "lsp", "description": "- Notify LSP that a document has been saved"},
    "lsp_close_document": {"category": "lsp", "description": "- Notify LSP that a document has been closed"},
    "lsp_hover": {"category": "lsp", "description": "- Query hover information at a given position"},
    "lsp_definition": {"category": "lsp", "description": "- Query definition locations for a symbol"},
    "lsp_references": {"category": "lsp", "description": "- Query reference locations for a symbol"},
    "lsp_document_symbols": {"category": "lsp", "description": "- Retrieve document symbol tree"},
    "lsp_workspace_symbols": {"category": "lsp", "description": "- Search symbols across workspace"},
    "lsp_rename": {"category": "lsp", "description": "- Request rename and return workspace edits"},
    "lsp_completion": {"category": "lsp", "description": "- Request code completion candidates"},
    "lsp_code_actions": {"category": "lsp", "description": "- Request code actions in a source range"},
    "lsp_raw_request": {"category": "lsp", "description": "- Send raw LSP request for non-wrapped methods"},
    "lsp_format_document": {"category": "lsp", "description": "- Request formatting edits for a document"},
    "lsp_get_diagnostics": {"category": "lsp", "description": "- Retrieve cached diagnostics from LSP"},
    "lsp_wait_for_diagnostics": {"category": "lsp", "description": "- Wait for diagnostics update with timeout"},
    "lsp_get_notifications": {"category": "lsp", "description": "- Retrieve recent LSP notifications"},
    "lsp_get_server_logs": {"category": "lsp", "description": "- Retrieve recent LSP server stderr logs"},
    "lsp_wait": {"category": "lsp", "description": "- Sleep briefly between graph steps for async events"},
}
