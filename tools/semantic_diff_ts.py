
import json
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from langchain.tools import tool
from tree_sitter import Node, Parser, Query, QueryCursor
import tree_sitter_language_pack as ts_pack

from llm_utils import iter_code_files, normalize_include_glob as normalize_scope_glob, resolve_code_scope
from llm_utils.config_loader import DEFAULT_MODEL_NAME, get_client, load_model_name


SKIP_DIRS = {
    ".git",
    ".venv",
    ".uvcache",
    ".vs",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "temp_cache",
}
DEFAULT_INCLUDE_GLOB = "*.py,*.js,*.jsx,*.ts,*.tsx,*.go,*.c,*.h,*.cc,*.cpp,*.hpp"
MAX_FILE_BYTES = 1024 * 1024
MAX_SOURCE_CHARS = 2400
INDEX_CACHE_DIRNAME = ".llmautotest"
INDEX_CACHE_SUBDIR = "semantic_index"
SUPPORTED_LANGUAGES = ("python", "javascript", "typescript", "tsx", "go", "c", "cpp")
STRICT_NATIVE_LANGUAGES = {"c", "cpp"}

CHINESE_KEYWORD_ALIASES = {
    "静态": ["static"],
    "扫描": ["scan"],
    "安全": ["security", "secure"],
    "审计": ["audit"],
    "诊断": ["diagnostic", "diagnostics"],
    "会话": ["session"],
    "接口": ["api", "endpoint"],
    "文档": ["doc", "spec"],
    "函数": ["function", "method"],
    "索引": ["index"],
    "检索": ["search", "retrieve"],
    "对比": ["diff", "compare"],
    "需求": ["requirement"],
    "实现": ["implementation"],
    "缓存": ["cache"],
    "配置": ["config"],
}

_C_QUERY_FUNCTION_DEFINITION = r"""
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @func.name
    parameters: (parameter_list) @func.params)
  body: (compound_statement) @func.body) @func.def

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @func.name
      parameters: (parameter_list) @func.params))
  body: (compound_statement) @func.body) @func.def
"""

_C_QUERY_FUNCTION_DECLARATION = r"""
(declaration
  declarator: (function_declarator
    declarator: (identifier) @func.name
    parameters: (parameter_list) @func.params)) @func.decl

(declaration
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @func.name
      parameters: (parameter_list) @func.params))) @func.decl
"""

_CPP_QUERY_FUNCTION_DEFINITION = r"""
(function_definition
  declarator: [
    (function_declarator
      declarator: (identifier) @func.name
      parameters: (parameter_list) @func.params)

    (function_declarator
      declarator: (field_identifier) @func.name
      parameters: (parameter_list) @func.params)

    (function_declarator
      declarator: (qualified_identifier
        name: (identifier) @func.name)
      parameters: (parameter_list) @func.params)

    (function_declarator
      declarator: (qualified_identifier
        name: (destructor_name) @func.name)
      parameters: (parameter_list) @func.params)
  ]
  body: (compound_statement) @func.body) @func.def
"""

_CPP_QUERY_FUNCTION_DECLARATION = r"""
(declaration
  declarator: [
    (function_declarator
      declarator: (identifier) @func.name
      parameters: (parameter_list) @func.params)

    (function_declarator
      declarator: (field_identifier) @func.name
      parameters: (parameter_list) @func.params)

    (function_declarator
      declarator: (qualified_identifier
        name: (identifier) @func.name)
      parameters: (parameter_list) @func.params)

    (function_declarator
      declarator: (qualified_identifier
        name: (destructor_name) @func.name)
      parameters: (parameter_list) @func.params)
  ]) @func.decl
"""


@dataclass
class FunctionRecord:
    file: str
    language: str
    kind: str
    name: str
    signature: str
    start_line: int
    end_line: int
    doc: str
    source: str
    token_tf: dict[str, float]
    vector: dict[str, float]
    norm: float
    summary: str = ""


@dataclass
class SemanticIndex:
    root: str
    resolved_path: str
    target_path: str
    scope_type: str
    include_glob: str
    created_at: str
    file_count: int
    function_count: int
    target_files: list[str]
    functions: list[FunctionRecord]
    function_name_index: dict[str, list[dict[str, Any]]]
    idf: dict[str, float]
    parser_status: dict[str, str]
    parser_errors: dict[str, str]
    language_symbol_counts: dict[str, dict[str, int]]
    indexed_files_by_language: dict[str, int]
    summary_mode: str = "deterministic"
    summary_model: str = ""
    summary_errors: list[str] | None = None


_INDEX_CACHE: dict[str, SemanticIndex] = {}
_PARSER_FACTORIES: dict[str, Callable[[], Parser]] = {}
_PARSER_STATUS: dict[str, str] = {}
_PARSER_ERRORS: dict[str, str] = {}
_QUERY_CACHE: dict[str, Query] = {}


def _summary_cache_tag(use_llm_summary: bool, model_name: str = "") -> str:
    if not use_llm_summary:
        return "summary:deterministic"
    model = (model_name or load_model_name(DEFAULT_MODEL_NAME)).strip()
    return f"summary:llm:{model}"


def _cache_key(scope_type: str, resolved_path: str, include_glob: str, use_llm_summary: bool = False, model_name: str = "") -> str:
    return (
        f"{scope_type}::{Path(resolved_path).resolve()}::{include_glob.strip().lower()}"
        f"::{_summary_cache_tag(use_llm_summary, model_name)}"
    )


def _index_cache_dir(index_root_path: str) -> Path:
    return Path(index_root_path).resolve() / INDEX_CACHE_DIRNAME / INDEX_CACHE_SUBDIR


def _index_cache_file(
    index_root_path: str,
    scope_type: str,
    resolved_path: str,
    include_glob: str,
    use_llm_summary: bool = False,
    model_name: str = "",
) -> Path:
    digest = hashlib.sha1(
        _cache_key(scope_type, resolved_path, include_glob, use_llm_summary, model_name).encode("utf-8")
    ).hexdigest()
    return _index_cache_dir(index_root_path) / f"{digest}.json"


def _index_to_payload(index: SemanticIndex) -> dict[str, Any]:
    return {
        "version": 1,
        "root": index.root,
        "resolved_path": index.resolved_path,
        "target_path": index.target_path,
        "scope_type": index.scope_type,
        "include_glob": index.include_glob,
        "created_at": index.created_at,
        "file_count": index.file_count,
        "function_count": index.function_count,
        "target_files": index.target_files,
        "function_name_index": index.function_name_index,
        "idf": index.idf,
        "parser_status": index.parser_status,
        "parser_errors": index.parser_errors,
        "language_symbol_counts": index.language_symbol_counts,
        "indexed_files_by_language": index.indexed_files_by_language,
        "summary_mode": index.summary_mode,
        "summary_model": index.summary_model,
        "summary_errors": index.summary_errors or [],
        "functions": [
            {
                "file": item.file,
                "language": item.language,
                "kind": item.kind,
                "name": item.name,
                "signature": item.signature,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "doc": item.doc,
                "source": item.source,
                "token_tf": item.token_tf,
                "vector": item.vector,
                "norm": item.norm,
                "summary": item.summary,
            }
            for item in index.functions
        ],
    }


def _index_from_payload(payload: dict[str, Any]) -> SemanticIndex:
    functions: list[FunctionRecord] = []
    for item in payload.get("functions", []):
        if not isinstance(item, dict):
            continue
        functions.append(
            FunctionRecord(
                file=str(item.get("file", "")),
                language=str(item.get("language", "unknown")),
                kind=str(item.get("kind", "function_definition")),
                name=str(item.get("name", "")),
                signature=str(item.get("signature", "")),
                start_line=int(item.get("start_line", 0) or 0),
                end_line=int(item.get("end_line", 0) or 0),
                doc=str(item.get("doc", "")),
                source=str(item.get("source", "")),
                token_tf={str(k): float(v) for k, v in dict(item.get("token_tf", {})).items()},
                vector={str(k): float(v) for k, v in dict(item.get("vector", {})).items()},
                norm=float(item.get("norm", 0.0) or 0.0),
                summary=str(item.get("summary", "")),
            )
        )

    raw_lsc = payload.get("language_symbol_counts", {})
    language_symbol_counts: dict[str, dict[str, int]] = {}
    if isinstance(raw_lsc, dict):
        for language, kinds in raw_lsc.items():
            if not isinstance(kinds, dict):
                continue
            language_symbol_counts[str(language)] = {str(k): int(v) for k, v in kinds.items()}

    raw_ifl = payload.get("indexed_files_by_language", {})
    indexed_files_by_language = (
        {str(k): int(v) for k, v in raw_ifl.items()} if isinstance(raw_ifl, dict) else {}
    )
    raw_parser_status = payload.get("parser_status", {})
    parser_status = (
        {str(k): str(v) for k, v in raw_parser_status.items()} if isinstance(raw_parser_status, dict) else {}
    )
    raw_parser_errors = payload.get("parser_errors", {})
    parser_errors = (
        {str(k): str(v) for k, v in raw_parser_errors.items()} if isinstance(raw_parser_errors, dict) else {}
    )
    raw_idf = payload.get("idf", {})
    idf = {str(k): float(v) for k, v in raw_idf.items()} if isinstance(raw_idf, dict) else {}
    raw_name_index = payload.get("function_name_index", {})
    function_name_index: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw_name_index, dict):
        for key, items in raw_name_index.items():
            if not isinstance(items, list):
                continue
            function_name_index[str(key)] = [item for item in items if isinstance(item, dict)]

    return SemanticIndex(
        root=str(payload.get("root", "")),
        resolved_path=str(payload.get("resolved_path", payload.get("root", ""))),
        target_path=str(payload.get("target_path", payload.get("resolved_path", payload.get("root", "")))),
        scope_type=str(payload.get("scope_type", "directory")),
        include_glob=str(payload.get("include_glob", DEFAULT_INCLUDE_GLOB)),
        created_at=str(payload.get("created_at", "")),
        file_count=int(payload.get("file_count", 0) or 0),
        function_count=int(payload.get("function_count", len(functions)) or len(functions)),
        target_files=[str(item) for item in payload.get("target_files", []) if str(item).strip()],
        functions=functions,
        function_name_index=function_name_index,
        idf=idf,
        parser_status=parser_status,
        parser_errors=parser_errors,
        language_symbol_counts=language_symbol_counts,
        indexed_files_by_language=indexed_files_by_language,
        summary_mode=str(payload.get("summary_mode", "deterministic")),
        summary_model=str(payload.get("summary_model", "")),
        summary_errors=[str(item) for item in payload.get("summary_errors", []) if str(item).strip()],
    )


def _save_index_to_disk(scope: Any, include_glob: str, index: SemanticIndex, use_llm_summary: bool = False, model_name: str = "") -> None:
    cache_file = _index_cache_file(
        index_root_path=scope.index_root_path,
        scope_type=scope.scope_type,
        resolved_path=scope.resolved_path,
        include_glob=include_glob,
        use_llm_summary=use_llm_summary,
        model_name=model_name,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = _index_to_payload(index)
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_index_from_disk(scope: Any, include_glob: str, use_llm_summary: bool = False, model_name: str = "") -> SemanticIndex | None:
    cache_file = _index_cache_file(
        index_root_path=scope.index_root_path,
        scope_type=scope.scope_type,
        resolved_path=scope.resolved_path,
        include_glob=include_glob,
        use_llm_summary=use_llm_summary,
        model_name=model_name,
    )
    if not cache_file.exists() or not cache_file.is_file():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if str(payload.get("resolved_path", payload.get("root", ""))).strip() != scope.resolved_path:
            return None
        if str(payload.get("scope_type", "")).strip() != scope.scope_type:
            return None
        if str(payload.get("include_glob", "")).strip().lower() != include_glob.strip().lower():
            return None
        return _index_from_payload(payload)
    except Exception:
        return None


def _normalize_include_glob(include_glob: str) -> list[str]:
    return normalize_scope_glob(include_glob, DEFAULT_INCLUDE_GLOB)


def _is_code_file(path: Path, patterns: list[str]) -> bool:
    return any(fnmatch(path.name, pattern) for pattern in patterns)


def _iter_code_files(root: Path, include_glob: str, max_files: int) -> list[Path]:
    patterns = _normalize_include_glob(include_glob)
    files = iter_code_files(root=root, patterns=patterns, max_files=max_files, skip_dirs=SKIP_DIRS)
    return [
        full
        for full in files
        if full.exists() and full.stat().st_size <= MAX_FILE_BYTES and _is_code_file(full, patterns)
    ]


def _split_identifier(text: str) -> list[str]:
    if not text:
        return []
    converted = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    converted = converted.replace("_", " ").replace("-", " ")
    return [part.lower() for part in converted.split() if part.strip()]


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    words: list[str] = []
    for chunk in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}", text):
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*", chunk):
            words.extend(_split_identifier(chunk))
        else:
            normalized = chunk.lower()
            words.append(normalized)
            if len(normalized) > 2:
                for index in range(0, len(normalized) - 1):
                    words.append(normalized[index : index + 2])
            for key, aliases in CHINESE_KEYWORD_ALIASES.items():
                if key in normalized:
                    words.extend(aliases)
    return [item for item in words if len(item) >= 2]

def _calc_tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counter = Counter(tokens)
    total = float(len(tokens))
    return {token: count / total for token, count in counter.items()}


def _calc_vector(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token, score in tf.items():
        idf_score = idf.get(token)
        if idf_score is None:
            continue
        vector[token] = score * idf_score
    return vector


def _calc_norm(vector: dict[str, float]) -> float:
    if not vector:
        return 0.0
    return math.sqrt(sum(value * value for value in vector.values()))


def _register_parser_factory(language: str, factory: Callable[[], Parser]) -> None:
    if language not in _PARSER_FACTORIES:
        _PARSER_FACTORIES[language] = factory


def _update_parser_error(language: str, message: str) -> None:
    current = _PARSER_ERRORS.get(language)
    if not current:
        _PARSER_ERRORS[language] = message
    elif message not in current:
        _PARSER_ERRORS[language] = f"{current}; {message}"


def _call_ts_pack_init(languages: list[str]) -> None:
    init_fn = getattr(ts_pack, "init", None)
    if init_fn is None:
        return
    attempts = (
        ({"languages": languages},),
        ({"parsers": languages},),
        (languages,),
        tuple(),
    )
    last_error: Exception | None = None
    for args in attempts:
        try:
            init_fn(*args)
            return
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"failed to call tree-sitter-language-pack init: {last_error}") from last_error


def _call_ts_pack_download(languages: list[str]) -> None:
    if not languages:
        return
    download_fn = getattr(ts_pack, "download", None)
    if download_fn is None:
        return
    try:
        download_fn(languages)
    except TypeError as exc:
        raise RuntimeError(f"failed to call tree-sitter-language-pack download: {exc}") from exc


def _init_parser_factories() -> None:
    if _PARSER_FACTORIES:
        return

    _PARSER_STATUS.clear()
    _PARSER_ERRORS.clear()
    for language in SUPPORTED_LANGUAGES:
        _PARSER_STATUS[language] = "pending"

    try:
        _call_ts_pack_init(list(SUPPORTED_LANGUAGES))
        available = set(ts_pack.available_languages())
        missing = [language for language in SUPPORTED_LANGUAGES if language not in available]
        if missing:
            _call_ts_pack_download(missing)

        for language in SUPPORTED_LANGUAGES:
            try:
                _ = ts_pack.get_language(language)
                parser = ts_pack.get_parser(language)
                _ = parser.parse(b"")
                _register_parser_factory(language, lambda lang=language: ts_pack.get_parser(lang))
                _PARSER_STATUS[language] = "ok"
            except Exception as exc:
                _PARSER_STATUS[language] = "failed"
                _update_parser_error(language, f"parser validation failed: {exc}")
    except Exception as exc:
        for language in SUPPORTED_LANGUAGES:
            if _PARSER_STATUS.get(language) == "pending":
                _PARSER_STATUS[language] = "failed"
                _update_parser_error(language, f"init stage failed: {exc}")
        raise RuntimeError(f"failed to initialize tree-sitter parsers: {exc}") from exc


def _parser_for_language(language: str) -> Parser | None:
    _init_parser_factories()
    factory = _PARSER_FACTORIES.get(language)
    if factory is None:
        return None
    return factory()


def _require_parsers(required_languages: set[str]) -> None:
    if not required_languages:
        return
    _init_parser_factories()
    missing = [lang for lang in sorted(required_languages) if _PARSER_STATUS.get(lang) != "ok"]
    if not missing:
        return
    detail = "; ".join(f"{lang}: {_PARSER_ERRORS.get(lang, 'unknown error')}" for lang in missing)
    raise RuntimeError(f"tree-sitter parser unavailable for {', '.join(missing)}; {detail}")


def _detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext == ".ts":
        return "typescript"
    if ext == ".tsx":
        return "tsx"
    if ext == ".go":
        return "go"
    if ext in {".c", ".h"}:
        return "c"
    if ext in {".cc", ".cpp", ".hpp"}:
        return "cpp"
    return "unknown"


def _iter_nodes(root: Node) -> list[Node]:
    stack = [root]
    output: list[Node] = []
    while stack:
        node = stack.pop()
        output.append(node)
        if node.children:
            stack.extend(reversed(node.children))
    return output


def _node_text(source: bytes, node: Node | None) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _signature_from_lines(lines: list[str], start_line: int) -> str:
    idx = max(0, start_line - 1)
    if idx >= len(lines):
        return ""
    return lines[idx].strip()[:240]


def _extract_python_doc_from_source(source: str) -> str:
    body = source.splitlines()[1:20]
    if not body:
        return ""
    text = "\n".join(body)
    match = re.search(r"^\s*(?:[rRuUbBfF]{0,2})('''|\"\"\")(.*?)(\1)", text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(2).strip()[:600]

def _extract_functions_python(path: Path, source: bytes, tree: Any) -> list[dict[str, Any]]:
    lines = source.decode("utf-8", errors="ignore").splitlines()
    functions: list[dict[str, Any]] = []
    for node in _iter_nodes(tree.root_node):
        if node.type != "function_definition":
            continue
        name = _node_text(source, node.child_by_field_name("name")).strip()
        if not name:
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        code = _node_text(source, node)[:MAX_SOURCE_CHARS]
        functions.append(
            {
                "file": str(path.resolve()),
                "language": "python",
                "kind": "function_definition",
                "name": name,
                "signature": _signature_from_lines(lines, start_line),
                "start_line": start_line,
                "end_line": end_line,
                "doc": _extract_python_doc_from_source(code),
                "source": code,
            }
        )
    return functions


def _extract_functions_js_ts(path: Path, source: bytes, tree: Any, language: str) -> list[dict[str, Any]]:
    lines = source.decode("utf-8", errors="ignore").splitlines()
    functions: list[dict[str, Any]] = []
    for node in _iter_nodes(tree.root_node):
        if node.type in {"function_declaration", "generator_function_declaration", "method_definition"}:
            name = _node_text(source, node.child_by_field_name("name")).strip()
            if not name:
                continue
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            functions.append(
                {
                    "file": str(path.resolve()),
                    "language": language,
                    "kind": "function_definition",
                    "name": name,
                    "signature": _signature_from_lines(lines, start_line),
                    "start_line": start_line,
                    "end_line": end_line,
                    "doc": "",
                    "source": _node_text(source, node)[:MAX_SOURCE_CHARS],
                }
            )
            continue

        if node.type != "variable_declarator":
            continue
        value = node.child_by_field_name("value")
        if value is None or value.type not in {"arrow_function", "function_expression"}:
            continue
        name = _node_text(source, node.child_by_field_name("name")).strip()
        if not name:
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        functions.append(
            {
                "file": str(path.resolve()),
                "language": language,
                "kind": "function_definition",
                "name": name,
                "signature": _signature_from_lines(lines, start_line),
                "start_line": start_line,
                "end_line": end_line,
                "doc": "",
                "source": _node_text(source, node)[:MAX_SOURCE_CHARS],
            }
        )
    return functions


def _extract_functions_go(path: Path, source: bytes, tree: Any) -> list[dict[str, Any]]:
    lines = source.decode("utf-8", errors="ignore").splitlines()
    functions: list[dict[str, Any]] = []
    for node in _iter_nodes(tree.root_node):
        if node.type not in {"function_declaration", "method_declaration"}:
            continue
        name = _node_text(source, node.child_by_field_name("name")).strip()
        if not name:
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        functions.append(
            {
                "file": str(path.resolve()),
                "language": "go",
                "kind": "method_definition" if node.type == "method_declaration" else "function_definition",
                "name": name,
                "signature": _signature_from_lines(lines, start_line),
                "start_line": start_line,
                "end_line": end_line,
                "doc": "",
                "source": _node_text(source, node)[:MAX_SOURCE_CHARS],
            }
        )
    return functions


def _query_source_for(language: str, mode: str) -> str:
    key = f"{language}:{mode}"
    mapping = {
        "c:def": _C_QUERY_FUNCTION_DEFINITION,
        "c:decl": _C_QUERY_FUNCTION_DECLARATION,
        "cpp:def": _CPP_QUERY_FUNCTION_DEFINITION,
        "cpp:decl": _CPP_QUERY_FUNCTION_DECLARATION,
    }
    if key not in mapping:
        raise ValueError(f"unsupported query key: {key}")
    return mapping[key]


def _get_query(language: str, mode: str) -> Query:
    cache_key = f"{language}:{mode}"
    cached = _QUERY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    lang_obj = ts_pack.get_language(language)
    query = Query(lang_obj, _query_source_for(language, mode))
    _QUERY_CACHE[cache_key] = query
    return query


def _capture_nodes(captures: Any, name: str) -> list[Node]:
    if isinstance(captures, dict):
        value = captures.get(name, [])
        return [node for node in value if hasattr(node, "start_byte")]
    nodes: list[Node] = []
    if isinstance(captures, list):
        for item in captures:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            node, capture_name = item
            if capture_name == name and hasattr(node, "start_byte"):
                nodes.append(node)
    return nodes


def _extract_symbols_by_query(path: Path, source: bytes, tree: Any, language: str, mode: str) -> list[dict[str, Any]]:
    query = _get_query(language, mode)
    cursor = QueryCursor(query)
    lines = source.decode("utf-8", errors="ignore").splitlines()
    symbols: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()

    for _, captures in cursor.matches(tree.root_node):
        root_capture = "func.def" if mode == "def" else "func.decl"
        root_nodes = _capture_nodes(captures, root_capture)
        name_nodes = _capture_nodes(captures, "func.name")
        params_nodes = _capture_nodes(captures, "func.params")

        root_node = root_nodes[0] if root_nodes else None
        name_node = name_nodes[0] if name_nodes else None
        params_node = params_nodes[0] if params_nodes else None
        if root_node is None or name_node is None:
            continue

        start_line = root_node.start_point[0] + 1
        end_line = root_node.end_point[0] + 1
        name = _node_text(source, name_node).strip()
        if not name:
            continue

        params_text = _node_text(source, params_node).strip() if params_node is not None else "()"
        line_sig = _signature_from_lines(lines, start_line)
        signature = line_sig or f"{name}{params_text}"
        kind = "function_definition" if mode == "def" else "function_declaration"

        dedupe_key = (name, start_line, end_line, kind)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        symbols.append(
            {
                "file": str(path.resolve()),
                "language": language,
                "kind": kind,
                "name": name,
                "signature": signature,
                "start_line": start_line,
                "end_line": end_line,
                "doc": "",
                "source": _node_text(source, root_node)[:MAX_SOURCE_CHARS],
            }
        )
    return symbols


def _extract_functions_c_cpp_by_query(path: Path, source: bytes, tree: Any, language: str) -> list[dict[str, Any]]:
    definitions = _extract_symbols_by_query(path, source, tree, language, mode="def")
    declarations = _extract_symbols_by_query(path, source, tree, language, mode="decl")
    return definitions + declarations

def _extract_functions(path: Path) -> list[dict[str, Any]]:
    language = _detect_language(path)
    source = path.read_bytes()
    if not source.strip():
        return []

    parser = _parser_for_language(language)
    if parser is None:
        detail = _PARSER_ERRORS.get(language, "parser is not initialized")
        raise RuntimeError(f"missing parser for {language}: {detail}")

    tree = parser.parse(source)

    if language == "python":
        return _extract_functions_python(path, source, tree)
    if language in {"javascript", "typescript", "tsx"}:
        return _extract_functions_js_ts(path, source, tree, language)
    if language == "go":
        return _extract_functions_go(path, source, tree)
    if language in STRICT_NATIVE_LANGUAGES:
        return _extract_functions_c_cpp_by_query(path, source, tree, language)
    return []


def _vectorize_query(query: str, idf: dict[str, float]) -> tuple[dict[str, float], float]:
    tokens = _tokenize(query)
    tf = _calc_tf(tokens)
    vector = _calc_vector(tf, idf)
    return vector, _calc_norm(vector)


def _cosine_similarity(v1: dict[str, float], n1: float, v2: dict[str, float], n2: float) -> float:
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
        n1, n2 = n2, n1
    dot = 0.0
    for token, score in v1.items():
        dot += score * v2.get(token, 0.0)
    return dot / (n1 * n2)


def _normalize_function_name(name: str) -> str:
    return (name or "").strip().lower()


def _deterministic_function_summary(item: dict[str, Any]) -> str:
    name = str(item.get("name", "")).strip()
    signature = str(item.get("signature", "")).strip()
    doc = str(item.get("doc", "")).strip()
    source = str(item.get("source", "")).strip()
    if doc:
        return f"{name}: {doc.splitlines()[0][:160]}".strip()
    first_line = source.splitlines()[0].strip() if source else ""
    return f"{name} {signature}".strip() + (f" | {first_line[:120]}" if first_line else "")


def _summarize_function_with_llm(item: dict[str, Any], model_name: str) -> str:
    name = str(item.get("name", "")).strip()
    language = str(item.get("language", "unknown")).strip()
    signature = str(item.get("signature", "")).strip()
    doc = str(item.get("doc", "")).strip()
    source = str(item.get("source", "")).strip()[:1800]
    prompt = (
        "Summarize this code function for semantic retrieval. "
        "Focus on behavior, inputs, outputs, side effects, and security-relevant intent. "
        "Return one concise sentence under 80 words.\n\n"
        f"Language: {language}\n"
        f"Name: {name}\n"
        f"Signature: {signature}\n"
        f"Doc: {doc or 'N/A'}\n"
        f"Source:\n{source}"
    )
    response = get_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You write concise code summaries for semantic code indexing."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=140,
    )
    content = response.choices[0].message.content if response.choices else ""
    summary = str(content or "").strip()
    return re.sub(r"\s+", " ", summary)[:600]


def _apply_function_summaries(
    raw_functions: list[dict[str, Any]],
    use_llm_summary: bool,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    active_model = (model_name or load_model_name(DEFAULT_MODEL_NAME)).strip()
    for item in raw_functions:
        fallback = _deterministic_function_summary(item)
        if not use_llm_summary:
            item["summary"] = fallback
            continue
        try:
            item["summary"] = _summarize_function_with_llm(item, active_model) or fallback
        except Exception as exc:
            item["summary"] = fallback
            name = str(item.get("name", "unknown")).strip() or "unknown"
            errors.append(f"{name}: {exc}")
    return raw_functions, errors[:50]


def _build_function_name_index(records: list[FunctionRecord]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        norm = _normalize_function_name(item.name)
        if not norm:
            continue
        index[norm].append(
            {
                "name": item.name,
                "kind": item.kind,
                "signature": item.signature,
                "file": item.file,
                "language": item.language,
                "start_line": item.start_line,
                "end_line": item.end_line,
            }
        )
    for value in index.values():
        value.sort(key=lambda row: (str(row.get("file", "")), int(row.get("start_line", 0) or 0)))
    return dict(index)


def _search_functions(index: SemanticIndex, query: str, top_k: int) -> list[dict[str, Any]]:
    query_vector, query_norm = _vectorize_query(query, index.idf)
    scored: list[tuple[float, FunctionRecord]] = []
    for item in index.functions:
        score = _cosine_similarity(query_vector, query_norm, item.vector, item.norm)
        if score <= 0:
            continue
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    result: list[dict[str, Any]] = []
    for score, item in scored[: max(1, top_k)]:
        result.append(
            {
                "score": round(score, 4),
                "kind": item.kind,
                "name": item.name,
                "signature": item.signature,
                "file": item.file,
                "language": item.language,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "doc": item.doc[:240],
                "summary": item.summary[:240],
            }
        )
    return result


def _lookup_function_name(index: SemanticIndex, name: str, exact: bool, top_k: int) -> list[dict[str, Any]]:
    normalized = _normalize_function_name(name)
    if not normalized:
        return []
    candidates: list[dict[str, Any]] = []
    if exact:
        candidates = list(index.function_name_index.get(normalized, []))
    else:
        for key, items in index.function_name_index.items():
            if normalized in key:
                candidates.extend(items)
    return candidates[: max(1, min(int(top_k), 100))]


def _read_description_text(description: str, description_file: str) -> str:
    if description.strip():
        return description.strip()
    if not description_file.strip():
        return ""
    path = Path(description_file).resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"description_file not found: {description_file}")
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _split_requirements(text: str) -> list[str]:
    if not text.strip():
        return []
    chunks = re.split(r"[\n\r]+|[。！？!?；;]+", text)
    lines: list[str] = []
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if len(normalized) < 8:
            continue
        lines.append(normalized)
    if not lines:
        return [text.strip()[:400]]
    return lines[:80]


def _status_by_score(score: float) -> str:
    if score >= 0.22:
        return "covered"
    if score >= 0.12:
        return "partial"
    return "missing"


def _build_index(
    path: str,
    include_glob: str,
    max_files: int,
    use_llm_summary: bool = False,
    model_name: str = "",
) -> SemanticIndex:
    try:
        root = Path(path).resolve()
        if not root.exists():
            raise ValueError(f"path not found: {path}")
        scope = resolve_code_scope(
            path=path,
            include_glob=include_glob,
            default_include_glob=DEFAULT_INCLUDE_GLOB,
            max_files=max_files,
            skip_dirs=SKIP_DIRS,
        )
        if root.is_file():
            file_paths = [root]
        else:
            file_paths = [Path(item) for item in scope.target_files]
        filtered_file_paths: list[Path] = []
        for file_path in file_paths:
            if not file_path.exists() or not file_path.is_file():
                continue
            if file_path.stat().st_size > MAX_FILE_BYTES:
                continue
            filtered_file_paths.append(file_path)
        file_paths = filtered_file_paths
        indexed_files_by_language: dict[str, int] = defaultdict(int)
        strict_languages_in_scope: set[str] = set()
        for file_path in file_paths:
            language = _detect_language(file_path)
            if language == "unknown":
                continue
            indexed_files_by_language[language] += 1
            if language in STRICT_NATIVE_LANGUAGES:
                strict_languages_in_scope.add(language)

        _require_parsers(strict_languages_in_scope)

        raw_functions: list[dict[str, Any]] = []
        for file_path in file_paths:
            raw_functions.extend(_extract_functions(file_path))
        active_model_name = (model_name or load_model_name(DEFAULT_MODEL_NAME)).strip()
        raw_functions, summary_errors = _apply_function_summaries(
            raw_functions=raw_functions,
            use_llm_summary=bool(use_llm_summary),
            model_name=active_model_name,
        )

        documents: list[list[str]] = []
        for item in raw_functions:
            semantic_text = " ".join(
                [
                    item.get("kind", ""),
                    item.get("name", ""),
                    item.get("signature", ""),
                    item.get("doc", ""),
                    item.get("summary", ""),
                    Path(item.get("file", "")).name,
                ]
            )
            body_text = item.get("source", "")
            semantic_tokens = _tokenize(semantic_text)
            body_tokens = _tokenize(body_text)
            documents.append((semantic_tokens * 3) + body_tokens)

        df_counter: Counter[str] = Counter()
        for tokens in documents:
            for token in set(tokens):
                df_counter[token] += 1
        total_docs = max(1, len(documents))
        idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1.0 for token, freq in df_counter.items()}

        language_symbol_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        function_records: list[FunctionRecord] = []
        for item, tokens in zip(raw_functions, documents):
            tf = _calc_tf(tokens)
            vector = _calc_vector(tf, idf)
            language_symbol_counts[item["language"]][item["kind"]] += 1
            function_records.append(
                FunctionRecord(
                    file=item["file"],
                    language=item["language"],
                    kind=item["kind"],
                    name=item["name"],
                    signature=item["signature"],
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                    doc=item["doc"],
                    source=item["source"],
                    token_tf=tf,
                    vector=vector,
                    norm=_calc_norm(vector),
                    summary=str(item.get("summary", "")),
                )
            )

        parser_status = {language: _PARSER_STATUS.get(language, "unknown") for language in SUPPORTED_LANGUAGES}
        parser_errors = {
            language: _PARSER_ERRORS.get(language, "")
            for language in SUPPORTED_LANGUAGES
            if _PARSER_ERRORS.get(language)
        }

        return SemanticIndex(
            root=scope.root_path,
            resolved_path=scope.resolved_path,
            target_path=scope.resolved_path,
            scope_type=scope.scope_type,
            include_glob=scope.include_glob,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            file_count=len(file_paths),
            function_count=len(function_records),
            target_files=[str(item.resolve()) for item in file_paths],
            functions=function_records,
            function_name_index=_build_function_name_index(function_records),
            idf=idf,
            parser_status=parser_status,
            parser_errors=parser_errors,
            language_symbol_counts={
                language: {kind: count for kind, count in kinds.items()}
                for language, kinds in language_symbol_counts.items()
            },
            indexed_files_by_language=dict(indexed_files_by_language),
            summary_mode="llm" if use_llm_summary else "deterministic",
            summary_model=active_model_name if use_llm_summary else "",
            summary_errors=summary_errors,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to build semantic index: {exc}") from exc


def _get_or_create_index(
    path: str,
    include_glob: str,
    max_files: int,
    rebuild: bool,
    use_llm_summary: bool = False,
    model_name: str = "",
) -> tuple[SemanticIndex, bool]:
    scope = resolve_code_scope(
        path=path,
        include_glob=include_glob,
        default_include_glob=DEFAULT_INCLUDE_GLOB,
        max_files=max_files,
        skip_dirs=SKIP_DIRS,
    )
    active_model_name = (model_name or load_model_name(DEFAULT_MODEL_NAME)).strip() if use_llm_summary else ""
    key = _cache_key(scope.scope_type, scope.resolved_path, scope.include_glob, use_llm_summary, active_model_name)
    if not rebuild and key in _INDEX_CACHE:
        return _INDEX_CACHE[key], True
    if not rebuild:
        disk_cached = _load_index_from_disk(
            scope=scope,
            include_glob=scope.include_glob,
            use_llm_summary=use_llm_summary,
            model_name=active_model_name,
        )
        if disk_cached is not None:
            _INDEX_CACHE[key] = disk_cached
            return disk_cached, True
    index = _build_index(
        path=path,
        include_glob=scope.include_glob,
        max_files=max_files,
        use_llm_summary=use_llm_summary,
        model_name=active_model_name,
    )
    _INDEX_CACHE[key] = index
    _save_index_to_disk(
        scope=scope,
        include_glob=scope.include_glob,
        index=index,
        use_llm_summary=use_llm_summary,
        model_name=active_model_name,
    )
    return index, False


def _execute_tool(handler: Callable[[], dict[str, Any]], error_prefix: str) -> str:
    try:
        payload = handler()
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"{error_prefix} error: {exc}"


@tool
def semantic_index_functions(
    path: str,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """
    Build or refresh function-level semantic index for a codebase (Tree-sitter AST).
    Set use_llm_summary=True to include LLM-generated function summaries in retrieval vectors.
    """

    def _handler() -> dict[str, Any]:
        index, cache_hit = _get_or_create_index(
            path=path,
            include_glob=include_glob,
            max_files=max_files,
            rebuild=rebuild,
            use_llm_summary=bool(use_llm_summary),
            model_name=summary_model_name,
        )
        return {
            "status": "ok",
            "root": index.root,
            "resolved_path": index.resolved_path,
            "target_path": index.target_path,
            "scope_type": index.scope_type,
            "include_glob": index.include_glob,
            "created_at": index.created_at,
            "file_count": index.file_count,
            "function_count": index.function_count,
            "function_name_count": len(index.function_name_index),
            "summary_mode": index.summary_mode,
            "summary_model": index.summary_model,
            "summary_error_count": len(index.summary_errors or []),
            "summary_errors": index.summary_errors or [],
            "cache_hit": cache_hit,
            "indexed_targets": index.target_files,
            "cache_path": str(
                _index_cache_file(
                    index_root_path=index.root if index.scope_type == "directory" else index.root,
                    scope_type=index.scope_type,
                    resolved_path=index.resolved_path,
                    include_glob=index.include_glob,
                    use_llm_summary=bool(use_llm_summary),
                    model_name=index.summary_model,
                )
            ),
            "parser_languages": sorted(_PARSER_FACTORIES.keys()),
            "parser_status": index.parser_status,
            "parser_errors": index.parser_errors,
            "language_symbol_counts": index.language_symbol_counts,
            "indexed_files_by_language": index.indexed_files_by_language,
        }

    return _execute_tool(_handler, "semantic_index_functions")


@tool
def semantic_search_functions(
    query: str,
    path: str,
    top_k: int = 8,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """
    Semantic search over indexed functions using natural language query.
    Set use_llm_summary=True to search an index built with LLM-generated function summaries.
    """

    def _handler() -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        index, _ = _get_or_create_index(
            path=path,
            include_glob=include_glob,
            max_files=max_files,
            rebuild=rebuild,
            use_llm_summary=bool(use_llm_summary),
            model_name=summary_model_name,
        )
        hits = _search_functions(index=index, query=query, top_k=max(1, min(int(top_k), 30)))
        return {
            "status": "ok",
            "query": query,
            "root": index.root,
            "resolved_path": index.resolved_path,
            "target_path": index.target_path,
            "scope_type": index.scope_type,
            "function_count": index.function_count,
            "summary_mode": index.summary_mode,
            "summary_model": index.summary_model,
            "top_k": top_k,
            "indexed_targets": index.target_files,
            "hits": hits,
        }

    return _execute_tool(_handler, "semantic_search_functions")


@tool
def semantic_lookup_function_name(
    name: str,
    path: str,
    exact: bool = True,
    top_k: int = 20,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """
    Lookup indexed functions by function name using the function-name index.
    """

    def _handler() -> dict[str, Any]:
        if not name.strip():
            raise ValueError("name cannot be empty")
        index, _ = _get_or_create_index(
            path=path,
            include_glob=include_glob,
            max_files=max_files,
            rebuild=rebuild,
            use_llm_summary=bool(use_llm_summary),
            model_name=summary_model_name,
        )
        matches = _lookup_function_name(index=index, name=name, exact=bool(exact), top_k=top_k)
        return {
            "status": "ok",
            "query": name,
            "exact": bool(exact),
            "root": index.root,
            "resolved_path": index.resolved_path,
            "target_path": index.target_path,
            "scope_type": index.scope_type,
            "indexed_targets": index.target_files,
            "summary_mode": index.summary_mode,
            "summary_model": index.summary_model,
            "function_name_count": len(index.function_name_index),
            "match_count": len(matches),
            "matches": matches,
        }

    return _execute_tool(_handler, "semantic_lookup_function_name")


@tool
def semantic_diff_with_description(
    path: str,
    description: str = "",
    description_file: str = "",
    top_k: int = 3,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """
    Compare implementation functions with user description (PRD/API doc/etc.) and output semantic diff.
    Set use_llm_summary=True to compare against LLM-enriched function summaries.
    """

    def _handler() -> dict[str, Any]:
        doc_text = _read_description_text(description=description, description_file=description_file)
        if not doc_text:
            raise ValueError("description or description_file is required")

        index, _ = _get_or_create_index(
            path=path,
            include_glob=include_glob,
            max_files=max_files,
            rebuild=rebuild,
            use_llm_summary=bool(use_llm_summary),
            model_name=summary_model_name,
        )
        requirements = _split_requirements(doc_text)
        if not requirements:
            raise ValueError("failed to parse requirements from description")

        requirement_matches: list[dict[str, Any]] = []
        matched_functions: set[tuple[str, str, int]] = set()
        for requirement in requirements:
            hits = _search_functions(index=index, query=requirement, top_k=max(1, min(int(top_k), 10)))
            best_score = float(hits[0]["score"]) if hits else 0.0
            status = _status_by_score(best_score)
            if hits:
                first = hits[0]
                matched_functions.add((first["file"], first["name"], int(first["start_line"])))
            requirement_matches.append(
                {
                    "requirement": requirement,
                    "status": status,
                    "score": round(best_score, 4),
                    "top_matches": hits,
                }
            )

        covered = sum(1 for item in requirement_matches if item["status"] == "covered")
        partial = sum(1 for item in requirement_matches if item["status"] == "partial")
        missing = sum(1 for item in requirement_matches if item["status"] == "missing")
        total_reqs = max(1, len(requirement_matches))
        strict_coverage_rate = round((covered / total_reqs) * 100.0, 2)
        weighted_coverage_rate = round(((covered + partial * 0.5) / total_reqs) * 100.0, 2)

        undocumented_candidates: list[dict[str, Any]] = []
        for record in index.functions:
            key = (record.file, record.name, record.start_line)
            if key in matched_functions:
                continue
            if len(undocumented_candidates) >= 20:
                break
            undocumented_candidates.append(
                {
                    "name": record.name,
                    "kind": record.kind,
                    "file": record.file,
                    "start_line": record.start_line,
                    "signature": record.signature,
                    "language": record.language,
                }
            )

        return {
            "status": "ok",
            "root": index.root,
            "resolved_path": index.resolved_path,
            "target_path": index.target_path,
            "scope_type": index.scope_type,
            "include_glob": index.include_glob,
            "indexed_targets": index.target_files,
            "index": {
                "created_at": index.created_at,
                "file_count": index.file_count,
                "function_count": index.function_count,
                "summary_mode": index.summary_mode,
                "summary_model": index.summary_model,
                "summary_error_count": len(index.summary_errors or []),
                "language_symbol_counts": index.language_symbol_counts,
                "indexed_files_by_language": index.indexed_files_by_language,
            },
            "summary": {
                "requirements": len(requirement_matches),
                "covered": covered,
                "partial": partial,
                "missing": missing,
                "strict_coverage_rate_percent": strict_coverage_rate,
                "weighted_coverage_rate_percent": weighted_coverage_rate,
            },
            "requirement_matches": requirement_matches,
            "missing_requirements": [item["requirement"] for item in requirement_matches if item["status"] == "missing"],
            "possibly_undocumented_functions": undocumented_candidates,
            "notes": [
                "This diff is semantic retrieval based (TF-IDF over function content), not formal verification.",
                "Function extraction is powered by Tree-sitter AST parsers.",
            ],
        }

    return _execute_tool(_handler, "semantic_diff_with_description")
