import ast
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from langchain.tools import tool


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
DEFAULT_INCLUDE_GLOB = "*.py,*.js,*.jsx,*.ts,*.tsx"
MAX_FILE_BYTES = 1024 * 1024
MAX_SOURCE_CHARS = 2400

_INDEX_CACHE: dict[str, "SemanticIndex"] = {}
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


@dataclass
class FunctionRecord:
    file: str
    language: str
    name: str
    signature: str
    start_line: int
    end_line: int
    doc: str
    source: str
    token_tf: dict[str, float]
    vector: dict[str, float]
    norm: float


@dataclass
class SemanticIndex:
    root: str
    include_glob: str
    created_at: str
    file_count: int
    function_count: int
    functions: list[FunctionRecord]
    idf: dict[str, float]


def _cache_key(root: str, include_glob: str) -> str:
    return f"{Path(root).resolve()}::{include_glob.strip().lower()}"


def _normalize_include_glob(include_glob: str) -> list[str]:
    raw = include_glob.strip() or DEFAULT_INCLUDE_GLOB
    return [part.strip() for part in raw.split(",") if part.strip()]


def _is_code_file(path: Path, patterns: list[str]) -> bool:
    return any(fnmatch(path.name, pattern) for pattern in patterns)


def _detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    return "unknown"


def _iter_code_files(root: Path, include_glob: str, max_files: int) -> list[Path]:
    patterns = _normalize_include_glob(include_glob)
    files: list[Path] = []
    for current_root, dirs, names in os.walk(root):
        dirs[:] = [item for item in dirs if item not in SKIP_DIRS]
        for name in names:
            full = Path(current_root) / name
            if not _is_code_file(full, patterns):
                continue
            try:
                if full.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(full)
            if len(files) >= max_files:
                return files
    return files


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
                words.extend(
                    normalized[index : index + 2]
                    for index in range(0, len(normalized) - 1)
                )
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


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source_line: str) -> str:
    line = source_line.strip()
    if line.startswith("def ") or line.startswith("async def "):
        return line
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}(...)"


def _extract_python_functions(path: Path, text: str) -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    try:
        tree = ast.parse(text)
    except Exception:
        return functions
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        def_line = lines[start_line - 1] if 0 <= start_line - 1 < len(lines) else ""
        source = "\n".join(lines[start_line - 1 : end_line]).strip()
        source = source[:MAX_SOURCE_CHARS]
        functions.append(
            {
                "file": str(path.resolve()),
                "language": "python",
                "name": node.name,
                "signature": _python_signature(node, def_line),
                "start_line": start_line,
                "end_line": end_line,
                "doc": (ast.get_docstring(node) or "").strip(),
                "source": source,
            }
        )
    return functions


def _line_number(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _estimate_js_block_end(text: str, start_idx: int) -> int:
    brace_start = text.find("{", start_idx)
    if brace_start < 0:
        return _line_number(text, start_idx)
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _line_number(text, idx)
    return _line_number(text, len(text))


def _extract_js_ts_functions(path: Path, text: str, language: str) -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    patterns = [
        re.compile(r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"),
        re.compile(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{"),
        re.compile(r"([A-Za-z_$][\w$]*)\s*:\s*(?:async\s*)?function\s*\([^)]*\)\s*\{"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = match.group(1)
            start = match.start()
            start_line = _line_number(text, start)
            end_line = _estimate_js_block_end(text, start)
            signature = match.group(0).strip()
            source_lines = text.splitlines()[start_line - 1 : end_line]
            source = "\n".join(source_lines).strip()[:MAX_SOURCE_CHARS]
            functions.append(
                {
                    "file": str(path.resolve()),
                    "language": language,
                    "name": name,
                    "signature": signature,
                    "start_line": start_line,
                    "end_line": end_line,
                    "doc": "",
                    "source": source,
                }
            )
    return functions


def _extract_functions(path: Path) -> list[dict[str, Any]]:
    text = _safe_read_text(path)
    if not text.strip():
        return []
    language = _detect_language(path)
    if language == "python":
        return _extract_python_functions(path, text)
    if language in {"javascript", "typescript"}:
        return _extract_js_ts_functions(path, text, language)
    return []


def _build_index(path: str, include_glob: str, max_files: int) -> SemanticIndex:
    root = Path(path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"invalid directory path '{path}'")

    file_paths = _iter_code_files(root=root, include_glob=include_glob, max_files=max_files)
    raw_functions: list[dict[str, Any]] = []
    for file_path in file_paths:
        raw_functions.extend(_extract_functions(file_path))

    documents: list[list[str]] = []
    for item in raw_functions:
        semantic_text = " ".join(
            [
                item.get("name", ""),
                item.get("signature", ""),
                item.get("doc", ""),
                Path(item.get("file", "")).name,
            ]
        )
        body_text = " ".join(
            [
                item.get("source", ""),
            ]
        )
        semantic_tokens = _tokenize(semantic_text)
        body_tokens = _tokenize(body_text)
        # Favor function identity and intent signals over long implementation details.
        documents.append((semantic_tokens * 3) + body_tokens)

    df_counter: Counter[str] = Counter()
    for tokens in documents:
        for token in set(tokens):
            df_counter[token] += 1
    total_docs = max(1, len(documents))
    idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1.0 for token, freq in df_counter.items()}

    function_records: list[FunctionRecord] = []
    for item, tokens in zip(raw_functions, documents):
        tf = _calc_tf(tokens)
        vector = _calc_vector(tf, idf)
        function_records.append(
            FunctionRecord(
                file=item["file"],
                language=item["language"],
                name=item["name"],
                signature=item["signature"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                doc=item["doc"],
                source=item["source"],
                token_tf=tf,
                vector=vector,
                norm=_calc_norm(vector),
            )
        )

    return SemanticIndex(
        root=str(root),
        include_glob=include_glob,
        created_at=datetime.utcnow().isoformat() + "Z",
        file_count=len(file_paths),
        function_count=len(function_records),
        functions=function_records,
        idf=idf,
    )


def _get_or_create_index(path: str, include_glob: str, max_files: int, rebuild: bool) -> tuple[SemanticIndex, bool]:
    key = _cache_key(path, include_glob)
    if not rebuild and key in _INDEX_CACHE:
        return _INDEX_CACHE[key], True
    index = _build_index(path=path, include_glob=include_glob, max_files=max_files)
    _INDEX_CACHE[key] = index
    return index, False


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
                "name": item.name,
                "signature": item.signature,
                "file": item.file,
                "language": item.language,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "doc": item.doc[:240],
            }
        )
    return result


def _read_description_text(description: str, description_file: str) -> str:
    if description.strip():
        return description.strip()
    if not description_file.strip():
        return ""
    path = Path(description_file).resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"description_file not found: {description_file}")
    return _safe_read_text(path).strip()


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


@tool
def semantic_index_functions(
    path: str,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
) -> str:
    """
    Build or refresh function-level semantic index for a codebase.
    """
    try:
        index, cache_hit = _get_or_create_index(
            path=path,
            include_glob=include_glob,
            max_files=max_files,
            rebuild=rebuild,
        )
        payload = {
            "status": "ok",
            "root": index.root,
            "include_glob": index.include_glob,
            "created_at": index.created_at,
            "file_count": index.file_count,
            "function_count": index.function_count,
            "cache_hit": cache_hit,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"semantic_index_functions error: {exc}"


@tool
def semantic_search_functions(
    query: str,
    path: str,
    top_k: int = 8,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
) -> str:
    """
    Semantic search over indexed functions using natural language query.
    """
    try:
        if not query.strip():
            return "semantic_search_functions error: query cannot be empty"
        index, _ = _get_or_create_index(path=path, include_glob=include_glob, max_files=max_files, rebuild=rebuild)
        hits = _search_functions(index=index, query=query, top_k=max(1, min(int(top_k), 30)))
        payload = {
            "status": "ok",
            "query": query,
            "root": index.root,
            "function_count": index.function_count,
            "top_k": top_k,
            "hits": hits,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"semantic_search_functions error: {exc}"


@tool
def semantic_diff_with_description(
    path: str,
    description: str = "",
    description_file: str = "",
    top_k: int = 3,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
) -> str:
    """
    Compare implementation functions with user description (PRD/API doc/etc.) and output semantic diff.
    """
    try:
        doc_text = _read_description_text(description=description, description_file=description_file)
        if not doc_text:
            return "semantic_diff_with_description error: description or description_file is required"
        index, _ = _get_or_create_index(path=path, include_glob=include_glob, max_files=max_files, rebuild=rebuild)
        requirements = _split_requirements(doc_text)
        if not requirements:
            return "semantic_diff_with_description error: failed to parse requirements from description"

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
                    "file": record.file,
                    "start_line": record.start_line,
                    "signature": record.signature,
                }
            )

        payload = {
            "status": "ok",
            "root": index.root,
            "include_glob": index.include_glob,
            "index": {
                "created_at": index.created_at,
                "file_count": index.file_count,
                "function_count": index.function_count,
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
                "For better precision, keep description statements atomic and explicit.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"semantic_diff_with_description error: {exc}"


# ======================== Tree-sitter implementation override ========================
# Keep public tool names unchanged while replacing the parser backend.
from typing import Callable

from tree_sitter import Language, Node, Parser
import tree_sitter_c as _ts_c
import tree_sitter_javascript as _ts_javascript
import tree_sitter_python as _ts_python
import tree_sitter_typescript as _ts_typescript


DEFAULT_INCLUDE_GLOB = "*.py,*.js,*.jsx,*.ts,*.tsx,*.c,*.h,*.cc,*.cpp,*.hpp"
MAX_SOURCE_CHARS = 2400
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

_INDEX_CACHE = {}
_PARSER_FACTORIES: dict[str, Callable[[], Parser]] = {}


def _build_parser(language_obj: Any) -> Parser:
    parser = Parser()
    parser.language = Language(language_obj)
    return parser


def _register_parser_factory(language: str, factory: Callable[[], Parser]) -> None:
    if language not in _PARSER_FACTORIES:
        _PARSER_FACTORIES[language] = factory


def _init_parser_factories() -> None:
    if _PARSER_FACTORIES:
        return
    _register_parser_factory("python", lambda: _build_parser(_ts_python.language()))
    _register_parser_factory("javascript", lambda: _build_parser(_ts_javascript.language()))
    _register_parser_factory("typescript", lambda: _build_parser(_ts_typescript.language_typescript()))
    _register_parser_factory("tsx", lambda: _build_parser(_ts_typescript.language_tsx()))
    _register_parser_factory("c", lambda: _build_parser(_ts_c.language()))


def _parser_for_language(language: str) -> Parser | None:
    _init_parser_factories()
    factory = _PARSER_FACTORIES.get(language)
    if factory is None:
        return None
    return factory()


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
                "name": name,
                "signature": _signature_from_lines(lines, start_line),
                "start_line": start_line,
                "end_line": end_line,
                "doc": "",
                "source": _node_text(source, node)[:MAX_SOURCE_CHARS],
            }
        )
    return functions


def _find_first_identifier(source: bytes, root: Node) -> str:
    for node in _iter_nodes(root):
        if node.type == "identifier":
            text = _node_text(source, node).strip()
            if text:
                return text
    return ""


def _extract_functions_c_cpp(path: Path, source: bytes, tree: Any, language: str) -> list[dict[str, Any]]:
    lines = source.decode("utf-8", errors="ignore").splitlines()
    functions: list[dict[str, Any]] = []
    for node in _iter_nodes(tree.root_node):
        if node.type != "function_definition":
            continue
        declarator = node.child_by_field_name("declarator")
        name = _find_first_identifier(source, declarator or node)
        if not name:
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        functions.append(
            {
                "file": str(path.resolve()),
                "language": language,
                "name": name,
                "signature": _signature_from_lines(lines, start_line),
                "start_line": start_line,
                "end_line": end_line,
                "doc": "",
                "source": _node_text(source, node)[:MAX_SOURCE_CHARS],
            }
        )
    return functions


def _extract_functions(path: Path) -> list[dict[str, Any]]:
    language = _detect_language(path)
    parser = _parser_for_language(language)
    if parser is None:
        return []
    try:
        source = path.read_bytes()
        if not source.strip():
            return []
        tree = parser.parse(source)
    except Exception:
        return []

    if language == "python":
        return _extract_functions_python(path, source, tree)
    if language in {"javascript", "typescript", "tsx"}:
        return _extract_functions_js_ts(path, source, tree, language)
    if language in {"c", "cpp"}:
        return _extract_functions_c_cpp(path, source, tree, language)
    return []


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
                for i in range(0, len(normalized) - 1):
                    words.append(normalized[i : i + 2])
            for key, aliases in CHINESE_KEYWORD_ALIASES.items():
                if key in normalized:
                    words.extend(aliases)
    return [item for item in words if len(item) >= 2]


def _build_index(path: str, include_glob: str, max_files: int) -> SemanticIndex:
    root = Path(path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"invalid directory path '{path}'")

    file_paths = _iter_code_files(root=root, include_glob=include_glob, max_files=max_files)
    raw_functions: list[dict[str, Any]] = []
    for file_path in file_paths:
        raw_functions.extend(_extract_functions(file_path))

    documents: list[list[str]] = []
    for item in raw_functions:
        semantic_text = " ".join(
            [
                item.get("name", ""),
                item.get("signature", ""),
                item.get("doc", ""),
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

    function_records: list[FunctionRecord] = []
    for item, tokens in zip(raw_functions, documents):
        tf = _calc_tf(tokens)
        vector = _calc_vector(tf, idf)
        function_records.append(
            FunctionRecord(
                file=item["file"],
                language=item["language"],
                name=item["name"],
                signature=item["signature"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                doc=item["doc"],
                source=item["source"],
                token_tf=tf,
                vector=vector,
                norm=_calc_norm(vector),
            )
        )

    return SemanticIndex(
        root=str(root),
        include_glob=include_glob,
        created_at=datetime.utcnow().isoformat() + "Z",
        file_count=len(file_paths),
        function_count=len(function_records),
        functions=function_records,
        idf=idf,
    )


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
) -> str:
    """
    Build or refresh function-level semantic index for a codebase (Tree-sitter AST).
    """
    def _handler() -> dict[str, Any]:
        index, cache_hit = _get_or_create_index(
            path=path,
            include_glob=include_glob,
            max_files=max_files,
            rebuild=rebuild,
        )
        parser_languages = sorted(_PARSER_FACTORIES.keys())
        return {
            "status": "ok",
            "root": index.root,
            "include_glob": index.include_glob,
            "created_at": index.created_at,
            "file_count": index.file_count,
            "function_count": index.function_count,
            "cache_hit": cache_hit,
            "parser_languages": parser_languages,
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
) -> str:
    """
    Semantic search over indexed functions using natural language query.
    """
    def _handler() -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        index, _ = _get_or_create_index(path=path, include_glob=include_glob, max_files=max_files, rebuild=rebuild)
        hits = _search_functions(index=index, query=query, top_k=max(1, min(int(top_k), 30)))
        return {
            "status": "ok",
            "query": query,
            "root": index.root,
            "function_count": index.function_count,
            "top_k": top_k,
            "hits": hits,
        }
    return _execute_tool(_handler, "semantic_search_functions")


@tool
def semantic_diff_with_description(
    path: str,
    description: str = "",
    description_file: str = "",
    top_k: int = 3,
    include_glob: str = DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
    rebuild: bool = False,
) -> str:
    """
    Compare implementation functions with user description (PRD/API doc/etc.) and output semantic diff.
    """
    def _handler() -> dict[str, Any]:
        doc_text = _read_description_text(description=description, description_file=description_file)
        if not doc_text:
            raise ValueError("description or description_file is required")
        index, _ = _get_or_create_index(path=path, include_glob=include_glob, max_files=max_files, rebuild=rebuild)
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
                    "file": record.file,
                    "start_line": record.start_line,
                    "signature": record.signature,
                    "language": record.language,
                }
            )

        return {
            "status": "ok",
            "root": index.root,
            "include_glob": index.include_glob,
            "index": {
                "created_at": index.created_at,
                "file_count": index.file_count,
                "function_count": index.function_count,
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
