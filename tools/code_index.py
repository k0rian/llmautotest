import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.tools import tool

from llm_utils import resolve_code_scope
from tools import semantic_diff_ts as semantic_mod

INDEX_VERSION = 2
HIER_PREFIX = "hierarchical"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _node_id(kind: str, path: str, name: str = "", extra: str = "") -> str:
    digest = _sha1_text(f"{kind}|{path}|{name}|{extra}")
    return f"{kind}:{digest[:16]}"


def _cache_file(path: str, include_glob: str, use_llm: bool = False, model_name: str = "") -> Path:
    scope = resolve_code_scope(
        path=path,
        include_glob=include_glob,
        default_include_glob=semantic_mod.DEFAULT_INCLUDE_GLOB,
        max_files=2000,
        skip_dirs=semantic_mod.SKIP_DIRS,
    )
    cache_dir = Path(scope.index_root_path).resolve() / semantic_mod.INDEX_CACHE_DIRNAME / semantic_mod.INDEX_CACHE_SUBDIR
    summary_tag = semantic_mod._summary_cache_tag(use_llm, model_name)
    digest = _sha1_text(f"{HIER_PREFIX}|{scope.scope_type}|{scope.resolved_path}|{include_glob.lower()}|{summary_tag}")
    return cache_dir / f"{digest}.hier.json"


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _deterministic_function_summary(item: dict[str, Any]) -> str:
    name = str(item.get("name", "")).strip()
    signature = str(item.get("signature", "")).strip()
    doc = str(item.get("doc", "")).strip()
    source = str(item.get("source", "")).strip()
    if doc:
        return f"{name}: {doc.splitlines()[0][:160]}"
    first_line = source.splitlines()[0].strip() if source else ""
    return f"{name} {signature}".strip() + (f" | {first_line[:120]}" if first_line else "")


def _aggregate_summary(items: list[str], fallback: str) -> str:
    cleaned = [value.strip() for value in items if value and value.strip()]
    if not cleaned:
        return fallback
    merged = " | ".join(cleaned[:5])
    return merged[:600]


def _compute_file_hash(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    return hashlib.sha1(data).hexdigest()


def _extract_functions_for_file(file_path: Path) -> list[dict[str, Any]]:
    try:
        return semantic_mod._extract_functions(file_path)
    except Exception:
        return []


def _collect_all_directory_paths(files: list[Path], root: Path) -> list[str]:
    root_resolved = str(root.resolve())
    dirs: set[str] = set()
    for file_path in files:
        cursor = file_path.parent.resolve()
        while True:
            current = str(cursor)
            if not current.startswith(root_resolved):
                break
            dirs.add(current)
            if current == root_resolved:
                break
            cursor = cursor.parent
    return sorted(dirs, key=lambda item: len(Path(item).parts), reverse=True)


def _build_hierarchical_index_payload(
    path: str,
    rebuild: bool,
    use_llm: bool,
    include_glob: str,
    max_files: int,
    summary_model_name: str = "",
) -> dict[str, Any]:
    scope = resolve_code_scope(
        path=path,
        include_glob=include_glob,
        default_include_glob=semantic_mod.DEFAULT_INCLUDE_GLOB,
        max_files=max_files,
        skip_dirs=semantic_mod.SKIP_DIRS,
    )
    root = Path(scope.root_path)
    summary_model = (summary_model_name or semantic_mod.load_model_name(semantic_mod.DEFAULT_MODEL_NAME)) if use_llm else ""

    files = [Path(item) for item in scope.target_files]
    old_cache_path = _cache_file(path, include_glob, use_llm=use_llm, model_name=summary_model)
    previous = _safe_read_json(old_cache_path) if old_cache_path.exists() and not rebuild else {}
    previous_nodes = previous.get("nodes", {}) if isinstance(previous.get("nodes", {}), dict) else {}
    previous_meta = previous.get("artifacts", {}) if isinstance(previous.get("artifacts", {}), dict) else {}
    previous_file_hashes = (
        previous_meta.get("file_hashes", {}) if isinstance(previous_meta.get("file_hashes", {}), dict) else {}
    )

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    file_to_functions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_hashes: dict[str, str] = {}

    reused_file_count = 0
    rebuilt_file_count = 0

    for file_path in files:
        file_abs = str(file_path.resolve())
        file_hash = _compute_file_hash(file_path)
        file_hashes[file_abs] = file_hash
        unchanged = (
            not rebuild
            and file_hash
            and str(previous_file_hashes.get(file_abs, "")) == file_hash
        )
        if unchanged:
            reused_file_count += 1
            # Reuse function nodes for this file from previous cache.
            for node in previous_nodes.values():
                if not isinstance(node, dict):
                    continue
                if node.get("kind") != "function":
                    continue
                if str(node.get("path", "")) != file_abs:
                    continue
                file_to_functions[file_abs].append(node)
            continue

        rebuilt_file_count += 1
        raw_functions = _extract_functions_for_file(file_path)
        raw_functions, _summary_errors = semantic_mod._apply_function_summaries(
            raw_functions=raw_functions,
            use_llm_summary=bool(use_llm),
            model_name=summary_model,
        )
        for item in raw_functions:
            node_hash = _sha1_text(
                "|".join(
                    [
                        str(item.get("file", "")),
                        str(item.get("name", "")),
                        str(item.get("start_line", "")),
                        str(item.get("signature", "")),
                        str(item.get("doc", "")),
                        str(item.get("source", "")),
                    ]
                )
            )
            node = {
                "id": _node_id(
                    "function",
                    str(item.get("file", "")),
                    str(item.get("name", "")),
                    str(item.get("start_line", "")),
                ),
                "kind": "function",
                "path": str(item.get("file", "")),
                "name": str(item.get("name", "")),
                "summary": str(item.get("summary", "")) or _deterministic_function_summary(item),
                "children": [],
                "language": str(item.get("language", "unknown")),
                "symbol_count": 1,
                "last_updated": _utc_now(),
                "hash": node_hash,
                "line_range": [int(item.get("start_line", 0) or 0), int(item.get("end_line", 0) or 0)],
                "signature": str(item.get("signature", "")),
                "source_summary": str(item.get("source", ""))[:280],
            }
            file_to_functions[file_abs].append(node)

    # Write function nodes into node table first.
    for function_nodes in file_to_functions.values():
        for fn in function_nodes:
            nodes[str(fn["id"])] = fn

    file_node_ids: list[str] = []
    file_nodes_by_path: dict[str, str] = {}
    dir_file_children: dict[str, list[str]] = defaultdict(list)

    for file_path in files:
        file_abs = str(file_path.resolve())
        function_nodes = file_to_functions.get(file_abs, [])
        child_ids = [str(item["id"]) for item in function_nodes]
        file_summary = _aggregate_summary([str(item.get("summary", "")) for item in function_nodes], file_path.name)
        file_hash = _sha1_text(
            "|".join([file_hashes.get(file_abs, ""), file_summary, ",".join(sorted(child_ids))])
        )
        file_id = _node_id("file", file_abs, file_path.name)
        file_node = {
            "id": file_id,
            "kind": "file",
            "path": file_abs,
            "name": file_path.name,
            "summary": file_summary,
            "children": child_ids,
            "language": semantic_mod._detect_language(file_path),
            "symbol_count": len(child_ids),
            "last_updated": _utc_now(),
            "hash": file_hash,
        }
        nodes[file_id] = file_node
        file_node_ids.append(file_id)
        file_nodes_by_path[file_abs] = file_id
        for child_id in child_ids:
            edges.append({"parent": file_id, "child": child_id})
        parent_dir = str(file_path.parent.resolve())
        dir_file_children[parent_dir].append(file_id)

    directory_paths = _collect_all_directory_paths(files=files, root=root)
    dir_dir_children: dict[str, list[str]] = defaultdict(list)
    for dir_path in directory_paths:
        if dir_path == str(root):
            continue
        parent = str(Path(dir_path).parent.resolve())
        if parent.startswith(str(root)):
            dir_dir_children[parent].append(dir_path)

    dir_node_ids: dict[str, str] = {}
    for dir_path in directory_paths:
        raw_child_file_ids = list(dict.fromkeys(dir_file_children.get(dir_path, [])))
        raw_child_dir_paths = list(dict.fromkeys(dir_dir_children.get(dir_path, [])))
        child_dir_ids = [dir_node_ids[path] for path in raw_child_dir_paths if path in dir_node_ids]
        children = raw_child_file_ids + child_dir_ids
        child_summaries = [str(nodes[cid].get("summary", "")) for cid in children if cid in nodes]
        symbol_count = sum(int(nodes[cid].get("symbol_count", 0) or 0) for cid in children if cid in nodes)
        name = Path(dir_path).name or Path(dir_path).anchor or "."
        summary = _aggregate_summary(child_summaries, f"Directory {name}")
        node_hash = _sha1_text("|".join(sorted(str(nodes[cid].get("hash", "")) for cid in children if cid in nodes)))
        node_id = _node_id("directory", dir_path, name)
        if dir_path == str(root):
            parent_dir = ""
            depth = 0
        else:
            parent_dir = str(Path(dir_path).parent.resolve())
            depth = len(Path(dir_path).relative_to(root).parts)
        node = {
            "id": node_id,
            "kind": "directory",
            "path": dir_path,
            "name": name,
            "summary": summary,
            "children": children,
            "language": "mixed",
            "symbol_count": symbol_count,
            "last_updated": _utc_now(),
            "hash": node_hash,
            "parent": parent_dir,
            "depth": depth,
        }
        nodes[node_id] = node
        dir_node_ids[dir_path] = node_id
        for child in children:
            edges.append({"parent": node_id, "child": child})

    repo_path = str(root)
    top_level_dir_ids: list[str] = []
    top_level_root_file_ids: list[str] = []
    for dir_path, node_id in dir_node_ids.items():
        if dir_path == repo_path:
            continue
        rel = Path(dir_path).relative_to(root)
        if len(rel.parts) == 1:
            top_level_dir_ids.append(node_id)
    for file_abs, file_id in file_nodes_by_path.items():
        try:
            rel = Path(file_abs).resolve().relative_to(root)
        except Exception:
            continue
        if len(rel.parts) == 1:
            top_level_root_file_ids.append(file_id)
    repo_children = list(dict.fromkeys(top_level_dir_ids + top_level_root_file_ids))
    repo_children = [cid for cid in repo_children if cid in nodes]
    repo_summary = _aggregate_summary([str(nodes[cid].get("summary", "")) for cid in repo_children], root.name)
    repo_hash = _sha1_text("|".join(sorted(str(nodes[cid].get("hash", "")) for cid in repo_children)))
    repo_id = _node_id("repository", repo_path, root.name)
    nodes[repo_id] = {
        "id": repo_id,
        "kind": "repository",
        "path": repo_path,
        "name": root.name or repo_path,
        "summary": repo_summary,
        "children": repo_children,
        "language": "mixed",
        "symbol_count": sum(int(nodes[cid].get("symbol_count", 0) or 0) for cid in repo_children),
        "last_updated": _utc_now(),
        "hash": repo_hash,
    }
    for child in repo_children:
        edges.append({"parent": repo_id, "child": child})

    payload = {
        "version": INDEX_VERSION,
        "root": repo_path,
        "resolved_path": scope.resolved_path,
        "scope_type": scope.scope_type,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "include_glob": include_glob,
        "indexed_targets": [str(item.resolve()) for item in files],
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "function_count": sum(1 for node in nodes.values() if node.get("kind") == "function"),
            "file_count": sum(1 for node in nodes.values() if node.get("kind") == "file"),
            "directory_count": sum(1 for node in nodes.values() if node.get("kind") == "directory"),
            "repository_count": sum(1 for node in nodes.values() if node.get("kind") == "repository"),
            "reused_file_count": reused_file_count,
            "rebuilt_file_count": rebuilt_file_count,
            "cache_hit": reused_file_count > 0 and rebuilt_file_count == 0,
            "summary_mode": "llm" if use_llm else "deterministic",
            "summary_model": summary_model,
        },
        "artifacts": {
            "cache_path": str(old_cache_path),
            "file_hashes": file_hashes,
        },
    }

    old_cache_path.parent.mkdir(parents=True, exist_ok=True)
    old_cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_hierarchical_code_index(
    path: str,
    include_glob: str = semantic_mod.DEFAULT_INCLUDE_GLOB,
    use_llm: bool = False,
    summary_model_name: str = "",
) -> dict[str, Any]:
    model_name = (summary_model_name or semantic_mod.load_model_name(semantic_mod.DEFAULT_MODEL_NAME)) if use_llm else ""
    cache = _cache_file(path, include_glob, use_llm=use_llm, model_name=model_name)
    if not cache.exists():
        return {}
    return _safe_read_json(cache)


@tool
def build_hierarchical_code_index(
    path: str,
    rebuild: bool = False,
    use_llm: bool = False,
    summary_model_name: str = "",
    include_glob: str = semantic_mod.DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
) -> str:
    """Build hierarchical semantic code index (function/file/directory/repository) with disk cache."""
    try:
        payload = _build_hierarchical_index_payload(
            path=path,
            rebuild=bool(rebuild),
            use_llm=bool(use_llm),
            include_glob=include_glob,
            max_files=max(1, int(max_files)),
            summary_model_name=summary_model_name,
        )
        return json.dumps(
            {
                "status": "ok",
                "root": payload.get("root", ""),
                "resolved_path": payload.get("resolved_path", ""),
                "scope_type": payload.get("scope_type", "directory"),
                "version": payload.get("version", INDEX_VERSION),
                "indexed_targets": payload.get("indexed_targets", []),
                "stats": payload.get("stats", {}),
                "cache_path": payload.get("artifacts", {}).get("cache_path", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"build_hierarchical_code_index error: {exc}"
