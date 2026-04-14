import json
from pathlib import Path
from typing import Any

from langchain.tools import tool

from tools import semantic_diff_ts as semantic_mod
from tools.code_index import build_hierarchical_code_index, load_hierarchical_code_index


def _tokenize(text: str) -> list[str]:
    return semantic_mod._tokenize(text)


def _score_text(query_tokens: list[str], text: str) -> tuple[float, list[str]]:
    text_tokens = set(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0, []
    overlap = [token for token in query_tokens if token in text_tokens]
    if not overlap:
        return 0.0, []
    precision = len(set(overlap)) / max(1, len(text_tokens))
    recall = len(set(overlap)) / max(1, len(set(query_tokens)))
    score = (0.7 * recall) + (0.3 * precision)
    return round(score, 4), sorted(set(overlap))[:8]


def _load_or_build(path: str) -> dict[str, Any]:
    payload = load_hierarchical_code_index(path)
    if payload:
        return payload
    result = build_hierarchical_code_index.func(path=path, rebuild=False, use_llm=False)
    parsed = json.loads(result) if result.strip().startswith("{") else {}
    if parsed.get("status") != "ok":
        raise RuntimeError(f"failed to build hierarchical index: {result}")
    payload = load_hierarchical_code_index(path)
    if not payload:
        raise RuntimeError("hierarchical index cache file missing after build")
    return payload


def _pick_top(nodes: list[dict[str, Any]], query_tokens: list[str], top_k: int) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for node in nodes:
        text = " ".join(
            [
                str(node.get("name", "")),
                str(node.get("path", "")),
                str(node.get("summary", "")),
            ]
        )
        score, overlap = _score_text(query_tokens, text)
        if score <= 0:
            continue
        scored.append(
            {
                "id": node.get("id", ""),
                "kind": node.get("kind", ""),
                "path": node.get("path", ""),
                "name": node.get("name", ""),
                "summary": str(node.get("summary", ""))[:240],
                "score": score,
                "why": f"matched tokens: {', '.join(overlap)}",
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(1, top_k)]


def _build_parent_chain(node: dict[str, Any], nodes_map: dict[str, dict[str, Any]]) -> list[str]:
    if not isinstance(node, dict):
        return []
    labels: list[str] = []
    kind = str(node.get("kind", ""))
    name = str(node.get("name", "")) or str(node.get("path", ""))
    if name:
        labels.append(name)

    if kind == "directory":
        current = node
        for _ in range(12):
            parent_path = str(current.get("parent", "")).strip()
            if not parent_path:
                break
            parent_node = None
            for candidate in nodes_map.values():
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("kind") != "directory":
                    continue
                if str(candidate.get("path", "")) == parent_path:
                    parent_node = candidate
                    break
            if not parent_node:
                break
            label = str(parent_node.get("name", "")) or str(parent_node.get("path", ""))
            if label:
                labels.append(label)
            current = parent_node
        return list(reversed(labels))

    if kind in {"file", "function"}:
        file_path = str(node.get("path", "")).strip()
        if file_path:
            file_parent = str(Path(file_path).parent)
            for candidate in nodes_map.values():
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("kind") != "directory":
                    continue
                if str(candidate.get("path", "")) == file_parent:
                    dir_chain = _build_parent_chain(candidate, nodes_map)
                    return dir_chain + labels
    return labels


def _node_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = payload.get("nodes", {})
    return nodes if isinstance(nodes, dict) else {}


def _children(nodes: dict[str, dict[str, Any]], node_id: str) -> list[dict[str, Any]]:
    node = nodes.get(node_id, {})
    child_ids = node.get("children", []) if isinstance(node, dict) else []
    if not isinstance(child_ids, list):
        return []
    out: list[dict[str, Any]] = []
    for child_id in child_ids:
        item = nodes.get(str(child_id))
        if isinstance(item, dict):
            out.append(item)
    return out


@tool
def semantic_localize_requirement(path: str, requirement: str, top_k: int = 5) -> str:
    """Localize requirement across repository -> directory -> file -> function using hierarchical semantic index."""
    try:
        req = (requirement or "").strip()
        if not req:
            raise ValueError("requirement cannot be empty")
        payload = _load_or_build(path)
        nodes = _node_map(payload)
        query_tokens = _tokenize(req)
        if not query_tokens:
            query_tokens = req.lower().split()

        repo_nodes = [node for node in nodes.values() if node.get("kind") == "repository"]
        repo_top = _pick_top(repo_nodes, query_tokens, top_k=1)
        repo = repo_top[0] if repo_top else None

        directory_pool: list[dict[str, Any]] = []
        if repo:
            for child in _children(nodes, str(repo["id"])):
                if child.get("kind") == "directory":
                    directory_pool.append(child)
        if not directory_pool:
            directory_pool = [node for node in nodes.values() if node.get("kind") == "directory"]
        top_dirs = _pick_top(directory_pool, query_tokens, top_k=max(2, top_k))
        for item in top_dirs:
            raw = nodes.get(str(item.get("id", "")), {})
            item["parent_chain"] = _build_parent_chain(raw, nodes)

        file_pool: list[dict[str, Any]] = []
        for item in top_dirs:
            for child in _children(nodes, str(item["id"])):
                if child.get("kind") == "file":
                    file_pool.append(child)
        if not file_pool:
            file_pool = [node for node in nodes.values() if node.get("kind") == "file"]
        top_files = _pick_top(file_pool, query_tokens, top_k=max(2, top_k))
        for item in top_files:
            raw = nodes.get(str(item.get("id", "")), {})
            item["parent_chain"] = _build_parent_chain(raw, nodes)

        function_pool: list[dict[str, Any]] = []
        file_ids = {str(item.get("id", "")) for item in top_files}
        for file_id in file_ids:
            for child in _children(nodes, file_id):
                if child.get("kind") == "function":
                    function_pool.append(child)
        if not function_pool:
            function_pool = [node for node in nodes.values() if node.get("kind") == "function"]
        top_functions = _pick_top(function_pool, query_tokens, top_k=max(2, top_k))
        for item in top_functions:
            raw = nodes.get(str(item.get("id", "")), {})
            item["parent_chain"] = _build_parent_chain(raw, nodes)

        return json.dumps(
            {
                "status": "ok",
                "requirement": req,
                "root": payload.get("root", str(Path(path).resolve())),
                "resolved_path": payload.get("resolved_path", str(Path(path).resolve())),
                "scope_type": payload.get("scope_type", "directory"),
                "indexed_targets": payload.get("indexed_targets", []),
                "localized_candidates": {
                    "directories": top_dirs,
                    "files": top_files,
                    "functions": top_functions,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"semantic_localize_requirement error: {exc}"
