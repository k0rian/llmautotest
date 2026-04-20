import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_utils import resolve_code_scope
from tools import semantic_diff_ts as semantic_mod


def _cache_dirs_for_scope(scope: Any) -> list[Path]:
    roots: list[Path] = []
    base = Path(scope.index_root_path).resolve()
    roots.append(base)
    if scope.scope_type == "file":
        roots.extend(base.parents)

    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        cache_dir = root / semantic_mod.INDEX_CACHE_DIRNAME / semantic_mod.INDEX_CACHE_SUBDIR
        key = str(cache_dir).lower()
        if key in seen:
            continue
        seen.add(key)
        if cache_dir.exists() and cache_dir.is_dir():
            out.append(cache_dir)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _include_matches(payload: dict[str, Any], include_glob: str) -> bool:
    actual = str(payload.get("include_glob", "")).strip().lower()
    expected = (include_glob or semantic_mod.DEFAULT_INCLUDE_GLOB).strip().lower()
    return not actual or actual == expected


def _payload_matches_scope(payload: dict[str, Any], scope: Any, include_glob: str) -> tuple[bool, bool]:
    if not _include_matches(payload, include_glob):
        return False, False

    resolved_path = str(payload.get("resolved_path", payload.get("root", ""))).strip()
    scope_type = str(payload.get("scope_type", "directory")).strip() or "directory"
    if resolved_path == scope.resolved_path and scope_type == scope.scope_type:
        return True, False

    if scope.scope_type != "file" or scope_type != "directory":
        return False, False
    target_files = {str(item) for item in payload.get("target_files", []) if str(item).strip()}
    indexed_targets = {str(item) for item in payload.get("indexed_targets", []) if str(item).strip()}
    return scope.resolved_path in (target_files | indexed_targets), True


def _sort_candidates(candidates: list[tuple[Path, dict[str, Any], bool]]) -> list[tuple[Path, dict[str, Any], bool]]:
    def _key(item: tuple[Path, dict[str, Any], bool]) -> tuple[int, str, float]:
        path, payload, covering_directory = item
        stats = payload.get("stats", {}) if isinstance(payload.get("stats", {}), dict) else {}
        mode = str(payload.get("summary_mode") or stats.get("summary_mode", "")).lower()
        stamp = str(payload.get("updated_at") or payload.get("created_at") or "")
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (0 if covering_directory else 1, stamp, mtime + (0.001 if mode == "llm" else 0.0))

    return sorted(candidates, key=_key, reverse=True)


def _candidate_payloads(scope: Any, include_glob: str, *, hierarchical: bool) -> list[tuple[Path, dict[str, Any], bool]]:
    suffix = ".hier.json" if hierarchical else ".json"
    candidates: list[tuple[Path, dict[str, Any], bool]] = []
    for cache_dir in _cache_dirs_for_scope(scope):
        for path in cache_dir.glob(f"*{suffix}"):
            if hierarchical is False and path.name.endswith(".hier.json"):
                continue
            payload = _read_json(path)
            if not payload:
                continue
            matched, covering_directory = _payload_matches_scope(payload, scope, include_glob)
            if matched:
                candidates.append((path, payload, covering_directory))
    return _sort_candidates(candidates)


def _derive_file_index(index: semantic_mod.SemanticIndex, file_path: str) -> semantic_mod.SemanticIndex:
    target = str(Path(file_path).resolve())
    functions = [item for item in index.functions if str(Path(item.file).resolve()) == target]
    language_counts: dict[str, dict[str, int]] = {}
    by_language = Counter(item.language for item in functions)
    for language in sorted(by_language):
        kind_counts = Counter(item.kind for item in functions if item.language == language)
        language_counts[language] = dict(kind_counts)
    return semantic_mod.SemanticIndex(
        root=str(Path(target).parent),
        resolved_path=target,
        target_path=target,
        scope_type="file",
        include_glob=index.include_glob,
        created_at=index.created_at,
        file_count=1,
        function_count=len(functions),
        target_files=[target],
        functions=functions,
        function_name_index=semantic_mod._build_function_name_index(functions),
        idf=index.idf,
        parser_status=index.parser_status,
        parser_errors=index.parser_errors,
        language_symbol_counts=language_counts,
        indexed_files_by_language=dict(by_language),
        summary_mode=index.summary_mode,
        summary_model=index.summary_model,
        summary_errors=index.summary_errors or [],
    )


def load_existing_function_index_any(
    path: str,
    include_glob: str = semantic_mod.DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
) -> tuple[semantic_mod.SemanticIndex | None, str, str]:
    try:
        scope = resolve_code_scope(
            path=path,
            include_glob=include_glob,
            default_include_glob=semantic_mod.DEFAULT_INCLUDE_GLOB,
            max_files=max_files,
            skip_dirs=semantic_mod.SKIP_DIRS,
        )
    except Exception as exc:
        return None, "", f"scope_invalid: {exc}"

    for cache_path, payload, covering_directory in _candidate_payloads(scope, scope.include_glob, hierarchical=False):
        try:
            index = semantic_mod._index_from_payload(payload)
            if covering_directory:
                index = _derive_file_index(index, scope.resolved_path)
            key = semantic_mod._cache_key(
                index.scope_type,
                index.resolved_path,
                index.include_glob,
                index.summary_mode == "llm",
                index.summary_model,
            )
            semantic_mod._INDEX_CACHE[key] = index
            return index, str(cache_path), ""
        except Exception:
            continue

    return (
        None,
        "",
        f"index_missing: function semantic index not found for {Path(path).resolve()}; run agent.py --build-index first",
    )


def load_existing_hierarchical_index_any(
    path: str,
    include_glob: str = semantic_mod.DEFAULT_INCLUDE_GLOB,
    max_files: int = 2000,
) -> tuple[dict[str, Any], str, str]:
    try:
        scope = resolve_code_scope(
            path=path,
            include_glob=include_glob,
            default_include_glob=semantic_mod.DEFAULT_INCLUDE_GLOB,
            max_files=max_files,
            skip_dirs=semantic_mod.SKIP_DIRS,
        )
    except Exception as exc:
        return {}, "", f"scope_invalid: {exc}"

    candidates = _candidate_payloads(scope, scope.include_glob, hierarchical=True)
    if not candidates:
        return (
            {},
            "",
            f"index_missing: hierarchical semantic index not found for {Path(path).resolve()}; run agent.py --build-index first",
        )
    cache_path, payload, _ = candidates[0]
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {}
    artifacts["cache_path"] = str(cache_path)
    payload["artifacts"] = artifacts
    return payload, str(cache_path), ""
