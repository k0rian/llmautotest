import json
import re
from pathlib import Path
from typing import Any

from langchain.tools import tool

from tools import semantic_diff_ts as semantic_mod

_CALL_EXCLUDE = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
    "new",
    "delete",
    "await",
}


def _guess_workspace_from_file(file_path: str) -> Path:
    path = Path(file_path).resolve()
    for candidate in [path.parent, *path.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return path.parent


def _load_function_records(
    root: Path | str,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> list[semantic_mod.FunctionRecord]:
    include_glob = semantic_mod.DEFAULT_INCLUDE_GLOB
    index, _ = semantic_mod._get_or_create_index(
        path=str(root),
        include_glob=include_glob,
        max_files=2000,
        rebuild=False,
        use_llm_summary=bool(use_llm_summary),
        model_name=summary_model_name,
    )
    return index.functions


def _normalize_path(path: str) -> str:
    return str(Path(path).resolve())


def _match_records(
    records: list[semantic_mod.FunctionRecord], file_path: str | None = None, function_name: str | None = None
) -> list[semantic_mod.FunctionRecord]:
    out = records
    if file_path:
        target = _normalize_path(file_path)
        out = [item for item in out if _normalize_path(item.file) == target]
    if function_name:
        name = function_name.strip()
        out = [item for item in out if item.name == name]
    return out


def _extract_calls(source: str) -> list[str]:
    hits = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", source or "")
    result: list[str] = []
    for token in hits:
        lowered = token.lower()
        if lowered in _CALL_EXCLUDE:
            continue
        result.append(token)
    return result


def _evidence(
    *,
    file_path: str,
    start: int,
    end: int,
    symbol_name: str,
    signature: str,
    source_summary: str,
    reason: str,
    kind: str = "function",
    summary: str = "",
    source_excerpt: str = "",
    call_site: dict[str, Any] | None = None,
    why_relevant: str = "",
) -> dict[str, Any]:
    return {
        "file_path": file_path,
        "line_range": [start, end],
        "symbol_name": symbol_name,
        "kind": kind,
        "signature": signature,
        "summary": (summary or source_summary)[:260],
        "source_summary": source_summary[:260],
        "source_excerpt": (source_excerpt or source_summary)[:260],
        "call_site": call_site or {},
        "why_relevant": why_relevant or reason,
        "reason": reason,
    }


def _parse_evidence_bundle(bundle: Any) -> list[dict[str, Any]]:
    if isinstance(bundle, list):
        return [item for item in bundle if isinstance(item, dict)]
    if isinstance(bundle, dict):
        value = bundle.get("evidence", [])
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []
    text = str(bundle or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return _parse_evidence_bundle(parsed)


@tool
def query_symbol_definition(
    name: str,
    path: str,
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """Find symbol definition candidates by symbol name and return path/line/signature/source summary."""
    try:
        symbol = (name or "").strip()
        if not symbol:
            raise ValueError("name cannot be empty")
        root = Path(path).resolve()
        records = _load_function_records(root, use_llm_summary=use_llm_summary, summary_model_name=summary_model_name)
        matched = [item for item in records if item.name == symbol]
        if not matched:
            matched = [item for item in records if symbol.lower() in item.name.lower()]

        evidence = [
            _evidence(
                file_path=item.file,
                start=item.start_line,
                end=item.end_line,
                symbol_name=item.name,
                signature=item.signature,
                source_summary=item.source,
                reason="symbol definition candidate",
                kind=item.kind,
                summary=f"definition of {item.name}",
                source_excerpt=item.source,
                why_relevant=f"symbol name matched query '{symbol}'",
            )
            for item in matched[:20]
        ]
        return json.dumps(
            {
                "status": "ok",
                "query": symbol,
                "root": str(root),
                "count": len(evidence),
                "definitions": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"query_symbol_definition error: {exc}"


@tool
def query_callee_functions(
    file_path: str,
    function_name: str,
    scope_path: str = "",
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """Find direct callee functions from a given function."""
    try:
        root = Path(scope_path).resolve() if scope_path.strip() else _guess_workspace_from_file(file_path)
        records = _load_function_records(root, use_llm_summary=use_llm_summary, summary_model_name=summary_model_name)
        matched = _match_records(records, file_path=file_path, function_name=function_name)
        if not matched:
            raise ValueError("function not found in semantic index")
        owner = matched[0]
        callee_names = [name for name in _extract_calls(owner.source) if name != owner.name]
        callee_set = set(callee_names)
        callee_records = [item for item in records if item.name in callee_set]
        evidence = [
            _evidence(
                file_path=item.file,
                start=item.start_line,
                end=item.end_line,
                symbol_name=item.name,
                signature=item.signature,
                source_summary=item.source,
                reason=f"called by {owner.name}",
                kind=item.kind,
                summary=f"callee of {owner.name}",
                source_excerpt=item.source,
                call_site={
                    "caller_file_path": owner.file,
                    "caller_symbol_name": owner.name,
                    "caller_line_range": [owner.start_line, owner.end_line],
                },
                why_relevant=f"{owner.name} contains call to {item.name}",
            )
            for item in callee_records[:30]
        ]
        return json.dumps(
            {
                "status": "ok",
                "root": str(root),
                "resolved_path": str(Path(scope_path).resolve()) if scope_path.strip() else str(Path(root).resolve()),
                "owner": {
                    "file_path": owner.file,
                    "line_range": [owner.start_line, owner.end_line],
                    "symbol_name": owner.name,
                    "signature": owner.signature,
                },
                "callees": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"query_callee_functions error: {exc}"


@tool
def query_caller_functions(
    file_path: str,
    function_name: str,
    scope_path: str = "",
    use_llm_summary: bool = False,
    summary_model_name: str = "",
) -> str:
    """Find direct caller functions for a given function."""
    try:
        target_name = (function_name or "").strip()
        if not target_name:
            raise ValueError("function_name cannot be empty")
        root = Path(scope_path).resolve() if scope_path.strip() else _guess_workspace_from_file(file_path)
        records = _load_function_records(root, use_llm_summary=use_llm_summary, summary_model_name=summary_model_name)
        callers: list[semantic_mod.FunctionRecord] = []
        for item in records:
            if item.name == target_name:
                continue
            calls = _extract_calls(item.source)
            if target_name in calls:
                callers.append(item)

        evidence = [
            _evidence(
                file_path=item.file,
                start=item.start_line,
                end=item.end_line,
                symbol_name=item.name,
                signature=item.signature,
                source_summary=item.source,
                reason=f"calls {target_name}",
                kind=item.kind,
                summary=f"caller of {target_name}",
                source_excerpt=item.source,
                call_site={
                    "callee_symbol_name": target_name,
                },
                why_relevant=f"{item.name} contains call to {target_name}",
            )
            for item in callers[:30]
        ]
        return json.dumps(
            {
                "status": "ok",
                "root": str(root),
                "resolved_path": str(Path(scope_path).resolve()) if scope_path.strip() else str(Path(root).resolve()),
                "target": target_name,
                "callers": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"query_caller_functions error: {exc}"


@tool
def validate_semantic_finding(requirement: str, evidence_bundle: Any, preliminary_finding: str) -> str:
    """Validate preliminary semantic finding and classify false-positive risk."""
    try:
        req = (requirement or "").strip()
        finding = (preliminary_finding or "").strip().lower()
        evidence = _parse_evidence_bundle(evidence_bundle)

        if not evidence:
            decision = "unknown"
            confidence = 0.2
            critique = "证据不足"
        else:
            reasons = " ".join(str(item.get("reason", "")) for item in evidence).lower()
            symbols = [str(item.get("symbol_name", "")).lower() for item in evidence]
            matched_keywords = sum(1 for token in semantic_mod._tokenize(req) if token in reasons or token in " ".join(symbols))

            if "violat" in finding or "missing" in finding:
                if matched_keywords <= 1:
                    decision = "evidence_insufficient"
                    confidence = 0.4
                    critique = "上下文未完整展开或命名误导，当前缺失结论风险较高"
                else:
                    decision = "missing"
                    confidence = 0.72
                    critique = "证据支持缺失/违反判断"
            elif "partial" in finding:
                decision = "partial"
                confidence = 0.68
                critique = "存在实现片段但覆盖不完整"
            elif "cover" in finding or "ok" in finding:
                decision = "covered"
                confidence = 0.75
                critique = "检索证据与需求关键词匹配"
            else:
                decision = "unknown"
                confidence = 0.5
                critique = "预结论语义不明确，需补充上下文"

        return json.dumps(
            {
                "status": "ok",
                "requirement": req,
                "preliminary_finding": preliminary_finding,
                "decision": decision,
                "confidence": round(float(confidence), 3),
                "critique": critique,
                "evidence_count": len(evidence),
                "classification": {
                    "true_missing": decision == "missing",
                    "partial_coverage": decision == "partial",
                    "naming_misleading": decision == "evidence_insufficient",
                    "insufficient_evidence": decision in {"unknown", "evidence_insufficient"},
                    "incomplete_context": decision in {"unknown", "evidence_insufficient"},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"validate_semantic_finding error: {exc}"
