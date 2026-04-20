import json
from dataclasses import dataclass
from typing import Any

from tools import semantic_diff_ts as semantic_mod
from tools.code_localization import semantic_localize_requirement
from tools.code_retrieval import (
    query_callee_functions,
    query_caller_functions,
    query_symbol_definition,
    validate_semantic_finding,
)


def _parse_json_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_line_range(value: Any) -> list[int]:
    if isinstance(value, list) and len(value) == 2:
        try:
            return [int(value[0] or 0), int(value[1] or 0)]
        except Exception:
            return [0, 0]
    return [0, 0]


def _token_set(text: str) -> set[str]:
    return set(semantic_mod._tokenize((text or "").lower()))


def _overlap_tokens(left: str, right: str) -> list[str]:
    overlap = sorted(_token_set(left) & _token_set(right))
    return overlap[:12]


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, int, int]:
    file_path = str(item.get("file_path", "")).strip()
    symbol = str(item.get("symbol_name", "")).strip()
    line_range = _normalize_line_range(item.get("line_range", [0, 0]))
    return (file_path, symbol, int(line_range[0]), int(line_range[1]))


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _collect_evidence_from_localized(localized: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    candidates = localized.get("localized_candidates", {}) if isinstance(localized, dict) else {}
    if not isinstance(candidates, dict):
        return out
    for bucket in ("functions", "files", "directories"):
        entries = candidates.get(bucket, [])
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "file_path": str(item.get("path", "")),
                    "line_range": [0, 0],
                    "symbol_name": str(item.get("name", "")),
                    "signature": str(item.get("signature", "")),
                    "source_summary": str(item.get("summary", ""))[:260],
                    "reason": str(item.get("why", "localized candidate")),
                }
            )
    return out


def _collect_evidence_from_retrieved(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in retrieved:
        if not isinstance(block, dict):
            continue
        for key in ("definition", "callees", "callers"):
            value = block.get(key, [])
            if not isinstance(value, list):
                continue
            out.extend([item for item in value if isinstance(item, dict)])
    return out


def build_validation_evidence(localized: dict[str, Any], retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_evidence(_collect_evidence_from_localized(localized) + _collect_evidence_from_retrieved(retrieved))


def _build_analysis_bundle(requirement: str, localized: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = localized.get("localized_candidates", {}) if isinstance(localized, dict) else {}
    top_functions = candidates.get("functions", []) if isinstance(candidates, dict) else []
    evidence = build_validation_evidence(localized=localized, retrieved=retrieved)
    return {
        "requirement": requirement,
        "localized_candidates": candidates,
        "top_functions": top_functions[:5] if isinstance(top_functions, list) else [],
        "retrieved_contexts": retrieved,
        "evidence": evidence,
    }


def _detect_requirement_coverage(bundle: dict[str, Any]) -> dict[str, Any]:
    requirement = str(bundle.get("requirement", "")).strip()
    evidence = bundle.get("evidence", []) if isinstance(bundle.get("evidence", []), list) else []
    top_functions = bundle.get("top_functions", []) if isinstance(bundle.get("top_functions", []), list) else []

    if not requirement:
        return {
            "preliminary_finding": "unknown",
            "reason": "empty requirement",
            "matched_symbols": [],
            "gaps": ["missing requirement text"],
            "confidence": 0.1,
        }

    req_tokens = _token_set(requirement)
    if not req_tokens:
        return {
            "preliminary_finding": "unknown",
            "reason": "requirement tokens unavailable",
            "matched_symbols": [],
            "gaps": ["requirement cannot be tokenized"],
            "confidence": 0.2,
        }

    matched_symbols: list[str] = []
    strong_hits = 0
    weak_hits = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        haystack = " ".join(
            [
                str(item.get("symbol_name", "")),
                str(item.get("signature", "")),
                str(item.get("reason", "")),
                str(item.get("source_summary", "")),
            ]
        )
        overlap = set(_overlap_tokens(requirement, haystack))
        if not overlap:
            continue
        symbol = str(item.get("symbol_name", "")).strip()
        if symbol:
            matched_symbols.append(symbol)
        overlap_ratio = len(overlap) / max(1, len(req_tokens))
        if overlap_ratio >= 0.3:
            strong_hits += 1
        else:
            weak_hits += 1

    localized_fn_count = len([item for item in top_functions if isinstance(item, dict)])
    gaps: list[str] = []

    if strong_hits >= 2:
        decision = "covered"
        confidence = min(0.9, 0.55 + strong_hits * 0.08)
        reason = "requirement tokens strongly align with multiple symbol evidences"
    elif strong_hits == 1 or (weak_hits >= 2 and localized_fn_count > 0):
        decision = "partial"
        confidence = 0.58 if strong_hits == 1 else 0.5
        reason = "some requirement tokens align, but coverage chain is incomplete"
        gaps.append("missing full supporting call/context chain")
    elif localized_fn_count > 0 and strong_hits == 0 and weak_hits == 0:
        decision = "missing"
        confidence = 0.62
        reason = "localized candidate area found but no requirement token evidence in retrieved context"
        gaps.append("no semantic evidence tied to requirement tokens")
    else:
        decision = "unknown"
        confidence = 0.35
        reason = "insufficient localization/retrieval evidence for requirement"
        gaps.append("insufficient evidence")

    if decision == "covered" and localized_fn_count <= 0:
        decision = "unknown"
        confidence = 0.3
        reason = "covered hypothesis invalidated due to empty localization"
        gaps.append("empty localized function candidates")

    return {
        "preliminary_finding": decision,
        "reason": reason,
        "matched_symbols": sorted(set(matched_symbols))[:10],
        "gaps": gaps,
        "confidence": round(float(confidence), 3),
    }


@dataclass
class SemanticStateResult:
    requirement: str
    localized_candidates: dict[str, Any]
    retrieved_contexts: list[dict[str, Any]]
    decision: str
    confidence: float
    evidence: list[dict[str, Any]]
    missing_requirements: list[str]
    covered_requirements: list[str]
    partial_requirements: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "localized_candidates": self.localized_candidates,
            "retrieved_contexts": self.retrieved_contexts,
            "decision": self.decision,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "missing_requirements": self.missing_requirements,
            "covered_requirements": self.covered_requirements,
            "partial_requirements": self.partial_requirements,
        }


class SemanticStateMachine:
    """Semantic workflow state machine: Localize -> Detect -> Retrieve -> Re-Detect -> Validate."""

    def localize(
        self,
        path: str,
        requirement: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return _parse_json_text(
            semantic_localize_requirement.func(
                path=path,
                requirement=requirement,
                top_k=max(1, int(top_k)),
            )
        )

    def detect(self, requirement: str, localized: dict[str, Any], retrieved: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        retrieved = retrieved or []
        bundle = _build_analysis_bundle(requirement=requirement, localized=localized, retrieved=retrieved)
        return _detect_requirement_coverage(bundle)

    def retrieve(
        self,
        path: str,
        localized: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        candidates = localized.get("localized_candidates", {}) if isinstance(localized, dict) else {}
        functions = candidates.get("functions", []) if isinstance(candidates, dict) else []
        out: list[dict[str, Any]] = []

        for item in functions[: max(1, int(top_k))] if isinstance(functions, list) else []:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("path", "")).strip()
            fn_name = str(item.get("name", "")).strip()
            if not file_path or not fn_name:
                continue

            definition = _parse_json_text(
                query_symbol_definition.func(
                    name=fn_name,
                    path=path,
                )
            )
            callees = _parse_json_text(
                query_callee_functions.func(
                    file_path=file_path,
                    function_name=fn_name,
                    scope_path=path,
                )
            )
            callers = _parse_json_text(
                query_caller_functions.func(
                    file_path=file_path,
                    function_name=fn_name,
                    scope_path=path,
                )
            )
            out.append(
                {
                    "target": {"file_path": file_path, "symbol_name": fn_name},
                    "definition": definition.get("definitions", []),
                    "callees": callees.get("callees", []),
                    "callers": callers.get("callers", []),
                }
            )
        return out

    def validate(self, requirement: str, evidence: list[dict[str, Any]], preliminary_finding: str) -> dict[str, Any]:
        return _parse_json_text(
            validate_semantic_finding.func(
                requirement=requirement,
                evidence_bundle={"evidence": evidence},
                preliminary_finding=preliminary_finding,
            )
        )

    def build_validation_evidence(self, localized: dict[str, Any], retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return build_validation_evidence(localized=localized, retrieved=retrieved)

    def run_full(
        self,
        path: str,
        requirement: str,
    ) -> SemanticStateResult:
        localized = self.localize(
            path=path,
            requirement=requirement,
            top_k=5,
        )
        detect_1 = self.detect(requirement=requirement, localized=localized, retrieved=[])
        retrieved = self.retrieve(
            path=path,
            localized=localized,
            top_k=3,
        )
        detect_2 = self.detect(requirement=requirement, localized=localized, retrieved=retrieved)
        final_finding = str(detect_2.get("preliminary_finding", detect_1.get("preliminary_finding", "unknown")))

        evidence = self.build_validation_evidence(localized=localized, retrieved=retrieved)
        validated = self.validate(requirement=requirement, evidence=evidence, preliminary_finding=final_finding)
        decision = str(validated.get("decision", final_finding))
        confidence = float(validated.get("confidence", detect_2.get("confidence", 0.5)) or 0.5)

        missing = [requirement] if decision in {"missing", "violated"} else []
        covered = [requirement] if decision == "covered" else []
        partial = [requirement] if decision == "partial" else []

        return SemanticStateResult(
            requirement=requirement,
            localized_candidates=localized.get("localized_candidates", {}),
            retrieved_contexts=retrieved,
            decision=decision,
            confidence=confidence,
            evidence=evidence,
            missing_requirements=missing,
            covered_requirements=covered,
            partial_requirements=partial,
        )
