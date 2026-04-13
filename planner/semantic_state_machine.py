import json
from dataclasses import dataclass
from typing import Any

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
                    "reason": str(item.get("why", "localized candidate")),
                }
            )
    return out


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

    def localize(self, path: str, requirement: str, top_k: int = 5) -> dict[str, Any]:
        return _parse_json_text(
            semantic_localize_requirement.func(path=path, requirement=requirement, top_k=max(1, int(top_k)))
        )

    def detect(self, requirement: str, localized: dict[str, Any], retrieved: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        retrieved = retrieved or []
        candidates = localized.get("localized_candidates", {}) if isinstance(localized, dict) else {}
        function_hits = len(candidates.get("functions", [])) if isinstance(candidates, dict) else 0
        evidence_count = len(retrieved)

        if function_hits <= 0 and evidence_count <= 0:
            return {"preliminary_finding": "unknown", "reason": "no localized candidates"}
        if function_hits > 0 and evidence_count <= 0:
            return {"preliminary_finding": "partial", "reason": "localized candidates exist but no retrieved graph context"}

        high_score_hits = 0
        for item in candidates.get("functions", []) if isinstance(candidates, dict) else []:
            if not isinstance(item, dict):
                continue
            if float(item.get("score", 0.0) or 0.0) >= 0.2:
                high_score_hits += 1

        if high_score_hits > 0 and evidence_count >= 2:
            return {"preliminary_finding": "covered", "reason": "high-score candidates with graph evidence"}
        if evidence_count >= 1:
            return {"preliminary_finding": "partial", "reason": "some graph evidence found"}
        return {"preliminary_finding": "unknown", "reason": "insufficient evidence"}

    def retrieve(self, path: str, localized: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        del top_k
        candidates = localized.get("localized_candidates", {}) if isinstance(localized, dict) else {}
        functions = candidates.get("functions", []) if isinstance(candidates, dict) else []
        out: list[dict[str, Any]] = []

        for item in functions[:3] if isinstance(functions, list) else []:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("path", "")).strip()
            fn_name = str(item.get("name", "")).strip()
            if not file_path or not fn_name:
                continue

            definition = _parse_json_text(query_symbol_definition.func(name=fn_name, path=path))
            callees = _parse_json_text(query_callee_functions.func(file_path=file_path, function_name=fn_name))
            callers = _parse_json_text(query_caller_functions.func(file_path=file_path, function_name=fn_name))
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

    def run_full(self, path: str, requirement: str) -> SemanticStateResult:
        localized = self.localize(path=path, requirement=requirement, top_k=5)
        detect_1 = self.detect(requirement=requirement, localized=localized, retrieved=[])
        retrieved = self.retrieve(path=path, localized=localized, top_k=3)

        retrieval_evidence: list[dict[str, Any]] = []
        for block in retrieved:
            if not isinstance(block, dict):
                continue
            for key in ("definition", "callees", "callers"):
                value = block.get(key, [])
                if isinstance(value, list):
                    retrieval_evidence.extend([item for item in value if isinstance(item, dict)])

        detect_2 = self.detect(requirement=requirement, localized=localized, retrieved=retrieval_evidence)
        final_finding = str(detect_2.get("preliminary_finding", detect_1.get("preliminary_finding", "unknown")))

        evidence = _collect_evidence_from_localized(localized) + retrieval_evidence
        validated = self.validate(requirement=requirement, evidence=evidence, preliminary_finding=final_finding)
        decision = str(validated.get("decision", final_finding))
        confidence = float(validated.get("confidence", 0.5) or 0.5)

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
