import json
from pathlib import Path
from typing import Any

from build.index_store import load_existing_function_index_any, load_existing_hierarchical_index_any
from planner.types import PlanStep, PlannerRuntimeState
from planner.semantic_state_machine import SemanticStateMachine
from tools.semantic_diff_ts import DEFAULT_INCLUDE_GLOB


def _parse_json_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ensure_semantic_context(runtime_state: PlannerRuntimeState) -> dict[str, Any]:
    if runtime_state.semantic_context is None:
        runtime_state.semantic_context = {}
    return runtime_state.semantic_context


class SemanticAgent:
    def __init__(self) -> None:
        self.state_machine = SemanticStateMachine()

    def execute_step(
        self,
        *,
        mode: str,
        objective: str,
        step: PlanStep,
        runtime_state: PlannerRuntimeState,
        workspace_path: str,
        target_path: str,
    ) -> tuple[str, dict[str, Any]]:
        context = _ensure_semantic_context(runtime_state)
        requirement = (step.objective or objective or "").strip()
        if not requirement:
            requirement = "semantic requirement"

        if mode == "semantic_index":
            cache, cache_path, cache_error = load_existing_hierarchical_index_any(
                target_path,
                include_glob=DEFAULT_INCLUDE_GLOB,
                max_files=2000,
            )
            function_index, function_cache_path, function_error = load_existing_function_index_any(
                path=target_path,
                include_glob=DEFAULT_INCLUDE_GLOB,
                max_files=2000,
            )
            index_success = bool(cache) and function_index is not None
            missing_error = ""
            if not cache:
                missing_error = cache_error
            elif function_index is None:
                missing_error = function_error
            context.update(
                {
                    "path": target_path,
                    "workspace_path": workspace_path,
                    "requirement": requirement,
                    "index_result": cache,
                    "index_stats": cache.get("stats", {}),
                    "index_error": missing_error,
                }
            )
            summary = (
                "semantic_index workflow summary\n"
                f"- root: {cache.get('root', '') or target_path}\n"
                f"- cache_path: {cache_path}\n"
                f"- function_cache_path: {function_cache_path}\n"
                f"- stats: {json.dumps(cache.get('stats', {}), ensure_ascii=False)}\n"
                f"- summary_mode: {cache.get('stats', {}).get('summary_mode', '') if isinstance(cache.get('stats', {}), dict) else ''}\n"
                f"- index_available: {index_success}\n"
                f"- error: {missing_error}"
            )
            return summary, {
                "stage": "semantic_index",
                "index_success": index_success,
                "root": cache.get("root", target_path),
                "cache_path": cache_path,
                "function_cache_path": function_cache_path,
                "stats": cache.get("stats", {}),
                "requirement": requirement,
                "error_stage": "" if index_success else "index_missing",
                "error": missing_error,
            }

        if mode == "semantic_localize":
            localized = self.state_machine.localize(
                path=target_path,
                requirement=requirement,
                top_k=5,
            )
            detect = self.state_machine.detect(requirement=requirement, localized=localized, retrieved=[])
            context.update(
                {
                    "path": target_path,
                    "requirement": requirement,
                    "localized": localized,
                    "preliminary_finding": detect,
                }
            )
            candidates = localized.get("localized_candidates", {}) if isinstance(localized, dict) else {}
            summary = (
                "semantic_localize workflow summary\n"
                f"- requirement: {requirement[:120]}\n"
                f"- directories: {len(candidates.get('directories', [])) if isinstance(candidates, dict) else 0}\n"
                f"- files: {len(candidates.get('files', [])) if isinstance(candidates, dict) else 0}\n"
                f"- functions: {len(candidates.get('functions', [])) if isinstance(candidates, dict) else 0}\n"
                f"- preliminary_finding: {detect.get('preliminary_finding', 'unknown')}"
            )
            return summary, {
                "stage": "semantic_localize",
                "requirement": requirement,
                "localized_candidates": candidates,
                "preliminary_finding": detect.get("preliminary_finding", "unknown"),
                "reason": detect.get("reason", ""),
                "gaps": detect.get("gaps", []),
                "matched_symbols": detect.get("matched_symbols", []),
                "decision": detect.get("preliminary_finding", "unknown"),
                "confidence": float(detect.get("confidence", 0.45) or 0.45),
                "retrieved_contexts": [],
                "evidence": [],
                "missing_requirements": [],
                "covered_requirements": [],
                "partial_requirements": [requirement] if detect.get("preliminary_finding") == "partial" else [],
            }

        if mode == "semantic_retrieve":
            localized = context.get("localized", {}) if isinstance(context.get("localized"), dict) else {}
            if not localized:
                localized = self.state_machine.localize(
                    path=target_path,
                    requirement=requirement,
                    top_k=5,
                )
                context["localized"] = localized

            detect_1 = self.state_machine.detect(requirement=requirement, localized=localized, retrieved=[])
            retrieved = self.state_machine.retrieve(
                path=target_path,
                localized=localized,
                top_k=3,
            )
            if not retrieved:
                fallback_localized = self.state_machine.localize(
                    path=target_path,
                    requirement=requirement,
                    top_k=12,
                )
                localized = fallback_localized or localized
                context["localized"] = localized
                retrieved = self.state_machine.retrieve(
                    path=target_path,
                    localized=localized,
                    top_k=8,
                )

            evidence = self.state_machine.build_validation_evidence(localized=localized, retrieved=retrieved)
            detect_2 = self.state_machine.detect(requirement=requirement, localized=localized, retrieved=retrieved)
            context.update(
                {
                    "path": target_path,
                    "requirement": requirement,
                    "retrieved": retrieved,
                    "retrieval_evidence": evidence,
                    "preliminary_finding": detect_2,
                }
            )
            summary = (
                "semantic_retrieve workflow summary\n"
                f"- detect_before: {detect_1.get('preliminary_finding', 'unknown')}\n"
                f"- retrieved_context_blocks: {len(retrieved)}\n"
                f"- retrieval_evidence: {len(evidence)}\n"
                f"- detect_after: {detect_2.get('preliminary_finding', 'unknown')}"
            )
            prelim = str(detect_2.get("preliminary_finding", "unknown"))
            return summary, {
                "stage": "semantic_retrieve",
                "requirement": requirement,
                "localized_candidates": localized.get("localized_candidates", {}),
                "retrieved_contexts": retrieved,
                "decision": prelim,
                "preliminary_finding": prelim,
                "reason": detect_2.get("reason", ""),
                "gaps": detect_2.get("gaps", []),
                "matched_symbols": detect_2.get("matched_symbols", []),
                "confidence": float(detect_2.get("confidence", 0.45) or 0.45),
                "evidence": evidence,
                "missing_requirements": [requirement] if prelim in {"missing", "violated"} else [],
                "covered_requirements": [requirement] if prelim == "covered" else [],
                "partial_requirements": [requirement] if prelim == "partial" else [],
            }

        if mode == "semantic_validate":
            localized = context.get("localized", {}) if isinstance(context.get("localized"), dict) else {}
            retrieved = context.get("retrieved", []) if isinstance(context.get("retrieved"), list) else []
            prelim_obj = context.get("preliminary_finding", {}) if isinstance(context.get("preliminary_finding"), dict) else {}
            preliminary = str(prelim_obj.get("preliminary_finding", "unknown"))
            if not localized:
                localized = self.state_machine.localize(
                    path=target_path,
                    requirement=requirement,
                    top_k=5,
                )
            if not retrieved:
                retrieved = self.state_machine.retrieve(
                    path=target_path,
                    localized=localized,
                    top_k=3,
                )
            evidence = self.state_machine.build_validation_evidence(localized=localized, retrieved=retrieved)

            validated = self.state_machine.validate(
                requirement=requirement,
                evidence=evidence,
                preliminary_finding=preliminary,
            )
            decision = str(validated.get("decision", preliminary))
            if not evidence and decision not in {"unknown", "evidence_insufficient"}:
                decision = "unknown"
            confidence = float(validated.get("confidence", 0.5) or 0.5)
            context.update(
                {
                    "path": target_path,
                    "requirement": requirement,
                    "validated": validated,
                    "final_decision": decision,
                    "final_confidence": confidence,
                }
            )

            summary = (
                "semantic_validate workflow summary\n"
                f"- preliminary: {preliminary}\n"
                f"- decision: {decision}\n"
                f"- confidence: {confidence}\n"
                f"- evidence_count: {len(evidence)}"
            )
            return summary, {
                "stage": "semantic_validate",
                "requirement": requirement,
                "localized_candidates": localized.get("localized_candidates", {}),
                "retrieved_contexts": retrieved,
                "decision": decision,
                "confidence": confidence,
                "evidence": evidence,
                "validation": validated,
                "missing_requirements": [requirement] if decision in {"missing", "violated"} else [],
                "covered_requirements": [requirement] if decision == "covered" else [],
                "partial_requirements": [requirement] if decision == "partial" else [],
            }

        raise ValueError(f"unsupported semantic mode: {mode}")
