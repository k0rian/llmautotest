import subprocess
import json
import os
import re
import fnmatch
import shutil
from pathlib import Path
from typing import Any, Dict, List
from langchain.tools import tool

DEFAULT_AUDIT_RULES = [
    {
        "id": "hardcoded-secret",
        "severity": "HIGH",
        "message": "Possible hardcoded credential",
        "pattern": r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
    },
    {
        "id": "shell-true",
        "severity": "HIGH",
        "message": "subprocess with shell=True may introduce command injection risk",
        "pattern": r"subprocess\.(run|Popen|call|check_output)\s*\([^)]*shell\s*=\s*True",
    },
    {
        "id": "eval-exec",
        "severity": "HIGH",
        "message": "Use of eval/exec is risky",
        "pattern": r"\b(eval|exec)\s*\(",
    },
    {
        "id": "pickle-loads",
        "severity": "MEDIUM",
        "message": "pickle.loads may execute untrusted code",
        "pattern": r"\bpickle\.loads\s*\(",
    },
    {
        "id": "yaml-load",
        "severity": "MEDIUM",
        "message": "yaml.load without safe loader is risky",
        "pattern": r"\byaml\.load\s*\(",
    },
    {
        "id": "tls-verify-false",
        "severity": "MEDIUM",
        "message": "TLS verification disabled",
        "pattern": r"\bverify\s*=\s*False\b",
    },
]
MAX_FILE_BYTES = 1024 * 1024
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode"}


def _normalize_severity(value: str) -> str:
    upper = str(value or "").upper()
    if upper in {"ERROR", "CRITICAL"}:
        return "HIGH"
    if upper in {"WARNING", "WARN"}:
        return "MEDIUM"
    if upper in {"INFO", "LOW"}:
        return "LOW"
    return "MEDIUM"


def _iter_files(root: str, include_glob: str):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not fnmatch.fnmatch(name, include_glob):
                continue
            file_path = os.path.join(current_root, name)
            try:
                if os.path.getsize(file_path) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield file_path


def _run_semgrep(path: str) -> Dict[str, Any]:
    if shutil.which("semgrep") is None:
        return {"status": "unavailable", "findings": [], "error": "semgrep not installed"}
    try:
        result = subprocess.run(
            [
                "semgrep",
                "--config",
                "auto",
                "--json",
                path
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        if result.returncode not in [0, 1]:
            return {"status": "error", "findings": [], "error": result.stderr.strip()}
        payload = json.loads(result.stdout or "{}")
        findings = []
        for issue in payload.get("results", []):
            findings.append({
                "source": "semgrep",
                "rule_id": issue.get("check_id", "semgrep-rule"),
                "file": issue.get("path"),
                "line": issue.get("start", {}).get("line", 1),
                "message": issue.get("extra", {}).get("message", ""),
                "severity": _normalize_severity(issue.get("extra", {}).get("severity", "MEDIUM")),
            })
        return {"status": "ok", "findings": findings, "error": ""}
    except Exception as e:
        return {"status": "error", "findings": [], "error": str(e)}


def _run_pattern_audit(path: str, include_glob: str, max_results: int) -> List[Dict[str, Any]]:
    compiled_rules = []
    for rule in DEFAULT_AUDIT_RULES:
        compiled_rules.append((rule, re.compile(rule["pattern"])))
    findings: List[Dict[str, Any]] = []
    for file_path in _iter_files(path, include_glob):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    for rule, pattern in compiled_rules:
                        if pattern.search(line):
                            findings.append({
                                "source": "pattern",
                                "rule_id": rule["id"],
                                "file": str(Path(file_path).resolve()),
                                "line": line_no,
                                "message": rule["message"],
                                "severity": rule["severity"],
                                "snippet": line.strip(),
                            })
                            if len(findings) >= max_results:
                                return findings
        except OSError:
            continue
    return findings

@tool
def semgrep_scan(path: str) -> str:
    """
    Run static code analysis on a project directory using semgrep.
    Input should be a path to the source code directory.
    """

    payload = _run_semgrep(path)
    if payload["status"] == "ok":
        return json.dumps(payload["findings"], ensure_ascii=False, indent=2)
    return f"Semgrep execution failed: {payload['error']}"


@tool
def list_audit_rules() -> str:
    """
    List built-in pattern audit rules.
    """
    return json.dumps(DEFAULT_AUDIT_RULES, ensure_ascii=False, indent=2)


@tool
def audit_codebase(path: str, include_glob: str = "*.py", max_results: int = 200, use_semgrep: bool = True) -> str:
    """
    Run combined security audit with semgrep and built-in pattern rules.
    """
    try:
        if not os.path.isdir(path):
            return f"Static analysis error: invalid directory path '{path}'"
        limit = max(1, min(int(max_results), 1000))
        semgrep_result = {"status": "skipped", "findings": [], "error": ""}
        if use_semgrep:
            semgrep_result = _run_semgrep(path)
        pattern_findings = _run_pattern_audit(path, include_glob=include_glob, max_results=limit)
        merged = []
        for item in semgrep_result["findings"]:
            merged.append(item)
        for item in pattern_findings:
            merged.append(item)
        merged = merged[:limit]
        severity_count = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for finding in merged:
            sev = _normalize_severity(finding.get("severity", "MEDIUM"))
            severity_count[sev] += 1
        result = {
            "path": str(Path(path).resolve()),
            "include_glob": include_glob,
            "max_results": limit,
            "semgrep": {
                "status": semgrep_result["status"],
                "error": semgrep_result["error"],
                "count": len(semgrep_result["findings"]),
            },
            "pattern_rules": {
                "count": len(pattern_findings),
                "rule_count": len(DEFAULT_AUDIT_RULES),
            },
            "summary": {
                "total_findings": len(merged),
                "severity": severity_count,
            },
            "findings": merged,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Static analysis error: {str(e)}"

