import subprocess
import json
from langchain.tools import tool

@tool
def semgrep_scan(path: str) -> str:
    """
    Run static code analysis on a project directory using semgrep.
    Input should be a path to the source code directory.
    """

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
            timeout=120
        )

        if result.returncode not in [0, 1]:
            return f"Semgrep execution failed: {result.stderr}"

        data = json.loads(result.stdout)

        findings = []

        for issue in data.get("results", []):
            findings.append({
                "file": issue["path"],
                "line": issue["start"]["line"],
                "message": issue["extra"]["message"],
                "severity": issue["extra"]["severity"]
            })

        return json.dumps(findings, indent=2)

    except Exception as e:
        return f"Static analysis error: {str(e)}"

