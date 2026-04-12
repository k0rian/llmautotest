import json
import os
import subprocess
from pathlib import Path

from langchain.tools import tool

from permission.command_permission import request_command_permission


@tool
def run_shell_command(command: str, cwd: str = "", timeout_sec: int = 120) -> str:
    """
    Execute a CLI command after interactive permission checks when needed.
    """
    try:
        allowed, detail = request_command_permission(command)
    except Exception as exc:
        return f"Shell tool error: {exc}"

    if not allowed:
        return f"Shell command denied: {detail.reason}"

    try:
        resolved_cwd = str(Path(cwd).resolve()) if cwd and cwd.strip() else os.getcwd()
        timeout = max(1, min(int(timeout_sec), 600))
        result = subprocess.run(
            detail.argv,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        payload = {
            "command": " ".join(detail.argv),
            "cwd": resolved_cwd,
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[:12000],
            "stderr": (result.stderr or "")[:6000],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except subprocess.TimeoutExpired:
        return f"Shell command timeout after {timeout_sec} seconds"
    except Exception as exc:
        return f"Shell execution error: {exc}"
