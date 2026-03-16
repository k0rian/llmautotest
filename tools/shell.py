import subprocess
from langchain.tools import tool


SAFE_COMMANDS = {"semgrep", "bandit", "eslint", "gitleaks"}
BLOCK_COMMANDS = {"rm", "sudo", "curl", "wget"}

def ask_user_confirmation(cmd: str) -> bool:
    answer = input(f"Agent wants to execute: {cmd}\nAllow? (y/n): ")
    return answer.lower().startswith("y")

@tool
def run_shell_command(command: str):
    """
    Execute shell commands for code analysis tools.
    """

    parts = command.split()
    base_cmd = parts[0]

    if base_cmd in BLOCK_COMMANDS:
        return "Command blocked for security reasons."

    if base_cmd not in SAFE_COMMANDS:
        allowed = ask_user_confirmation(command)
        if not allowed:
            return "User denied command execution."

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=120
        )

        return result.stdout[:10000]

    except Exception as e:
        return str(e)