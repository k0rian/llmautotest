from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Callable


InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]


ALWAYS_ALLOW_EXECUTABLES = {
    "ls",
    "dir",
    "type",
    "cat",
    "find",
    "findstr",
    "grep",
    "rg",
    "git",
    "semgrep",
    "bandit",
    "gitleaks",
    "eslint",
    "npm",
    "pnpm",
    "yarn",
    "pytest",
}

NEVER_ALLOW_EXECUTABLES = {
    "format",
    "diskpart",
    "shutdown",
    "reboot",
    "poweroff",
}

SHELL_WRAPPER_EXECUTABLES = {
    "cmd",
    "powershell",
    "pwsh",
    "bash",
    "sh",
    "zsh",
}

DANGEROUS_KEYWORDS = {
    "rm",
    "rmdir",
    "del",
    "erase",
    "mkfs",
    "dd",
    "chown",
    "chmod",
}


@dataclass
class CommandEvaluation:
    argv: list[str]
    executable: str
    allowed: bool
    needs_confirmation: bool
    reason: str


class SessionPermissionStore:
    """Simple in-memory approvals for the current process."""

    def __init__(self) -> None:
        self._approved_executables: set[str] = set()

    def is_approved(self, executable: str) -> bool:
        return executable in self._approved_executables

    def approve(self, executable: str) -> None:
        self._approved_executables.add(executable)


SESSION_PERMISSION_STORE = SessionPermissionStore()


def parse_command(command: str) -> list[str]:
    raw = (command or "").strip()
    if not raw:
        raise ValueError("command cannot be empty")
    try:
        argv = shlex.split(raw, posix=False)
    except Exception as exc:
        raise ValueError(f"invalid command syntax: {exc}") from exc
    if not argv:
        raise ValueError("command cannot be empty")
    return argv


def evaluate_command(argv: list[str]) -> CommandEvaluation:
    executable = os.path.basename(argv[0]).strip("\"' ").lower()
    if not executable:
        return CommandEvaluation(
            argv=argv,
            executable="",
            allowed=False,
            needs_confirmation=False,
            reason="missing executable",
        )

    if executable in NEVER_ALLOW_EXECUTABLES:
        return CommandEvaluation(
            argv=argv,
            executable=executable,
            allowed=False,
            needs_confirmation=False,
            reason=f"blocked executable: {executable}",
        )

    lowered_tokens = [token.lower() for token in argv]
    if any(token in DANGEROUS_KEYWORDS for token in lowered_tokens):
        return CommandEvaluation(
            argv=argv,
            executable=executable,
            allowed=True,
            needs_confirmation=True,
            reason="contains potentially destructive operation",
        )

    if executable in SHELL_WRAPPER_EXECUTABLES:
        return CommandEvaluation(
            argv=argv,
            executable=executable,
            allowed=True,
            needs_confirmation=True,
            reason="shell wrapper can execute arbitrary commands",
        )

    if executable in ALWAYS_ALLOW_EXECUTABLES:
        return CommandEvaluation(
            argv=argv,
            executable=executable,
            allowed=True,
            needs_confirmation=False,
            reason="trusted command",
        )

    return CommandEvaluation(
        argv=argv,
        executable=executable,
        allowed=True,
        needs_confirmation=True,
        reason="untrusted command requires approval",
    )


def request_command_permission(
    command: str,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> tuple[bool, CommandEvaluation]:
    argv = parse_command(command)
    evaluation = evaluate_command(argv)
    if not evaluation.allowed:
        return False, evaluation

    if SESSION_PERMISSION_STORE.is_approved(evaluation.executable):
        return True, evaluation

    if not evaluation.needs_confirmation:
        return True, evaluation

    if not os.isatty(0):
        return False, CommandEvaluation(
            argv=evaluation.argv,
            executable=evaluation.executable,
            allowed=False,
            needs_confirmation=False,
            reason="interactive confirmation required but stdin is not a tty",
        )

    print_fn("")
    print_fn("┌─────────────────────────────────────────────┐")
    print_fn("│ Command Permission Request                  │")
    print_fn("└─────────────────────────────────────────────┘")
    print_fn(f"Reason : {evaluation.reason}")
    print_fn(f"Command: {' '.join(evaluation.argv)}")
    print_fn("Choices: [y] allow once, [a] always allow this executable, [n] deny")
    answer = input_fn("> ").strip().lower()

    if answer == "a":
        SESSION_PERMISSION_STORE.approve(evaluation.executable)
        return True, evaluation
    if answer in {"y", "yes"}:
        return True, evaluation
    return False, evaluation

