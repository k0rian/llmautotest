from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from services.lsp.server import LSPServer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "PROMPT.md"
DEFAULT_SKILL_DIR = PROJECT_ROOT / "skill"
DEFAULT_IGNORES = {
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
}


@dataclass(frozen=True)
class SkillRule:
    key: str
    skill_file: str
    extensions: tuple[str, ...] = ()
    lsp_server: str = ""
    probe_extension: str = ""


DEFAULT_RULES: tuple[SkillRule, ...] = (
    SkillRule(
        key="python",
        skill_file="PY_SKILL.md",
        extensions=(".py", ".pyi"),
        lsp_server="python",
        probe_extension=".py",
    ),
    SkillRule(
        key="javascript",
        skill_file="JS_SKILL.md",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        lsp_server="typescript",
        probe_extension=".js",
    ),
    SkillRule(
        key="typescript",
        skill_file="JS_SKILL.md",
        extensions=(".ts", ".tsx", ".mts", ".cts"),
        lsp_server="typescript",
        probe_extension=".ts",
    ),
)


def _strip_frontmatter(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    return parts[2].strip()


def _read_markdown(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return _strip_frontmatter(content).strip()


def _iter_source_files(workspace_path: Path, ignores: set[str]) -> Iterable[Path]:
    stack: list[Path] = [workspace_path]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except Exception:
            continue
        for child in children:
            name = child.name
            if name in ignores:
                continue
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                yield child


def detect_languages(
    workspace_path: str | Path,
    rules: Iterable[SkillRule] = DEFAULT_RULES,
    ignores: set[str] | None = None,
) -> dict[str, int]:
    workspace = Path(workspace_path).resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError(f"invalid workspace path: {workspace_path}")
    rule_list = list(rules)
    counter = {rule.key: 0 for rule in rule_list}
    ignore_names = set(DEFAULT_IGNORES)
    if isinstance(ignores, set):
        ignore_names.update(ignores)
    extension_map: dict[str, list[SkillRule]] = {}
    for rule in rule_list:
        for ext in rule.extensions:
            key = ext.lower().strip()
            if not key:
                continue
            extension_map.setdefault(key, []).append(rule)
    for file_path in _iter_source_files(workspace, ignore_names):
        suffix = file_path.suffix.lower()
        targets = extension_map.get(suffix, [])
        for target in targets:
            counter[target.key] = counter.get(target.key, 0) + 1
    return counter


def route_skill_files(
    workspace_path: str | Path,
    skill_dir: str | Path = DEFAULT_SKILL_DIR,
    rules: Iterable[SkillRule] = DEFAULT_RULES,
) -> list[Path]:
    rule_list = list(rules)
    scores = detect_languages(workspace_path=workspace_path, rules=rule_list)
    selected: list[SkillRule] = [rule for rule in rule_list if scores.get(rule.key, 0) > 0]
    selected.sort(key=lambda item: (-scores.get(item.key, 0), item.key))
    root = Path(skill_dir).resolve()
    result: list[Path] = []
    visited: set[str] = set()
    for rule in selected:
        if rule.skill_file in visited:
            continue
        path = root / rule.skill_file
        if path.exists() and path.is_file():
            result.append(path)
            visited.add(rule.skill_file)
    return result


def build_system_prompt(
    workspace_path: str | Path,
    base_prompt_file: str | Path = DEFAULT_PROMPT_FILE,
    skill_dir: str | Path = DEFAULT_SKILL_DIR,
    rules: Iterable[SkillRule] = DEFAULT_RULES,
) -> str:
    base = _read_markdown(Path(base_prompt_file).resolve())
    skill_files = route_skill_files(workspace_path=workspace_path, skill_dir=skill_dir, rules=rules)
    sections = [base] if base else []
    for file_path in skill_files:
        content = _read_markdown(file_path)
        if content:
            sections.append(content)
    return "\n\n".join(part for part in sections if part).strip()


def validate_workspace_lsp(
    workspace_path: str | Path,
    rules: Iterable[SkillRule] = DEFAULT_RULES,
) -> dict[str, Any]:
    workspace = Path(workspace_path).resolve()
    rule_list = list(rules)
    scores = detect_languages(workspace_path=workspace, rules=rule_list)
    server = LSPServer(workspace_path=str(workspace))
    checks: list[dict[str, Any]] = []
    seen_servers: set[str] = set()
    for rule in rule_list:
        if scores.get(rule.key, 0) <= 0:
            continue
        profile = rule.lsp_server.strip()
        suffix = rule.probe_extension.strip()
        if not profile or not suffix:
            checks.append(
                {
                    "language": rule.key,
                    "server": "",
                    "status": "skipped",
                    "reason": "no_lsp_mapping",
                }
            )
            continue
        if profile in seen_servers:
            checks.append(
                {
                    "language": rule.key,
                    "server": profile,
                    "status": "ok",
                    "reason": "checked_by_shared_server",
                }
            )
            continue
        probe_path = str(workspace / f".skill_probe{suffix}")
        try:
            server.profile_for_file(probe_path, preferred_server=profile)
            checks.append({"language": rule.key, "server": profile, "status": "ok"})
            seen_servers.add(profile)
        except Exception as exc:
            checks.append(
                {
                    "language": rule.key,
                    "server": profile,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
    errors = [item for item in checks if item.get("status") == "failed"]
    return {
        "ready": len(errors) == 0,
        "checks": checks,
        "errors": errors,
    }


def build_perception_result(
    workspace_path: str | Path,
    base_prompt_file: str | Path = DEFAULT_PROMPT_FILE,
    skill_dir: str | Path = DEFAULT_SKILL_DIR,
    rules: Iterable[SkillRule] = DEFAULT_RULES,
) -> dict[str, Any]:
    snapshot = build_router_snapshot(workspace_path=workspace_path, skill_dir=skill_dir, rules=rules)
    lsp_validation = validate_workspace_lsp(workspace_path=workspace_path, rules=rules)
    prompt = build_system_prompt(
        workspace_path=workspace_path,
        base_prompt_file=base_prompt_file,
        skill_dir=skill_dir,
        rules=rules,
    )
    return {
        "workspace_path": str(Path(workspace_path).resolve()),
        "language_scores": snapshot["language_scores"],
        "skill_files": snapshot["skill_files"],
        "lsp_validation": lsp_validation,
        "system_prompt": prompt,
    }


def build_router_snapshot(
    workspace_path: str | Path,
    skill_dir: str | Path = DEFAULT_SKILL_DIR,
    rules: Iterable[SkillRule] = DEFAULT_RULES,
) -> dict[str, object]:
    rule_list = list(rules)
    scores = detect_languages(workspace_path=workspace_path, rules=rule_list)
    files = route_skill_files(workspace_path=workspace_path, skill_dir=skill_dir, rules=rule_list)
    return {
        "workspace_path": str(Path(workspace_path).resolve()),
        "language_scores": scores,
        "skill_files": [str(item) for item in files],
    }


__all__ = [
    "SkillRule",
    "DEFAULT_RULES",
    "detect_languages",
    "route_skill_files",
    "build_system_prompt",
    "validate_workspace_lsp",
    "build_perception_result",
    "build_router_snapshot",
]
