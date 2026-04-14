import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field


class CodeScope(BaseModel):
    input_path: str
    resolved_path: str
    scope_type: Literal["file", "directory"]
    root_path: str
    index_root_path: str
    include_glob: str
    patterns: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)


def normalize_include_glob(include_glob: str, default: str) -> list[str]:
    raw = (include_glob or "").strip() or default
    return [part.strip() for part in raw.split(",") if part.strip()]


def ensure_within_workspace(workspace_path: str, target_path: str) -> str:
    workspace = str(Path(workspace_path).resolve())
    resolved = Path(target_path)
    if not resolved.is_absolute():
        resolved = Path(workspace) / resolved
    resolved_str = str(resolved.resolve())
    try:
        Path(resolved_str).resolve().relative_to(Path(workspace).resolve())
    except Exception as exc:
        raise ValueError(f"scope_invalid: target path '{target_path}' is outside workspace") from exc
    return resolved_str


def resolve_workspace_target(workspace_path: str, candidate: str) -> tuple[str, str]:
    text = (candidate or "").strip()
    if not text:
        return str(Path(workspace_path).resolve()), ""
    try:
        return ensure_within_workspace(workspace_path, text), ""
    except Exception as exc:
        return "", str(exc)


def resolve_code_scope(
    path: str,
    include_glob: str,
    default_include_glob: str,
    max_files: int,
    skip_dirs: Iterable[str],
) -> CodeScope:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise ValueError(f"path not found: {path}")

    patterns = normalize_include_glob(include_glob, default_include_glob)
    if resolved.is_file():
        return CodeScope(
            input_path=path,
            resolved_path=str(resolved),
            scope_type="file",
            root_path=str(resolved.parent),
            index_root_path=str(resolved.parent),
            include_glob=include_glob or default_include_glob,
            patterns=patterns,
            target_files=[str(resolved)],
        )

    if not resolved.is_dir():
        raise ValueError(f"invalid path '{path}'")

    files = iter_code_files(
        root=resolved,
        patterns=patterns,
        max_files=max_files,
        skip_dirs=set(skip_dirs),
    )
    return CodeScope(
        input_path=path,
        resolved_path=str(resolved),
        scope_type="directory",
        root_path=str(resolved),
        index_root_path=str(resolved),
        include_glob=include_glob or default_include_glob,
        patterns=patterns,
        target_files=[str(item.resolve()) for item in files],
    )


def iter_code_files(root: Path, patterns: list[str], max_files: int, skip_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    for current_root, dirs, names in os.walk(root):
        dirs[:] = [item for item in dirs if item not in skip_dirs]
        for name in names:
            full = Path(current_root) / name
            if patterns and not any(fnmatch(full.name, pattern) for pattern in patterns):
                continue
            files.append(full)
            if len(files) >= max(1, int(max_files)):
                return files
    return files
