from __future__ import annotations

from pathlib import Path


class _StrictFormatDict(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def read_txt(path: str | Path) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Prompt txt file not found: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UnicodeError(f"Prompt txt file is not valid UTF-8: {target}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read prompt txt file: {target}") from exc

    normalized = text.strip()
    if not normalized:
        raise ValueError(f"Prompt txt file is empty: {target}")
    return normalized


def render_txt(path: str | Path, **kwargs: object) -> str:
    template = read_txt(path)
    try:
        return template.format_map(_StrictFormatDict(kwargs))
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Missing prompt template variable '{missing}' for: {path}") from exc
