import json
import re
from pathlib import Path
from typing import Any


def read_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "messages" in value and value["messages"]:
            return read_text(value["messages"][-1])
        if "output" in value:
            return read_text(value["output"])
        return json.dumps(value, ensure_ascii=False, indent=2)
    content = getattr(value, "content", None)
    if content is not None:
        return read_text(content)
    return str(value)


def parse_json_payload(
    text: str,
    default: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    raw = (text or "").strip()
    fallback = default or {}
    if not raw:
        return fallback
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
        if match:
            raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        if strict:
            raise
        return fallback


def strip_frontmatter(raw: str) -> str:
    text = (raw or "").strip()
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    return parts[2].strip()


def read_markdown(path: str | Path) -> str:
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        return strip_frontmatter(file_path.read_text(encoding="utf-8")).strip()
    except Exception:
        return ""
