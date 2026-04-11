import json
from typing import Any

from planner.types import AuditState


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


def format_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        title = read_text(step.get("title", f"Step {index}")).strip() or f"Step {index}"
        mode = read_text(step.get("mode", "code_audit")).strip() or "code_audit"
        objective = read_text(step.get("objective", "")).strip()
        lines.append(f"{index}. [{mode}] {title}")
        if objective:
            lines.append(f"   - {objective}")
    return "\n".join(lines)


def build_perception_blocked_state(message: str) -> AuditState:
    return {
        "planner_summary": "Stopped in perception stage",
        "plan": "Execution stage was skipped",
        "audit_output": f"Perception gate blocked execution: {message}",
    }
