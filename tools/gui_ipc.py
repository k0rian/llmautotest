import json
import os
from threading import Lock
from typing import Any

from langchain.tools import tool

_GUI_AGENT_INSTANCE: Any | None = None
_GUI_AGENT_LOCK = Lock()


def _get_gui_agent(backend: str | None = None):
    global _GUI_AGENT_INSTANCE
    with _GUI_AGENT_LOCK:
        if _GUI_AGENT_INSTANCE is not None:
            return _GUI_AGENT_INSTANCE

        # Lazy import to avoid loading heavy GUI dependencies at module import time.
        from gui_agent.agent import GUIAgent, GUIAgentConfig

        selected_backend = (backend or os.getenv("GUI_AGENT_BACKEND", "server")).strip().lower()
        if selected_backend not in {"local", "server"}:
            selected_backend = "server"
        config = GUIAgentConfig(backend=selected_backend)
        _GUI_AGENT_INSTANCE = GUIAgent(config=config)
        return _GUI_AGENT_INSTANCE


@tool
def gui_agent_run(
    instruction: str,
    max_steps: int = 8,
    reset_history: bool = False,
    backend: str = "",
) -> str:
    """
    Execute a GUI task through gui_agent and return structured JSON result.
    """
    try:
        task = (instruction or "").strip()
        if not task:
            return "GUI tool error: instruction cannot be empty"

        step_limit = max(1, min(int(max_steps), 50))
        agent = _get_gui_agent(backend=backend or None)

        if bool(reset_history) and hasattr(agent, "reset_history"):
            agent.reset_history()

        payload = agent.run(task, max_steps=step_limit)
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return json.dumps({"status": "failed", "message": str(payload)}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"GUI tool error: {exc}"
