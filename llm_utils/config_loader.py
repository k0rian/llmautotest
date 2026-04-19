from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from langchain_core.messages import HumanMessage

DEFAULT_API_KEY = "YOUR_API_KEY"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_NAME = "doubao-seed-2-0-lite-260215"
PROJECT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yml"


def _resolve_config_file() -> Path:
    if PROJECT_CONFIG_FILE.exists():
        return PROJECT_CONFIG_FILE
    return Path("config.yml")


def load_config() -> dict[str, Any]:
    config_file = _resolve_config_file()
    if not config_file.exists():
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def load_api_key() -> str:
    payload = load_config()
    llm_section = payload.get("LLM", {}) if isinstance(payload, dict) else {}
    value = llm_section.get("api_key", "") if isinstance(llm_section, dict) else ""
    return value or DEFAULT_API_KEY


def load_base_url() -> str:
    payload = load_config()
    llm_section = payload.get("LLM", {}) if isinstance(payload, dict) else {}
    value = llm_section.get("base_url", "") if isinstance(llm_section, dict) else ""
    return value or DEFAULT_BASE_URL


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "disabled"}:
            return False
    return default


def load_model_name(default: str = DEFAULT_MODEL_NAME) -> str:
    payload = load_config()
    llm_section = payload.get("LLM", {}) if isinstance(payload, dict) else {}
    if isinstance(llm_section, dict):
        value = llm_section.get("model_name", "") or llm_section.get("model", "")
        if value:
            return str(value)
    value = ""
    if isinstance(payload, dict):
        value = payload.get("model_name", "") or payload.get("model", "")
    return str(value) if value else default


def load_llm_summary_enabled(default: bool = False) -> bool:
    payload = load_config()
    llm_section = payload.get("LLM", {}) if isinstance(payload, dict) else {}
    if not isinstance(llm_section, dict):
        return default
    for key in ("use_summary", "use_llm_summary", "semantic_summary", "semantic_index_summary"):
        if key in llm_section:
            return _as_bool(llm_section.get(key), default=default)
    return default


def get_client() -> OpenAI:
    return OpenAI(
        base_url=load_base_url(),
        api_key=load_api_key(),
    )


def get_api_key() -> str:
    return load_api_key()


def wrap_output(chain_output: Any):
    if isinstance(chain_output, str):
        return HumanMessage(content=chain_output)
    return chain_output
