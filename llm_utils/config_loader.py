from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from langchain_core.messages import HumanMessage

DEFAULT_API_KEY = "YOUR_API_KEY"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
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
    api_section = payload.get("ApiKey", {}) if isinstance(payload, dict) else {}
    value = api_section.get("key", "") if isinstance(api_section, dict) else ""
    return value or DEFAULT_API_KEY


def load_base_url() -> str:
    payload = load_config()
    llm_section = payload.get("LLM", {}) if isinstance(payload, dict) else {}
    value = llm_section.get("base_url", "") if isinstance(llm_section, dict) else ""
    return value or DEFAULT_BASE_URL


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
