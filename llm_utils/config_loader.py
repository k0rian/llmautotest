from typing import Any
import yaml
import os
from openai import OpenAI

from langchain_core.messages import HumanMessage

# Load config
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
    if not os.path.exists(config_path):
        # Try current directory
        config_path = "config.yml"
        
    cfg = yaml.safe_load(open(config_path))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(cfg, dict) or "ApiKey" not in cfg or "key" not in cfg["ApiKey"]:
        raise ValueError("ApiKey or key not found in config.yml")
    api_key:str = cfg["ApiKey"]["key"]  # pyright: ignore[reportCallIssue]
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
except Exception as e:
    print(f"Warning: Failed to load config.yml: {e}")
    api_key = "YOUR_API_KEY"
    base_url = "https://ark.cn-beijing.volces.com/api/v3"

def get_client():
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

def get_api_key():
    return api_key

def wrap_output(chain_output:Any):
    if isinstance(chain_output,str):
        return HumanMessage(content=chain_output)
    return chain_output
