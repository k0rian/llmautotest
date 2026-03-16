from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from llm_utils.config_loader import get_api_key
from tools.tools import build_tools


tools = build_tools()

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=get_api_key(),
)

agent = create_agent(
    model,
    tools=tools,
    verbose=True,
)





