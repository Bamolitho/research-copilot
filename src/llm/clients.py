from langchain_openrouter import ChatOpenRouter

from src.config.env_config import settings

planner_llm = ChatOpenRouter(
    model=settings.PLANNER_MODEL,
    temperature=settings.TEMPERATURE,
    api_key=settings.OPEN_ROUTER_API_KEY,
    max_tokens=settings.MAX_TOKENS,
)

answer_llm = ChatOpenRouter(
    model=settings.GENERATOR_MODEL,
    temperature=settings.TEMPERATURE,
    api_key=settings.OPEN_ROUTER_API_KEY,
    max_tokens=settings.MAX_TOKENS,
)
