import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.pydantic_models.source import Source


class ResearchResult(TypedDict):
    query: str
    sources: list[Source]


class ResearchState(TypedDict):
    question: str
    answer: str
    research_results: Annotated[list[ResearchResult], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]
