import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.pydantic_models.source import Source


class ResearchState(TypedDict):
    question: str
    answer: str
    sources: Annotated[list[Source], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]
