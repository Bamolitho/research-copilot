from typing import Literal

from langchain_core.messages import AIMessage

from src.graph.states import ResearchState


def route_after_planner(state: ResearchState) -> Literal["tool_node", "generate_answer"]:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    return "generate_answer"
