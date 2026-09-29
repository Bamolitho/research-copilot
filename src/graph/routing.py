from typing import Literal

from langchain_core.messages import AIMessage

from src.graph.states import ResearchState


def route_after_planner(state: ResearchState) -> Literal["research_tools", "generate_answer"]:
    """Route the graph according to the planner LLM output.

    Inspects the last message produced by the planner. Routes to the
    research ToolNode when the planner requested tool calls, otherwise
    routes directly to the answer generator.

    Args:
        state: Current research graph state.

    Returns:
        ``"research_tools"`` when tool calls are present, otherwise
        ``"generate_answer"``.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "research_tools"

    return "generate_answer"
