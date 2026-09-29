from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.graph.nodes import generate_answer, planner
from src.graph.routing import route_after_planner
from src.graph.states import ResearchState
from src.tools.arxiv.search import arxiv_search

research_tool_node = ToolNode([arxiv_search])


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner)
    builder.add_node("research_tools", research_tool_node)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"research_tools": "research_tools", "generate_answer": "generate_answer"},
    )

    builder.add_edge("research_tools", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()
