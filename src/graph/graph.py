from langgraph.graph import END, START, StateGraph

from src.graph.nodes import arxiv_tool_node, generate_answer, planner
from src.graph.routing import route_after_planner
from src.graph.states import ResearchState


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner)
    builder.add_node("tool_node", arxiv_tool_node)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tool_node": "tool_node", "generate_answer": "generate_answer"},
    )

    builder.add_edge("tool_node", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()
