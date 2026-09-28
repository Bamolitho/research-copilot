import asyncio

from langchain_core.messages import HumanMessage

from src.graph.graph import build_graph
from src.graph.states import ResearchState
from src.utils.pretty import pretty_print_state


async def main():
    graph = build_graph()

    initial_state = ResearchState(
        question="What are the main application of graph neural networks?",
        answer="",
        sources=[],
        messages=[HumanMessage(content="What are the main application of graph neural networks?")],
    )

    result = await graph.ainvoke(initial_state)
    pretty_print_state(result)


if __name__ == "__main__":
    asyncio.run(main())
