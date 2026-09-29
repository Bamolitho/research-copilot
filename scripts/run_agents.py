import asyncio

from langchain_core.messages import HumanMessage

from src.graph.graph import build_graph
from src.graph.states import ResearchState
from src.utils.pretty import pretty_print_state

QUESTIONS = [
    "What are the main applications of graph neural networks?",
    "What are the advantages of graph neural networks?",
    "What are the main challenges of graph neural networks?",
    "Which team has won the 2026 world cup?",
    "Why can LLMs support tool calls right now?",
]


async def main():
    graph = build_graph()

    for question in QUESTIONS:
        initial_state = ResearchState(
            question=question,
            answer="",
            research_results=[],
            messages=[HumanMessage(content=question)],
        )

        result = await graph.ainvoke(initial_state)
        pretty_print_state(result)


if __name__ == "__main__":
    asyncio.run(main())
