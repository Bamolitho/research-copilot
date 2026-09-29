from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.states import ResearchState
from src.llm.clients import answer_llm, planner_llm
from src.llm.prompt_loader import load_prompt
from src.tools.arxiv.search import arxiv_search

planner_with_tools = planner_llm.bind_tools([arxiv_search])


async def planner(state: ResearchState) -> dict:
    """Plan research by selecting the tools required to answer the question.

    Loads the planner system prompt, combines it with the current message
    history, and invokes the planner LLM. The LLM may return tool calls
    that are subsequently executed by the appropriate ToolNode.

    Args:
        state: Current research graph state containing the message history.

    Returns:
        A dictionary containing the planner AIMessage in ``messages``.
    """
    planner_system_prompt = load_prompt("planner")

    messages = [
        SystemMessage(content=planner_system_prompt),
        *state["messages"],
    ]

    response = await planner_with_tools.ainvoke(messages)

    return {"messages": [response]}


async def generate_answer(state: ResearchState) -> dict:
    """Generate the final answer from the question and research results.

    Loads the answer generator system prompt and builds a textual research
    context from the research results in the state. Each research query is
    included together with its corresponding structured sources before
    being provided to the answer LLM.

    Args:
        state: Current research graph state containing the question and
            research results grouped by query.

    Returns:
        A dictionary containing:
        - messages: The generated AIMessage.
        - answer: The generated answer as a string.
    """
    generator_system_prompt = load_prompt("generate_answer")

    # Serialize each source as text separated by blank lines before being provided to the answer LLM
    sources_context = "\n\n".join(
        f"Research query: {result['query']}\n"
        + "\n\n".join(
            f"Source {index}:\n"
            f"Title: {source.title}\n"
            f"Authors: {', '.join(source.authors)}\n"
            f"Abstract: {source.abstract}\n"
            f"arXiv ID: {source.arxiv_id}\n"
            f"URL: {source.url}"
            for source in result["sources"]
        )
        for index, result in enumerate(state["research_results"], start=1)
    )

    messages = [
        SystemMessage(content=generator_system_prompt),
        HumanMessage(
            content=(f"Question: {state['question']}\n\nResearch sources:\n\n{sources_context}")
        ),
    ]

    response = await answer_llm.ainvoke(messages)

    return {
        "messages": [response],
        "answer": response.content,
    }
