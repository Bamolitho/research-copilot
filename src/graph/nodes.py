from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph.states import ResearchState
from src.llm.clients import answer_llm, planner_llm
from src.llm.prompt_loader import load_prompt
from src.tools.arxiv.search import arxiv_search

planner_with_tools = planner_llm.bind_tools([arxiv_search])


async def planner(state: ResearchState) -> dict:
    planner_system_prompt = load_prompt("planner")

    messages = [
        SystemMessage(content=planner_system_prompt),
        *state["messages"],
    ]

    response = await planner_with_tools.ainvoke(messages)

    return {"messages": [response]}


async def arxiv_tool_node(state: ResearchState) -> dict:
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return {
            "messages": [],
            "sources": [],
        }

    tool_messages = []
    sources = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] != "arxiv_search":
            continue

        result = await arxiv_search.ainvoke(tool_call["args"])
        sources.extend(result)

        content = "\n\n".join(
            f"Title: {source.title}\n"
            f"Authors: {', '.join(source.authors)}\n"
            f"Abstract: {source.abstract}\n"
            f"URL: {source.url}"
            for source in result
        )

        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name="arxiv_search",
            )
        )

    return {
        "messages": tool_messages,
        "sources": sources,
    }


async def generate_answer(state: ResearchState) -> dict:
    system_prompt = load_prompt("generate_answer")

    sources_context = "\n\n".join(
        f"Title: {source.title}\n"
        f"Authors: {', '.join(source.authors)}\n"
        f"Abstract: {source.abstract}\n"
        f"arXiv: {source.arxiv_id}\n"
        f"URL: {source.url}"
        for source in state["sources"]
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(f"Question: {state['question']}\n\nResearch sources:\n\n{sources_context}")
        ),
    ]

    response = await answer_llm.ainvoke(messages)

    return {
        "messages": [response],
        "answer": response.content,
    }
