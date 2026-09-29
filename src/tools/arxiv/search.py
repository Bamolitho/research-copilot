import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from pydantic import HttpUrl
from typing import Annotated

from src.pydantic_models.source import Source

ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS = 1

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
}


@tool
async def arxiv_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search scientific papers on arXiv.

    Searches arXiv for relevant papers and updates the graph state with
    structured research sources. The tool_call_id is injected automatically
    by LangGraph and is used to associate the ToolMessage with the
    corresponding tool call.

    Args:
        query: Search query used against arXiv.
        tool_call_id: Automatically injected identifier of the tool call.

    Returns:
        Command updating the state with:
        - sources: A list of structured Source objects, e.g.
          [Source(...), Source(...), ...].
        - messages: A ToolMessage confirming the number of sources found.
    """

    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": MAX_RESULTS,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )

    url = f"{ARXIV_API_URL}?{params}"

    with urllib.request.urlopen(url, timeout=10) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)

    sources: list[Source] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
        abstract = entry.findtext("atom:summary", default="", namespaces=ATOM_NS)

        arxiv_id_url = entry.findtext(
            "atom:id",
            default="",
            namespaces=ATOM_NS,
        )

        authors = [
            author.findtext(
                "atom:name",
                default="",
                namespaces=ATOM_NS,
            )
            for author in entry.findall("atom:author", ATOM_NS)
        ]

        if not arxiv_id_url:
            continue

        arxiv_id = arxiv_id_url.rstrip("/").split("/")[-1]

        sources.append(
            Source(
                title=" ".join(title.split()),
                authors=authors,
                abstract=" ".join(abstract.split()),
                url=HttpUrl(arxiv_id_url),
                arxiv_id=arxiv_id,
            )
        )

    return Command(
        update={
            "sources": sources,
            "messages": [
                ToolMessage(
                    content=f"Found {len(sources)} research sources.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )