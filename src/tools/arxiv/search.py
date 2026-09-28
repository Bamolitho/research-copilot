import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from langchain_core.tools import tool
from pydantic import HttpUrl

from src.pydantic_models.source import Source

ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS = 1

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
}


@tool
async def arxiv_search(query: str) -> list[Source]:
    """Search scientific papers on arXiv."""

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

    return sources
