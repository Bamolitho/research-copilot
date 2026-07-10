"""Minimal arXiv client for automating the acquisition of source papers.

Talks to arXiv's public Atom API directly over HTTP, rather than using
a third-party `arxiv` package. This keeps the dependency footprint
small and every HTTP call easy to audit, in line with the project's
"self-hosted, auditable" principle.
"""

from __future__ import annotations

import argparse
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Recognizes queries that already use arXiv's field syntax (e.g. "all:RAG
# AND all:attention", "ti:transformer"), so we don't double-prefix them.
_FIELD_PREFIX_PATTERN = re.compile(r"\b(all|ti|abs|cat|au|co|jr|rn|id):", re.IGNORECASE)

# arXiv's own usage guideline asks for at most one request every 3 seconds.
REQUEST_DELAY_SECONDS = 3.0


@dataclass(frozen=True)
class ArxivPaper:
    """Metadata for one arXiv paper, enough to download and cite it.

    Attributes:
        arxiv_id: The arXiv identifier, e.g. "1704.00051v2".
        title: The paper's title.
        authors: Author names, in the order arXiv lists them.
        published: ISO 8601 publication timestamp, as returned by arXiv.
        pdf_url: Direct URL to the paper's PDF.
    """

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    pdf_url: str


class ArxivDownloader:
    """Searches arXiv and downloads paper PDFs to a local directory.

    Args:
        download_dir: Directory where PDFs are saved. Created if it
            does not already exist.
        session: Optional `requests.Session` to reuse (e.g. to inject
            a mock session in tests). A new session is created if omitted.
    """

    def __init__(
        self,
        download_dir: Path | str,
        session: requests.Session | None = None,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()

    def search(self, query: str, max_results: int = 20) -> list[ArxivPaper]:
        """Search arXiv and return paper metadata, most relevant first.

        Args:
            query: Either a plain-text topic, e.g. "attention mechanism"
                (automatically searched across all fields), or an
                already-qualified arXiv query, e.g.
                "all:RAG AND all:attention" (used as-is, untouched).
            max_results: Maximum number of papers to return.

        Returns:
            A list of ArxivPaper. May be shorter than `max_results` if
            arXiv has fewer matches.

        Raises:
            requests.HTTPError: If the arXiv API request fails.
        """
        search_query = query if _FIELD_PREFIX_PATTERN.search(query) else f"all:{query}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
        }
        response = self._session.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return self._parse_feed(response.text)

    def download(self, paper: ArxivPaper, overwrite: bool = False) -> Path:
        """Download one paper's PDF, skipping it if already present locally.

        Args:
            paper: The paper to download, as returned by `search`.
            overwrite: If False (default), an existing local file is
                left untouched and no network request is made.

        Returns:
            The local path to the downloaded (or already-existing) PDF.

        Raises:
            requests.HTTPError: If the download request fails.
        """
        destination = self.download_dir / f"{self._safe_filename(paper.arxiv_id)}.pdf"
        if destination.exists() and not overwrite:
            return destination

        response = self._session.get(paper.pdf_url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        time.sleep(REQUEST_DELAY_SECONDS)
        return destination

    @staticmethod
    def _safe_filename(arxiv_id: str) -> str:
        """Turn an arXiv id into a flat, filesystem-safe filename.

        Old-style ids contain a slash (e.g. "hep-ex/0307069v1"), which
        Path would otherwise interpret as a subdirectory that doesn't exist.
        """
        return arxiv_id.replace("/", "_")

    @staticmethod
    def _extract_arxiv_id(raw_id: str) -> str:
        """Extract the arXiv id from an Atom <id> URL.

        New-style ids have no slash, e.g. "2405.02292v1". Old-style ids
        (pre-2007) include a subject-class prefix with a slash, e.g.
        "hep-ex/0307069v1" -- that prefix must be kept, or the resulting
        PDF URL silently points at the wrong (non-existent) paper.
        """
        marker = "/abs/"
        index = raw_id.find(marker)
        if index == -1:
            raise ValueError(f"Unexpected arXiv id URL: {raw_id!r}")
        return raw_id[index + len(marker) :]

    @staticmethod
    def _parse_feed(atom_xml: str) -> list[ArxivPaper]:
        """Turn an arXiv Atom feed response into a list of ArxivPaper."""
        root = ET.fromstring(atom_xml)
        papers: list[ArxivPaper] = []

        for entry in root.findall("atom:entry", _ATOM_NS):
            raw_id = entry.findtext("atom:id", default="", namespaces=_ATOM_NS)
            arxiv_id = ArxivDownloader._extract_arxiv_id(raw_id)
            title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
            authors = [
                author.findtext("atom:name", default="", namespaces=_ATOM_NS) or ""
                for author in entry.findall("atom:author", _ATOM_NS)
            ]
            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    published=published,
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                )
            )
        return papers


def _run_cli() -> None:
    """Command-line entry point: search arXiv and download matching papers.

    Nothing is downloaded without at least one explicit --query: this is
    a search, not a bulk export of arXiv.

    Pass --query multiple times to pull several topics into one corpus;
    a paper matching more than one topic is only downloaded once.

    Example:
        uv run python3 -m ingestion.downloader \\
            --query "retrieval augmented generation" \\
            --query "attention mechanism" \\
            --query "large language models" \\
            --max-results 20 \\
            --download-dir data/papers
    """
    parser = argparse.ArgumentParser(
        description="Search arXiv and download the matching papers as PDFs."
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Search topic, e.g. 'retrieval augmented generation'. "
        "Repeat --query for several topics (RAG, AI, attention mechanism, ...).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of papers to fetch per topic (default: 20)",
    )
    parser.add_argument(
        "--download-dir", default="data/papers", help="Directory to save PDFs into (default: data/papers)"
    )
    args = parser.parse_args()

    downloader = ArxivDownloader(download_dir=args.download_dir)

    # Search each topic separately, then dedupe by arxiv_id: a paper
    # matching two topics (e.g. "RAG" and "attention mechanism") must
    # only be downloaded once.
    papers_by_id: dict[str, ArxivPaper] = {}
    for query in args.queries:
        found = downloader.search(query, max_results=args.max_results)
        print(f"Found {len(found)} paper(s) for topic: {query!r}")
        for paper in found:
            papers_by_id[paper.arxiv_id] = paper

    papers = list(papers_by_id.values())
    print(f"\n{len(papers)} unique paper(s) to download across {len(args.queries)} topic(s)")

    for index, paper in enumerate(papers, start=1):
        print(f"[{index}/{len(papers)}] {paper.arxiv_id} \u2014 {paper.title}")
        path = downloader.download(paper)
        print(f"    saved to {path}")


if __name__ == "__main__":
    _run_cli()
