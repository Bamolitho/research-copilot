"""A thin, pure HTTP client for the Research Copilot API.

Deliberately has no dependency on Streamlit (or any UI library): every
function here just calls the API and returns plain data. This is what
makes it testable without a UI runtime, and reusable from any frontend
that might replace Streamlit later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class HealthStatus:
    """Result of a successful GET /health call.

    Attributes:
        indexed_chunks: Number of chunks currently loaded in the index.
        embedding_model: Name of the embedding model the API is using.
    """

    indexed_chunks: int
    embedding_model: str


@dataclass(frozen=True)
class Citation:
    """One citation attached to an answer.

    Attributes:
        number: The citation number as referenced in the answer text (e.g. "[1]").
        source: The source document's filename.
        page_start: First page the excerpt appears on.
        page_end: Last page the excerpt appears on.
        score: Similarity score of the retrieved chunk.
        excerpt: The chunk's text.
    """

    number: int
    source: str
    page_start: int
    page_end: int
    score: float
    excerpt: str


@dataclass(frozen=True)
class AnswerResult:
    """Result of a successful POST /ask call.

    Attributes:
        answer: The generated answer text.
        citations: Every citation referenced in the answer, in order.
    """

    answer: str
    citations: list[Citation]


def get_health(base_url: str) -> HealthStatus | None:
    """Check whether the API is reachable and report its index status.

    Args:
        base_url: Base URL of the API, e.g. "http://127.0.0.1:8000".

    Returns:
        A HealthStatus if the API responded successfully, or None if
        the API is unreachable (connection refused, timeout, DNS
        failure) -- distinguished on purpose from an HTTP error
        response, which is a different kind of problem.
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    return HealthStatus(
        indexed_chunks=data["indexed_chunks"], embedding_model=data["embedding_model"]
    )


def ask_question(base_url: str, question: str, top_k: int | None = None) -> AnswerResult:
    """Ask a question and get back a grounded, cited answer.

    Args:
        base_url: Base URL of the API, e.g. "http://127.0.0.1:8000".
        question: The natural-language question to ask.
        top_k: Optional override for the number of retrieved chunks.

    Returns:
        The generated answer and its citations.

    Raises:
        requests.RequestException: If the API is unreachable, times
            out, or returns an error status (422 invalid request, 503
            empty index, 502 LLM backend failure). The caller decides
            how to present each case to the user.
    """
    payload: dict[str, Any] = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k

    response = requests.post(f"{base_url}/ask", json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    citations = [
        Citation(
            number=c["number"],
            source=c["source"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            score=c["score"],
            excerpt=c["excerpt"],
        )
        for c in data["citations"]
    ]
    return AnswerResult(answer=data["answer"], citations=citations)
