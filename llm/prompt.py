"""Builds the final RAG prompt from a question and its retrieved chunks.

Turns a list of vector_db.faiss_store.SearchResult into a single prompt
string -- a system instruction, numbered context chunks, then the
question -- following the same context-only, cite-your-sources
structure taught in the course. The LLM never sees embeddings here,
only the chunks' original text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vector_db.faiss_store import SearchResult

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Answer only using the numbered excerpts "
    "provided in the context below. Do not use any outside knowledge, even "
    "if you believe it is correct. If the context does not contain the "
    "answer, respond exactly: \"I don't know based on the retrieved "
    'documents." Cite the excerpt numbers you relied on, e.g. [1], [2].'
)


@dataclass(frozen=True)
class RagPrompt:
    """A fully assembled prompt, plus the citation numbers it defines.

    Attributes:
        text: The prompt string to send to the generator.
        citations: Maps each citation number used in the prompt (e.g. 1
            for the excerpt marked "[1]") back to the SearchResult it
            came from, so a caller can later resolve a "[1]" in the
            generated answer to a real source, page, and score.
    """

    text: str
    citations: dict[int, SearchResult]


def build_prompt(
    question: str,
    results: Sequence[SearchResult],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> RagPrompt:
    """Assemble the final prompt sent to the LLM.

    Args:
        question: The user's natural-language question.
        results: Retrieved chunks, best match first, as returned by
            vector_db.faiss_store.FaissVectorStore.search.
        system_prompt: The system instruction prepended to the prompt.
            Defaults to a strict context-only instruction.

    Returns:
        A RagPrompt: the prompt text, and a citation map from number to
        SearchResult.

    Raises:
        ValueError: If `results` is empty. There is nothing to ground
            an answer in, and the caller should decide explicitly how
            to handle a failed retrieval, rather than silently
            prompting the model with no context at all.
    """
    if not results:
        raise ValueError("results must not be empty: nothing to ground the answer in")

    citations = dict(enumerate(results, start=1))
    context_block = "\n\n".join(
        f"[{index}] {result.chunk.text}" for index, result in citations.items()
    )

    text = (
        f"{system_prompt}\n\n"
        "-------------------------\n"
        "CONTEXT\n\n"
        f"{context_block}\n\n"
        "-------------------------\n"
        "QUESTION\n"
        f"{question}\n\n"
        "-------------------------\n"
        "ANSWER"
    )

    return RagPrompt(text=text, citations=citations)
