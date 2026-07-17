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

# Some models (notably Qwen3) default to emitting a slow internal
# reasoning trace before every answer, even for a simple grounded
# question -- pure overhead here, since the answer must come from the
# context, not from extended reasoning. This instruction is Qwen's
# documented way to disable it, in theory -- in practice, it was
# CONFIRMED NON-FUNCTIONAL against qwen3:4b on Ollama: the model
# treated it as literal text to analyze rather than a control
# instruction. Kept as an opt-in (default off) in case it behaves
# differently on another build; do not assume it works untested.
NO_THINK_SUFFIX = "\n\n/no_think"


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
    disable_thinking: bool = False,
) -> RagPrompt:
    """Assemble the final prompt sent to the LLM.

    Args:
        question: The user's natural-language question.
        results: Retrieved chunks, best match first, as returned by
            vector_db.faiss_store.FaissVectorStore.search.
        system_prompt: The system instruction prepended to the prompt.
            Defaults to a strict context-only instruction.
        disable_thinking: If True, appends an instruction intended to
            turn off Qwen3's internal reasoning trace (see
            NO_THINK_SUFFIX). Not guaranteed to work: confirmed
            non-functional against qwen3:4b on Ollama in practice, so
            this defaults to False. Test it against your specific
            model/server before relying on it.

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

    effective_system_prompt = system_prompt + NO_THINK_SUFFIX if disable_thinking else system_prompt

    citations = dict(enumerate(results, start=1))
    context_block = "\n\n".join(
        f"[{index}] {result.chunk.text}" for index, result in citations.items()
    )

    text = (
        f"{effective_system_prompt}\n\n"
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
