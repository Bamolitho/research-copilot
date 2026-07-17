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
    'documents." '
    "Cite a source after every factual sentence, using the excerpt's number "
    'in brackets, e.g. "Panels reach 22% efficiency [2]." If a sentence '
    'draws on more than one excerpt, cite all of them, e.g. "[1][3]." '
    "Never state a fact without a citation, and never merge several "
    "excerpts into one sentence without citing each of them. "
    "Start your answer with a direct sentence, not with a citation number "
    "on its own. "
    "Address information from every excerpt that is relevant to the "
    "question, not only the first one or two. Do not repeat the same "
    "point twice in different words. "
    "The excerpts may contain their own citations from the original "
    'paper (e.g. author-year references like "[Smith, 2020]") -- ignore '
    "these completely and never reproduce them; the only citation numbers "
    "you may ever write are the excerpt numbers given to you here."
)

# A worked example showing the exact citation pattern expected, not just
# describing it in prose. Small models follow a concrete demonstration
# far more reliably than an abstract instruction -- this was added after
# observing a 3B model merge multiple distinct sources into a single,
# uncited paragraph despite the system prompt already asking for
# per-sentence citations. Uses letters ([A], [B]), not numbers, so the
# model can never confuse this example's citations with the real
# numbered context that follows it.
_EXAMPLE_BLOCK = (
    "-------------------------\n"
    "EXAMPLE (illustrates the expected format only, not real data)\n\n"
    "CONTEXT\n\n"
    "[A] Solar panels convert sunlight into electricity through the "
    "photovoltaic effect.\n\n"
    "[B] Commercial silicon panels typically reach 15% to 22% efficiency.\n\n"
    "QUESTION\n"
    "How efficient are solar panels?\n\n"
    "ANSWER:\n"
    "Solar panels convert sunlight into electricity through the "
    "photovoltaic effect [A]. Commercial silicon panels typically reach "
    "between 15% and 22% efficiency [B].\n\n"
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
    include_example: bool = True,
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
        include_example: If True (default), inserts a short worked
            example demonstrating per-sentence citation before the
            real context. Improves citation reliability on smaller
            models; disable only if you've confirmed your model
            doesn't need it and want a shorter prompt.

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

    example_block = _EXAMPLE_BLOCK if include_example else ""

    text = (
        f"{effective_system_prompt}\n\n"
        f"{example_block}"
        "-------------------------\n"
        "CONTEXT\n\n"
        f"{context_block}\n\n"
        "-------------------------\n"
        "QUESTION\n"
        f"{question}\n\n"
        "-------------------------\n"
        "ANSWER:\n"
    )

    return RagPrompt(text=text, citations=citations)
