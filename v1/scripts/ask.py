"""End-to-end query pipeline: a question -> a grounded, cited answer.

Run as a script (see the CLI at the bottom), or import `ask` to call it
from other code -- e.g. a future FastAPI endpoint.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import load_settings

from ingestion.embeddings import Embedder
from llm.generate import GeneratedAnswer, Generator
from llm.prompt import build_prompt
from vector_db.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


def ask(
    question: str,
    store: FaissVectorStore,
    embedder: Embedder,
    generator: Generator,
    top_k: int = 5,
    disable_thinking: bool = False,
) -> GeneratedAnswer:
    """Answer a question, grounded in the given index.

    Args:
        question: The user's natural-language question.
        store: A loaded FaissVectorStore to search.
        embedder: Embedding model wrapper -- must be the same model
            used to build `store`, or the similarity scores are meaningless.
        generator: LLM client used to generate the final answer.
        top_k: Number of chunks to retrieve as context.
        disable_thinking: Passed through to llm.prompt.build_prompt --
            turns off Qwen3's internal reasoning trace, a pure latency
            cost for a context-grounded answer. Harmless no-op for
            other models.

    Returns:
        The generated answer, with citations resolvable back to chunks.

    Raises:
        ValueError: If `store` has nothing to retrieve (e.g. it's
            empty) -- there is nothing to ground an answer in.
    """
    query_vector = embedder.embed_query(question)
    results = store.search(query_vector, k=top_k)
    if not results:
        raise ValueError("No indexed documents to search -- build the index first.")

    prompt = build_prompt(question, results, disable_thinking=disable_thinking)
    return generator.generate(prompt)


def _run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question, answered from the indexed papers."
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Overrides INDEX_DIR")
    parser.add_argument("--top-k", type=int, default=None, help="Overrides TOP_K")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()

    store = FaissVectorStore.load(args.index_dir or settings.index_dir)
    embedder = Embedder(model_name=settings.embedding_model)
    generator = Generator(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )

    answer = ask(
        args.question,
        store=store,
        embedder=embedder,
        generator=generator,
        top_k=args.top_k or settings.top_k,
        disable_thinking=settings.disable_llm_thinking,
    )

    print(answer.text)
    print("\nSources:")
    for number, result in answer.citations.items():
        print(
            f"  [{number}] {result.chunk.source_path.name} "
            f"(page {result.chunk.page_start}), score={result.score:.3f}"
        )


if __name__ == "__main__":
    _run_cli()
