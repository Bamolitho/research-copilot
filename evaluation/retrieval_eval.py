"""Evaluates the retriever against a hand-annotated gold set.

Loads the real FAISS index and embedder, runs every question in the
gold set through actual retrieval, and reports Precision@k, Recall@k,
and MRR -- per question and averaged over the whole set.

Run as a script (see the CLI at the bottom), or import `evaluate` to
call it from other code.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import load_settings

from evaluation.metrics import ChunkKey, mean, precision_at_k, recall_at_k, reciprocal_rank
from ingestion.embeddings import Embedder
from vector_db.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)

DEFAULT_GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"


@dataclass(frozen=True)
class GoldQuestion:
    """One annotated question from the gold set.

    Attributes:
        question: The natural-language question.
        relevant: The ground-truth set of (source_path, chunk_id) keys.
    """

    question: str
    relevant: set[ChunkKey]


@dataclass(frozen=True)
class QuestionResult:
    """Retrieval metrics for a single gold-set question.

    Attributes:
        question: The question this result is for.
        precision_at_k: Precision@k for this question.
        recall_at_k: Recall@k for this question.
        reciprocal_rank: Reciprocal rank for this question.
    """

    question: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def load_gold_set(path: Path | str = DEFAULT_GOLD_SET_PATH) -> list[GoldQuestion]:
    """Load and parse the gold set JSON file.

    Args:
        path: Path to a gold set JSON file (see evaluation/README.md
            for the expected format).

    Returns:
        One GoldQuestion per entry in the file.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No gold set found at '{path}'")

    raw = json.loads(path.read_text())
    return [
        GoldQuestion(
            question=entry["question"],
            relevant={(r["source_path"], r["chunk_id"]) for r in entry["relevant"]},
        )
        for entry in raw
    ]


def evaluate(
    gold_questions: list[GoldQuestion],
    store: FaissVectorStore,
    embedder: Embedder,
    k: int = 5,
) -> list[QuestionResult]:
    """Run every gold-set question through real retrieval and score it.

    Args:
        gold_questions: Annotated questions to evaluate against.
        store: A loaded FaissVectorStore to search.
        embedder: Embedding model wrapper -- must be the same model
            used to build `store`, or scores are meaningless.
        k: Number of chunks to retrieve per question.

    Returns:
        One QuestionResult per gold question, in the same order.
    """
    results = []
    for gold_question in gold_questions:
        query_vector = embedder.embed_query(gold_question.question)
        search_results = store.search(query_vector, k=k)
        retrieved: list[ChunkKey] = [
            (str(result.chunk.source_path), result.chunk.chunk_id) for result in search_results
        ]

        results.append(
            QuestionResult(
                question=gold_question.question,
                precision_at_k=precision_at_k(retrieved, gold_question.relevant, k),
                recall_at_k=recall_at_k(retrieved, gold_question.relevant, k),
                reciprocal_rank=reciprocal_rank(retrieved, gold_question.relevant),
            )
        )
    return results


def print_report(results: list[QuestionResult], k: int) -> None:
    """Print a per-question and aggregate report to stdout."""
    print(f"Retrieval evaluation (k={k}), {len(results)} question(s)\n")
    for result in results:
        print(
            f"  P@{k}={result.precision_at_k:.2f}  R@{k}={result.recall_at_k:.2f}  "
            f"RR={result.reciprocal_rank:.2f}  | {result.question}"
        )

    print("\n--- Averages ---")
    print(f"  Mean Precision@{k}: {mean([r.precision_at_k for r in results]):.3f}")
    print(f"  Mean Recall@{k}:    {mean([r.recall_at_k for r in results]):.3f}")
    print(f"  MRR:                {mean([r.reciprocal_rank for r in results]):.3f}")


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the retriever against the gold set.")
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET_PATH)
    parser.add_argument("--index-dir", type=Path, default=None, help="Overrides INDEX_DIR")
    parser.add_argument("--k", type=int, default=None, help="Overrides TOP_K")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    k = args.k or settings.top_k

    gold_questions = load_gold_set(args.gold_set)
    store = FaissVectorStore.load(args.index_dir or settings.index_dir)
    embedder = Embedder(model_name=settings.embedding_model)

    results = evaluate(gold_questions, store, embedder, k=k)
    print_report(results, k=k)


if __name__ == "__main__":
    _run_cli()
