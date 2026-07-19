"""Pure retrieval evaluation metrics: no I/O, no FAISS, no file access.

Every function here takes plain Python data (a ranked list of chunk
keys, a set of relevant chunk keys) and returns a float. Keeping this
file free of any dependency on vector_db or ingestion is what makes it
trivially unit-testable without a real index or model.
"""

from __future__ import annotations

from collections.abc import Sequence

# A chunk is identified by (source_path, chunk_id), not chunk_id alone:
# ingestion.chunker.Chunker numbers chunks per document, starting at 0
# each time, so chunk_id 0 exists in every document -- only the pair
# is unique across a whole corpus. See evaluation/gold_set.json.
ChunkKey = tuple[str, int]


def precision_at_k(retrieved: Sequence[ChunkKey], relevant: set[ChunkKey], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are actually relevant.

    Args:
        retrieved: Retrieved chunk keys, best match first.
        relevant: The ground-truth set of relevant chunk keys.
        k: How many of the top results to consider.

    Returns:
        A value in [0, 1]. 0.0 if `retrieved` is empty or k <= 0.
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for key in top_k if key in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[ChunkKey], relevant: set[ChunkKey], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top-k retrieved.

    Args:
        retrieved: Retrieved chunk keys, best match first.
        relevant: The ground-truth set of relevant chunk keys.
        k: How many of the top results to consider.

    Returns:
        A value in [0, 1]. 0.0 if `relevant` is empty (nothing to find,
        so a recall fraction is undefined rather than misleadingly 1.0).
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k] if k > 0 else []
    hits = sum(1 for key in top_k if key in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[ChunkKey], relevant: set[ChunkKey]) -> float:
    """1 / (rank of the first relevant chunk), or 0.0 if none was found.

    Args:
        retrieved: Retrieved chunk keys, best match first.
        relevant: The ground-truth set of relevant chunk keys.

    Returns:
        The reciprocal rank of the first hit, in (0, 1], or 0.0 if no
        relevant chunk appears anywhere in `retrieved`.
    """
    for rank, key in enumerate(retrieved, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean. Returns 0.0 for an empty sequence, not an error."""
    if not values:
        return 0.0
    return sum(values) / len(values)
