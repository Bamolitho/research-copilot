"""Stores chunk embeddings in a FAISS index and maps results back to text.

FAISS itself only ever deals in vectors and integer positions -- it has
no idea what a "chunk" or a "page" is. This module pairs a FAISS index
with a parallel, position-aligned list of Chunk metadata, so a search
can return real, citable text instead of bare vector ids.

Named `faiss_store.py`, not `faiss.py`: a module in this package that
imports the real `faiss` package must not share its name, or Python can
resolve `import faiss` back to this very file instead of the installed
library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from ingestion.chunker import Chunk
from ingestion.embeddings import EmbeddedChunk

_INDEX_FILENAME = "index.faiss"
_CHUNKS_FILENAME = "chunks.json"


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk, with its similarity score.

    Attributes:
        chunk: The retrieved chunk (text, source path, page range).
        score: Inner-product similarity to the query. Since embeddings
            are normalized (see ingestion.embeddings), this is
            equivalent to cosine similarity, in [-1, 1].
    """

    chunk: Chunk
    score: float


class FaissVectorStore:
    """A FAISS index paired with the chunk metadata it was built from.

    Args:
        dimension: Size of the embedding vectors this store will hold
            (1024 for BGE-M3). All vectors added later must match it.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks_by_position: list[Chunk] = []

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    def add(self, chunks: Sequence[Chunk], embedded_chunks: Sequence[EmbeddedChunk]) -> None:
        """Add a batch of chunks and their embeddings to the index.

        Args:
            chunks: The chunks being indexed.
            embedded_chunks: Their embeddings, in the same order, as
                produced by ingestion.embeddings.Embedder.embed_chunks.

        Raises:
            ValueError: If `chunks` and `embedded_chunks` have
                different lengths, or their chunk_ids don't line up
                pairwise -- both indicate the two lists were built
                from different data and must not be indexed together.
        """
        if len(chunks) != len(embedded_chunks):
            raise ValueError(
                f"chunks ({len(chunks)}) and embedded_chunks ({len(embedded_chunks)}) "
                "must have the same length"
            )
        for chunk, embedded in zip(chunks, embedded_chunks, strict=True):
            if chunk.chunk_id != embedded.chunk_id:
                raise ValueError(
                    f"chunk_id mismatch: chunk {chunk.chunk_id} paired with "
                    f"embedding {embedded.chunk_id}"
                )
        if not chunks:
            return

        vectors = np.stack([ec.vector for ec in embedded_chunks]).astype(np.float32)
        self._index.add(vectors)
        self._chunks_by_position.extend(chunks)

    def search(self, query_vector: NDArray[np.float32], k: int = 5) -> list[SearchResult]:
        """Find the k chunks whose embeddings are closest to a query vector.

        Args:
            query_vector: A single embedding, from
                ingestion.embeddings.Embedder.embed_query. Must have
                been produced by the same embedding model used to
                build this index.
            k: Maximum number of results to return.

        Returns:
            Up to k SearchResult, best match first. Fewer than k if the
            index holds fewer than k vectors. Empty if the index is empty.
        """
        if self.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, positions = self._index.search(query, min(k, self.size))

        return [
            SearchResult(chunk=self._chunks_by_position[position], score=float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position != -1
        ]

    def save(self, directory: Path | str) -> None:
        """Persist the index and chunk metadata to a directory.

        Args:
            directory: Directory to write into. Created if missing.
                Existing contents of a matching store are overwritten.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / _INDEX_FILENAME))

        records = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_path": str(chunk.source_path),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
            for chunk in self._chunks_by_position
        ]
        (directory / _CHUNKS_FILENAME).write_text(json.dumps(records, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, directory: Path | str) -> FaissVectorStore:
        """Load a previously saved index and its chunk metadata.

        Args:
            directory: Directory previously written by `save`.

        Returns:
            A FaissVectorStore ready to search, with no re-embedding.

        Raises:
            FileNotFoundError: If the directory doesn't contain a
                previously saved store.
        """
        directory = Path(directory)
        index_path = directory / _INDEX_FILENAME
        chunks_path = directory / _CHUNKS_FILENAME
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(f"No FAISS store found in '{directory}'")

        index = faiss.read_index(str(index_path))
        records = json.loads(chunks_path.read_text())

        store = cls(dimension=index.d)
        store._index = index
        store._chunks_by_position = [
            Chunk(
                chunk_id=record["chunk_id"],
                text=record["text"],
                source_path=Path(record["source_path"]),
                page_start=record["page_start"],
                page_end=record["page_end"],
            )
            for record in records
        ]
        return store
