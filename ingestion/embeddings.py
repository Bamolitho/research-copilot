"""Turns chunk and query text into dense vectors using BGE-M3.

The embedding model is used twice in the pipeline, and must be the
*same* model both times so chunks and questions land in the same
vector space (see the ingestion README): once per chunk at indexing
time, and once per question at query time.

The actual model (sentence-transformers) is imported lazily, inside
`Embedder.model`, not at module load time. This keeps the module
importable -- and unit-testable with a fake encoder -- without
requiring the real (large) model download.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ingestion.chunker import Chunk

DEFAULT_MODEL_NAME = "BAAI/bge-m3"


class _EncoderModel(Protocol):
    """The subset of the sentence-transformers API this module relies on.

    Any object with a matching `encode` method satisfies this protocol,
    which is what lets tests inject a lightweight fake instead of a
    real, multi-gigabyte model.
    """

    def encode(
        self,
        sentences: str | list[str],
        normalize_embeddings: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk's text paired with its embedding vector.

    Attributes:
        chunk_id: The id of the chunk this embedding belongs to (see
            ingestion.chunker.Chunk).
        vector: The dense embedding, as a 1-D float32 array, normalized
            to unit length.
    """

    chunk_id: int
    vector: NDArray[np.float32]


class Embedder:
    """Encodes text into BGE-M3 dense embeddings.

    Args:
        model_name: HuggingFace model id to load if `model` is not
            provided. Defaults to BGE-M3.
        model: A pre-loaded, sentence-transformers-compatible encoder.
            Inject this in tests to avoid loading real weights. If
            omitted, `model_name` is downloaded and loaded lazily, on
            first use.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: _EncoderModel | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> _EncoderModel:
        """The underlying encoder, loaded on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed a batch of chunks, once, at indexing time.

        Args:
            chunks: Chunks produced by ingestion.chunker.Chunker.

        Returns:
            One EmbeddedChunk per input chunk, in the same order.
            Empty if `chunks` is empty (no model call is made).
        """
        if not chunks:
            return []

        vectors = self.model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [
            EmbeddedChunk(chunk_id=chunk.chunk_id, vector=np.asarray(vector, dtype=np.float32))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed a single user question, at query time.

        Args:
            query: The user's natural-language question.

        Returns:
            A 1-D float32 embedding vector, in the same space as the
            vectors from `embed_chunks` -- required for FAISS to
            compare them meaningfully.
        """
        vector = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32)
