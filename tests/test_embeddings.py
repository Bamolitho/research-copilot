"""Unit tests for ingestion.embeddings.

All tests use FakeEncoder, a lightweight stand-in for SentenceTransformer,
so none of them download or run the real (multi-gigabyte) BGE-M3 model.
A real end-to-end check belongs in a separate, explicitly marked
integration test, run in an environment with the model available.
"""

from pathlib import Path

import numpy as np

from ingestion.chunker import Chunk
from ingestion.embeddings import EmbeddedChunk, Embedder

SOURCE = Path("fake.pdf")


class FakeEncoder:
    """Records what it was asked to encode and returns predictable vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: str | list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        self.calls.append(
            {
                "sentences": sentences,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        if isinstance(sentences, str):
            return np.full(self.dim, 0.5, dtype=np.float32)
        return np.stack(
            [np.full(self.dim, 0.1 * (i + 1), dtype=np.float32) for i in range(len(sentences))]
        )


def _chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source_path=SOURCE, page_start=1, page_end=1)


class TestEmbedderConstruction:
    def test_injected_model_is_used_without_lazy_loading(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)

        assert embedder.model is fake  # no import/download was triggered


class TestEmbedChunks:
    def test_empty_input_makes_no_model_call(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)

        result = embedder.embed_chunks([])

        assert result == []
        assert fake.calls == []

    def test_returns_one_embedded_chunk_per_chunk_in_order(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)
        chunks = [_chunk(0, "first chunk"), _chunk(1, "second chunk")]

        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert [ec.chunk_id for ec in result] == [0, 1]
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)

    def test_sends_chunk_text_not_chunk_objects_to_the_encoder(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)
        chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]

        embedder.embed_chunks(chunks)

        assert fake.calls[0]["sentences"] == ["alpha", "beta"]

    def test_requests_normalized_embeddings(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)

        embedder.embed_chunks([_chunk(0, "alpha")])

        assert fake.calls[0]["normalize_embeddings"] is True

    def test_vectors_are_float32(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)

        result = embedder.embed_chunks([_chunk(0, "alpha")])

        assert result[0].vector.dtype == np.float32


class TestEmbedQuery:
    def test_returns_a_1d_float32_vector(self) -> None:
        fake = FakeEncoder(dim=8)
        embedder = Embedder(model=fake)

        vector = embedder.embed_query("what are the challenges of ML-based IDS?")

        assert vector.shape == (8,)
        assert vector.dtype == np.float32

    def test_sends_the_raw_query_string_to_the_encoder(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)

        embedder.embed_query("a question")

        assert fake.calls[0]["sentences"] == "a question"

    def test_requests_normalized_embeddings(self) -> None:
        fake = FakeEncoder()
        embedder = Embedder(model=fake)

        embedder.embed_query("a question")

        assert fake.calls[0]["normalize_embeddings"] is True
