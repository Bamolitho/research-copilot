"""Unit tests for vector_db.faiss_store.

Unlike the downloader or embedder, FAISS itself is a local library with
no network or heavyweight model to mock -- these tests run it for real.
"""

from pathlib import Path

import numpy as np
import pytest

from ingestion.chunker import Chunk
from ingestion.embeddings import EmbeddedChunk
from vector_db.faiss_store import FaissVectorStore, SearchResult

SOURCE = Path("fake.pdf")
DIM = 4


def _chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source_path=SOURCE, page_start=1, page_end=1)


def _embedded(chunk_id: int, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(chunk_id=chunk_id, vector=np.array(vector, dtype=np.float32))


class TestAdd:
    def test_size_reflects_added_vectors(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        chunks = [_chunk(0, "a"), _chunk(1, "b")]
        embedded = [_embedded(0, [1, 0, 0, 0]), _embedded(1, [0, 1, 0, 0])]

        store.add(chunks, embedded)

        assert store.size == 2

    def test_adding_an_empty_batch_is_a_no_op(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        store.add([], [])
        assert store.size == 0

    def test_rejects_mismatched_lengths(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        with pytest.raises(ValueError):
            store.add([_chunk(0, "a")], [])

    def test_rejects_mismatched_chunk_ids(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        with pytest.raises(ValueError):
            store.add([_chunk(0, "a")], [_embedded(1, [1, 0, 0, 0])])


class TestSearch:
    def test_empty_index_returns_no_results(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=5)
        assert results == []

    def test_finds_the_closest_vector_first(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        chunks = [_chunk(0, "about cats"), _chunk(1, "about dogs"), _chunk(2, "about cars")]
        embedded = [
            _embedded(0, [1.0, 0.0, 0.0, 0.0]),
            _embedded(1, [0.9, 0.1, 0.0, 0.0]),
            _embedded(2, [0.0, 0.0, 0.0, 1.0]),
        ]
        store.add(chunks, embedded)

        results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)

        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.text == "about cats"  # exact match, highest score
        assert results[0].score >= results[1].score

    def test_k_larger_than_index_size_returns_all_vectors(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        store.add([_chunk(0, "only one")], [_embedded(0, [1, 0, 0, 0])])

        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=50)

        assert len(results) == 1


class TestSaveAndLoad:
    def test_round_trips_vectors_and_chunk_metadata(self, tmp_path: Path) -> None:
        store = FaissVectorStore(dimension=DIM)
        chunks = [_chunk(0, "first chunk"), _chunk(1, "second chunk")]
        embedded = [_embedded(0, [1, 0, 0, 0]), _embedded(1, [0, 1, 0, 0])]
        store.add(chunks, embedded)

        store.save(tmp_path)
        reloaded = FaissVectorStore.load(tmp_path)

        assert reloaded.size == 2
        assert reloaded.dimension == DIM
        results = reloaded.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
        assert results[0].chunk.text == "first chunk"
        assert results[0].chunk.source_path == SOURCE

    def test_load_raises_if_directory_has_no_saved_store(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FaissVectorStore.load(tmp_path)

    def test_save_creates_the_directory_if_missing(self, tmp_path: Path) -> None:
        store = FaissVectorStore(dimension=DIM)
        store.add([_chunk(0, "a")], [_embedded(0, [1, 0, 0, 0])])
        target = tmp_path / "nested" / "store"

        store.save(target)

        assert (target / "index.faiss").exists()
        assert (target / "chunks.json").exists()
