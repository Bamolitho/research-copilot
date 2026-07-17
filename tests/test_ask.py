"""Unit tests for scripts.ask.

Uses a real, small FaissVectorStore (local, no network), a
fake-encoder-backed Embedder, and a fully mocked Generator -- no real
model or LLM server is needed to run these.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scripts.ask import ask

from ingestion.chunker import Chunk
from ingestion.embeddings import EmbeddedChunk, Embedder
from llm.generate import Generator
from vector_db.faiss_store import FaissVectorStore

SOURCE = Path("fake.pdf")
DIM = 4


def _chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source_path=SOURCE, page_start=1, page_end=1)


def _populated_store() -> FaissVectorStore:
    store = FaissVectorStore(dimension=DIM)
    chunks = [
        _chunk(0, "about false positives in IDS"),
        _chunk(1, "about unrelated cooking recipes"),
    ]
    embedded = [
        EmbeddedChunk(chunk_id=0, vector=np.array([1, 0, 0, 0], dtype=np.float32)),
        EmbeddedChunk(chunk_id=1, vector=np.array([0, 0, 0, 1], dtype=np.float32)),
    ]
    store.add(chunks, embedded)
    return store


def _embedder_returning(vector: list[float]) -> Embedder:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array(vector, dtype=np.float32)
    return Embedder(model=fake_model)


def _mocked_generator() -> tuple[Generator, MagicMock]:
    session = MagicMock()
    session.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "False positives are a major challenge. [1]"}}]
    }
    return Generator(
        base_url="http://localhost:11434/v1", model="test-model", session=session
    ), session


class TestAsk:
    def test_returns_a_generated_answer_grounded_in_the_best_match(self) -> None:
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])  # matches chunk 0
        generator, _ = _mocked_generator()

        answer = ask("What causes false positives?", store, embedder, generator, top_k=1)

        assert answer.text == "False positives are a major challenge. [1]"
        assert answer.citations[1].chunk.text == "about false positives in IDS"

    def test_raises_if_the_index_is_empty(self) -> None:
        store = FaissVectorStore(dimension=DIM)
        embedder = _embedder_returning([1, 0, 0, 0])
        generator, _ = _mocked_generator()

        with pytest.raises(ValueError):
            ask("any question", store, embedder, generator)

    def test_top_k_limits_how_many_chunks_are_sent_as_context(self) -> None:
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])
        generator, session = _mocked_generator()

        ask("a question", store, embedder, generator, top_k=1)

        sent_prompt = session.post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "about false positives in IDS" in sent_prompt
        assert "about unrelated cooking recipes" not in sent_prompt
