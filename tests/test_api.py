"""Unit tests for api.main.

Uses create_app(fake_state) directly -- never get_app() -- so none of
these tests load a real index, embedding model, or LLM server.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import requests
from fastapi.testclient import TestClient

from api.main import AppState, create_app
from ingestion.chunker import Chunk
from ingestion.embeddings import EmbeddedChunk, Embedder
from llm.generate import Generator
from vector_db.faiss_store import FaissVectorStore

SOURCE = Path("data/papers/example.pdf")
DIM = 4


def _populated_store() -> FaissVectorStore:
    store = FaissVectorStore(dimension=DIM)
    chunk = Chunk(
        chunk_id=0, text="RAG reduces hallucination.", source_path=SOURCE, page_start=3, page_end=3
    )
    embedded = EmbeddedChunk(chunk_id=0, vector=np.array([1, 0, 0, 0], dtype=np.float32))
    store.add([chunk], [embedded])
    return store


def _embedder_returning(vector: list[float]) -> Embedder:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array(vector, dtype=np.float32)
    return Embedder(model=fake_model)


def _generator_returning(content: str) -> Generator:
    session = MagicMock()
    session.post.return_value.json.return_value = {"choices": [{"message": {"content": content}}]}
    return Generator(base_url="http://fake:11434/v1", model="fake-model", session=session)


def _client(
    store=None, embedder=None, generator=None, top_k=5, disable_thinking=False
) -> TestClient:
    state = AppState(
        store=store or _populated_store(),
        embedder=embedder or _embedder_returning([1, 0, 0, 0]),
        generator=generator or _generator_returning("A grounded answer. [1]"),
        default_top_k=top_k,
        disable_llm_thinking=disable_thinking,
    )
    return TestClient(create_app(state))


class TestHealth:
    def test_reports_ok_and_chunk_count(self) -> None:
        client = _client(store=_populated_store())

        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["indexed_chunks"] == 1

    def test_reports_the_embedding_model_name(self) -> None:
        embedder = Embedder(model_name="BAAI/bge-m3", model=MagicMock())
        client = _client(embedder=embedder)

        response = client.get("/health")

        assert response.json()["embedding_model"] == "BAAI/bge-m3"


class TestAsk:
    def test_returns_answer_and_citations(self) -> None:
        client = _client()

        response = client.post("/ask", json={"question": "What does RAG reduce?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "A grounded answer. [1]"
        assert len(body["citations"]) == 1
        assert body["citations"][0]["source"] == "example.pdf"
        assert body["citations"][0]["page_start"] == 3

    def test_rejects_an_empty_question(self) -> None:
        client = _client()

        response = client.post("/ask", json={"question": ""})

        assert response.status_code == 422  # FastAPI/Pydantic validation error

    def test_missing_question_field_is_rejected(self) -> None:
        client = _client()

        response = client.post("/ask", json={})

        assert response.status_code == 422

    def test_top_k_override_is_honored(self) -> None:
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])
        generator = _generator_returning("answer")
        client = _client(store=store, embedder=embedder, generator=generator, top_k=5)

        response = client.post("/ask", json={"question": "a question", "top_k": 1})

        assert response.status_code == 200

    def test_rejects_a_non_positive_top_k(self) -> None:
        client = _client()

        response = client.post("/ask", json={"question": "a question", "top_k": 0})

        assert response.status_code == 422

    def test_empty_index_returns_503(self) -> None:
        empty_store = FaissVectorStore(dimension=DIM)
        client = _client(store=empty_store)

        response = client.post("/ask", json={"question": "a question"})

        assert response.status_code == 503

    def test_llm_backend_failure_returns_502(self) -> None:
        session = MagicMock()
        session.post.side_effect = requests.ConnectionError("connection refused")
        generator = Generator(base_url="http://fake:11434/v1", model="m", session=session)
        client = _client(generator=generator)

        response = client.post("/ask", json={"question": "a question"})

        assert response.status_code == 502
