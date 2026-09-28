"""Unit tests for evaluation.retrieval_eval.

Uses a real, small FaissVectorStore (local, no network) and a
fake-encoder-backed Embedder -- no real BGE-M3 model is downloaded.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.retrieval_eval import GoldQuestion, evaluate, load_gold_set
from ingestion.chunker import Chunk
from ingestion.embeddings import EmbeddedChunk, Embedder
from vector_db.faiss_store import FaissVectorStore

SOURCE_A = "a.pdf"
SOURCE_B = "b.pdf"
DIM = 4


def _chunk(chunk_id: int, source: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source_path=Path(source), page_start=1, page_end=1)


def _populated_store() -> FaissVectorStore:
    """A store with chunks from two documents, both using chunk_id 0 and 1.

    Deliberately mirrors the real corpus: chunk_id resets per document,
    so (source_path, chunk_id) is the only safe key -- this fixture
    would catch a regression back to using chunk_id alone.
    """
    store = FaissVectorStore(dimension=DIM)
    chunks = [
        _chunk(0, SOURCE_A, "about false positives in IDS"),
        _chunk(1, SOURCE_A, "about unrelated cooking recipes"),
        _chunk(0, SOURCE_B, "about motion tracking in AR"),
        _chunk(1, SOURCE_B, "about embedded ECU memory limits"),
    ]
    embedded = [
        EmbeddedChunk(chunk_id=0, vector=np.array([1, 0, 0, 0], dtype=np.float32)),
        EmbeddedChunk(chunk_id=1, vector=np.array([0, 1, 0, 0], dtype=np.float32)),
        EmbeddedChunk(chunk_id=0, vector=np.array([0, 0, 1, 0], dtype=np.float32)),
        EmbeddedChunk(chunk_id=1, vector=np.array([0, 0, 0, 1], dtype=np.float32)),
    ]
    store.add(chunks, embedded)
    return store


def _embedder_returning(vector: list[float]) -> Embedder:
    from unittest.mock import MagicMock

    fake_model = MagicMock()
    fake_model.encode.return_value = np.array(vector, dtype=np.float32)
    return Embedder(model=fake_model)


class TestLoadGoldSet:
    def test_parses_a_valid_file(self, tmp_path: Path) -> None:
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                [
                    {
                        "question": "a question",
                        "relevant": [{"source_path": "a.pdf", "chunk_id": 0}],
                    }
                ]
            )
        )

        questions = load_gold_set(gold_set_path)

        assert len(questions) == 1
        assert questions[0].question == "a question"
        assert questions[0].relevant == {("a.pdf", 0)}

    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gold_set(tmp_path / "does_not_exist.json")

    def test_multiple_relevant_chunks_per_question(self, tmp_path: Path) -> None:
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                [
                    {
                        "question": "a question",
                        "relevant": [
                            {"source_path": "a.pdf", "chunk_id": 0},
                            {"source_path": "b.pdf", "chunk_id": 0},
                        ],
                    }
                ]
            )
        )

        questions = load_gold_set(gold_set_path)

        assert questions[0].relevant == {("a.pdf", 0), ("b.pdf", 0)}


class TestEvaluate:
    def test_correctly_matches_by_source_path_and_chunk_id_pair(self) -> None:
        # Both documents have a chunk_id=0 -- if the code compared by
        # chunk_id alone, this would incorrectly match SOURCE_B's chunk 0
        # against a gold answer that only wants SOURCE_A's chunk 0.
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])  # matches (SOURCE_A, 0) exactly
        gold_questions = [GoldQuestion(question="q", relevant={(SOURCE_A, 0)})]

        results = evaluate(gold_questions, store, embedder, k=1)

        assert results[0].precision_at_k == 1.0
        assert results[0].recall_at_k == 1.0
        assert results[0].reciprocal_rank == 1.0

    def test_a_wrong_document_with_the_same_chunk_id_is_not_a_false_hit(self) -> None:
        store = _populated_store()
        # this vector matches (SOURCE_B, 0) best, not (SOURCE_A, 0)
        embedder = _embedder_returning([0, 0, 1, 0])
        gold_questions = [GoldQuestion(question="q", relevant={(SOURCE_A, 0)})]

        results = evaluate(gold_questions, store, embedder, k=1)

        assert results[0].precision_at_k == 0.0
        assert results[0].reciprocal_rank == 0.0

    def test_returns_one_result_per_gold_question_in_order(self) -> None:
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])
        gold_questions = [
            GoldQuestion(question="first", relevant={(SOURCE_A, 0)}),
            GoldQuestion(question="second", relevant={(SOURCE_A, 1)}),
        ]

        results = evaluate(gold_questions, store, embedder, k=2)

        assert [r.question for r in results] == ["first", "second"]

    def test_empty_gold_set_returns_empty_results(self) -> None:
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])

        assert evaluate([], store, embedder) == []

    def test_result_carries_relevant_set_and_retrieved_chunks(self) -> None:
        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])
        gold_questions = [GoldQuestion(question="q", relevant={(SOURCE_A, 0)})]

        results = evaluate(gold_questions, store, embedder, k=2)

        assert results[0].relevant == {(SOURCE_A, 0)}
        assert len(results[0].retrieved) == 2
        assert results[0].retrieved[0].chunk.text == "about false positives in IDS"


class TestPrintReportVerbose:
    def test_verbose_shows_expected_and_retrieved_with_hit_marker(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from evaluation.retrieval_eval import print_report

        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])
        gold_questions = [GoldQuestion(question="a question", relevant={(SOURCE_A, 0)})]
        results = evaluate(gold_questions, store, embedder, k=2)

        print_report(results, k=2, verbose=True)

        output = capsys.readouterr().out
        assert "expected:" in output
        assert "retrieved:" in output
        assert "\u2713" in output  # the correct chunk is marked as a hit
        assert "about false positives in IDS" in output

    def test_non_verbose_does_not_print_chunk_detail(self, capsys: pytest.CaptureFixture) -> None:
        from evaluation.retrieval_eval import print_report

        store = _populated_store()
        embedder = _embedder_returning([1, 0, 0, 0])
        gold_questions = [GoldQuestion(question="a question", relevant={(SOURCE_A, 0)})]
        results = evaluate(gold_questions, store, embedder, k=2)

        print_report(results, k=2, verbose=False)

        output = capsys.readouterr().out
        assert "expected:" not in output
        assert "retrieved:" not in output
