"""Unit tests for scripts.build_index.

Uses the real sample_paper.pdf fixture and a real Chunker/FaissVectorStore
(all local, no network), but a fake-encoder-backed Embedder to avoid
downloading the real BGE-M3 model.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from scripts.build_index import build_index

from ingestion.chunker import Chunker
from ingestion.embeddings import Embedder

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
DIM = 8


class FakeEncoder:
    """Deterministic stand-in for the real BGE-M3 model."""

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
        if isinstance(sentences, str):
            return np.ones(DIM, dtype=np.float32)
        return np.stack(
            [np.full(DIM, 0.1 * (i + 1), dtype=np.float32) for i in range(len(sentences))]
        )


def _fake_embedder() -> Embedder:
    return Embedder(model=FakeEncoder())


class TestBuildIndex:
    def test_indexes_a_directory_of_pdfs(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        shutil.copy(FIXTURE_PDF, papers_dir / "sample_paper.pdf")
        index_dir = tmp_path / "index"

        store = build_index(
            papers_dir=papers_dir,
            index_dir=index_dir,
            embedder=_fake_embedder(),
            chunker=Chunker(chunk_size=200, overlap=40),
            dimension=DIM,
        )

        assert store.size > 0
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "chunks.json").exists()

    def test_reloaded_index_matches_the_built_one(self, tmp_path: Path) -> None:
        from vector_db.faiss_store import FaissVectorStore

        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        shutil.copy(FIXTURE_PDF, papers_dir / "sample_paper.pdf")
        index_dir = tmp_path / "index"

        built = build_index(
            papers_dir=papers_dir,
            index_dir=index_dir,
            embedder=_fake_embedder(),
            chunker=Chunker(chunk_size=200, overlap=40),
            dimension=DIM,
        )
        reloaded = FaissVectorStore.load(index_dir)

        assert reloaded.size == built.size

    def test_empty_directory_produces_an_empty_but_valid_index(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        index_dir = tmp_path / "index"

        store = build_index(
            papers_dir=papers_dir,
            index_dir=index_dir,
            embedder=_fake_embedder(),
            chunker=Chunker(chunk_size=200, overlap=40),
            dimension=DIM,
        )

        assert store.size == 0
        assert (index_dir / "index.faiss").exists()

    def test_a_corrupted_pdf_is_skipped_without_crashing_the_run(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        shutil.copy(FIXTURE_PDF, papers_dir / "good_paper.pdf")
        (papers_dir / "corrupted.pdf").write_bytes(b"not a real pdf")
        index_dir = tmp_path / "index"

        store = build_index(
            papers_dir=papers_dir,
            index_dir=index_dir,
            embedder=_fake_embedder(),
            chunker=Chunker(chunk_size=200, overlap=40),
            dimension=DIM,
        )

        # the good PDF was still indexed despite the corrupted one existing
        assert store.size > 0


class TestCheckpointing:
    def test_save_is_called_after_every_n_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vector_db.faiss_store import FaissVectorStore

        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        for i in range(4):
            shutil.copy(FIXTURE_PDF, papers_dir / f"paper_{i}.pdf")
        index_dir = tmp_path / "index"

        save_calls = []
        original_save = FaissVectorStore.save

        def counting_save(self, directory):
            save_calls.append(self.size)
            return original_save(self, directory)

        monkeypatch.setattr(FaissVectorStore, "save", counting_save)

        build_index(
            papers_dir=papers_dir,
            index_dir=index_dir,
            embedder=_fake_embedder(),
            chunker=Chunker(chunk_size=200, overlap=40),
            dimension=DIM,
            save_every=2,
        )

        # checkpoint after file 2, checkpoint after file 4, plus the final save = 3 calls
        assert len(save_calls) == 3

    def test_save_every_zero_disables_checkpointing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vector_db.faiss_store import FaissVectorStore

        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        for i in range(4):
            shutil.copy(FIXTURE_PDF, papers_dir / f"paper_{i}.pdf")
        index_dir = tmp_path / "index"

        save_calls = []
        original_save = FaissVectorStore.save

        def counting_save(self, directory):
            save_calls.append(self.size)
            return original_save(self, directory)

        monkeypatch.setattr(FaissVectorStore, "save", counting_save)

        build_index(
            papers_dir=papers_dir,
            index_dir=index_dir,
            embedder=_fake_embedder(),
            chunker=Chunker(chunk_size=200, overlap=40),
            dimension=DIM,
            save_every=0,
        )

        # only the final save, no periodic checkpoints
        assert len(save_calls) == 1


class InterruptingEncoder:
    """Raises KeyboardInterrupt once a given number of files have been embedded."""

    def __init__(self, dim: int, interrupt_after: int) -> None:
        self.dim = dim
        self.interrupt_after = interrupt_after
        self.calls = 0

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
        self.calls += 1
        if self.calls > self.interrupt_after:
            raise KeyboardInterrupt
        return np.stack(
            [np.full(self.dim, 0.1 * (i + 1), dtype=np.float32) for i in range(len(sentences))]
        )


class TestInterruptHandling:
    def test_progress_is_saved_when_interrupted(self, tmp_path: Path) -> None:
        from ingestion.embeddings import Embedder
        from vector_db.faiss_store import FaissVectorStore

        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        for i in range(4):
            shutil.copy(FIXTURE_PDF, papers_dir / f"paper_{i}.pdf")
        index_dir = tmp_path / "index"

        # allow 2 files to embed successfully, then simulate Ctrl+C on the 3rd
        embedder = Embedder(model=InterruptingEncoder(dim=DIM, interrupt_after=2))

        with pytest.raises(KeyboardInterrupt):
            build_index(
                papers_dir=papers_dir,
                index_dir=index_dir,
                embedder=embedder,
                chunker=Chunker(chunk_size=200, overlap=40),
                dimension=DIM,
                save_every=0,  # only the interrupt-triggered save should happen
            )

        # the work done before the interruption must not be lost
        saved = FaissVectorStore.load(index_dir)
        assert saved.size > 0
