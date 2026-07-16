"""Unit tests for ingestion.chunker.

The 50-word PASSAGE and its expected 3-chunk split are the same example
worked by hand in the course's exercise workbook (Exercise 2.6) -- kept
in sync on purpose, as a cross-check that the code matches the theory.
"""

from pathlib import Path

import pytest

from ingestion.chunker import Chunker
from ingestion.pdf_loader import PageText, ParsedDocument

SOURCE = Path("fake.pdf")

PASSAGE = (
    "Retrieval augmented generation grounds language model answers in "
    "retrieved passages instead of relying purely on parametric memory. "
    "The retriever finds relevant chunks from an indexed corpus using "
    "dense or sparse similarity search. The generator then reads those "
    "chunks alongside the question and produces a grounded, citable "
    "response for the user."
)


def _document(*page_texts: str) -> ParsedDocument:
    """Build a ParsedDocument from raw page strings, for tests only."""
    pages = [PageText(page_number=i + 1, text=text) for i, text in enumerate(page_texts)]
    return ParsedDocument(source_path=SOURCE, pages=pages)


class TestChunkerConstruction:
    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=0, overlap=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, overlap=-1)

    def test_rejects_overlap_not_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, overlap=10)


class TestSplit:
    def test_matches_the_hand_worked_workbook_example(self) -> None:
        # See workbook Exercise 2.6: 50 words, size 20, overlap 5 -> 3 chunks.
        chunker = Chunker(chunk_size=20, overlap=5)
        document = _document(PASSAGE)

        chunks = chunker.split(document)

        assert len(chunks) == 3
        assert chunks[0].text.split()[:3] == ["Retrieval", "augmented", "generation"]
        assert chunks[-1].text.split()[-3:] == ["for", "the", "user."]

    def test_consecutive_chunks_share_the_overlap(self) -> None:
        chunker = Chunker(chunk_size=20, overlap=5)
        document = _document(PASSAGE)

        chunks = chunker.split(document)

        assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]

    def test_last_chunk_is_not_padded(self) -> None:
        # 22 words, chunk_size=20, overlap=5 (step=15): second chunk has 7 words.
        words = " ".join(f"w{i}" for i in range(22))
        chunker = Chunker(chunk_size=20, overlap=5)

        chunks = chunker.split(_document(words))

        assert len(chunks[-1].text.split()) == 7

    def test_chunk_ids_are_sequential_from_zero(self) -> None:
        chunker = Chunker(chunk_size=20, overlap=5)
        chunks = chunker.split(_document(PASSAGE))

        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_empty_document_produces_no_chunks(self) -> None:
        chunker = Chunker()
        assert chunker.split(_document("", "   ")) == []

    def test_source_path_is_preserved(self) -> None:
        chunker = Chunker(chunk_size=20, overlap=5)
        chunks = chunker.split(_document(PASSAGE))

        assert all(c.source_path == SOURCE for c in chunks)


class TestPageTracking:
    def test_single_page_chunk_has_matching_start_and_end(self) -> None:
        chunker = Chunker(chunk_size=20, overlap=5)
        chunks = chunker.split(_document(PASSAGE))  # everything on page 1

        assert all(c.page_start == 1 and c.page_end == 1 for c in chunks)

    def test_chunk_spanning_two_pages_records_both(self) -> None:
        page_1_words = " ".join(f"a{i}" for i in range(15))
        page_2_words = " ".join(f"b{i}" for i in range(15))
        chunker = Chunker(chunk_size=20, overlap=5)

        chunks = chunker.split(_document(page_1_words, page_2_words))

        # First chunk = words 0-19: 15 words from page 1 + 5 from page 2.
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2
