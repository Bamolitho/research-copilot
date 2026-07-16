"""Splits parsed documents into overlapping, page-aware chunks.

Chunking is done by word count rather than exact model tokens, to keep
this module free of a tokenizer dependency. Word count is a reasonable
proxy for chunk size (roughly 0.75-1.3 tokens per English word,
depending on the tokenizer) -- if a project needs exact token budgets,
chunk by the embedding model's own tokenizer instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ingestion.pdf_loader import ParsedDocument


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document's text, sized for retrieval.

    Attributes:
        chunk_id: 0-indexed position of this chunk within its document.
        text: The chunk's text content.
        source_path: Path to the PDF this chunk was extracted from.
        page_start: The first page number this chunk's text appears on.
        page_end: The last page number this chunk's text appears on.
            Equal to page_start unless the chunk straddles a page break.
    """

    chunk_id: int
    text: str
    source_path: Path
    page_start: int
    page_end: int


@dataclass(frozen=True)
class _WordOnPage:
    """Internal: one word tagged with the page it came from."""

    word: str
    page_number: int


class Chunker:
    """Splits a ParsedDocument into overlapping, fixed-size word chunks.

    Args:
        chunk_size: Number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Raises:
        ValueError: If chunk_size is not positive, if overlap is
            negative, or if overlap is not strictly smaller than
            chunk_size (otherwise chunking would never advance).
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must not be negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into overlapping chunks.

        Args:
            document: The parsed document to split.

        Returns:
            A list of Chunk, in reading order. Empty if the document
            contains no words at all. The last chunk is not padded and
            may be shorter than chunk_size.
        """
        words = self._words_with_page_numbers(document)
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []

        for chunk_id, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=" ".join(w.word for w in window),
                    source_path=document.source_path,
                    page_start=window[0].page_number,
                    page_end=window[-1].page_number,
                )
            )
            if start + self.chunk_size >= len(words):
                break  # the last window already reached the end; stop

        return chunks

    @staticmethod
    def _words_with_page_numbers(document: ParsedDocument) -> list[_WordOnPage]:
        """Flatten a document's pages into a single page-tagged word list."""
        return [
            _WordOnPage(word=word, page_number=page.page_number)
            for page in document.pages
            for word in page.text.split()
        ]
