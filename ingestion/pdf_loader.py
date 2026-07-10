"""PDF text extraction for the Research Copilot ingestion pipeline.

Wraps PyMuPDF (fitz) to turn a PDF file on disk into a structured,
page-aware text representation that the chunking stage can consume.

This module has one job: get clean text out of a PDF, one page at a
time. Chunking, cleaning, and embedding are handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


class PDFParsingError(Exception):
    """Raised when a PDF file exists but cannot be opened or read.

    Typical causes: a corrupted file, a file that is not actually a
    PDF despite its extension, or a PDF encrypted without a password.
    """


@dataclass(frozen=True)
class PageText:
    """The extracted text of a single PDF page.

    Attributes:
        page_number: 1-indexed page number, matching how a human would
            cite it (page 1 is the first page).
        text: The raw extracted text of that page. May be an empty
            string for blank pages or image-only pages with no OCR layer.
    """

    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """The full result of parsing one PDF file.

    Attributes:
        source_path: Path to the original PDF file on disk.
        pages: The extracted text of every page, in page order.
    """

    source_path: Path
    pages: list[PageText]

    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """The full document text, with pages joined by a blank line.

        This is a convenience for callers that want the whole document
        as one string (e.g. for a quick keyword search). The chunking
        stage should generally use `pages` directly, so it can keep
        track of which page each chunk came from.
        """
        return "\n\n".join(page.text for page in self.pages)


class PDFLoader:
    """Extracts text from PDF files using PyMuPDF.

    A single loader instance holds no per-file state and can safely be
    reused to load many files, e.g. in a loop over a directory of PDFs.
    """

    def load(self, path: Path | str) -> ParsedDocument:
        """Parse a single PDF file into a ParsedDocument.

        Args:
            path: Path to the PDF file on disk.

        Returns:
            A ParsedDocument containing the text of every page, in order.

        Raises:
            FileNotFoundError: If `path` does not point to an existing file.
            PDFParsingError: If the file exists but cannot be opened or
                read as a PDF, or if it contains zero pages.
        """
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"No such file: {pdf_path}")

        document = self._open(pdf_path)
        try:
            pages = [
                PageText(page_number=index + 1, text=page.get_text("text"))
                for index, page in enumerate(document)
            ]
        finally:
            document.close()

        if not pages:
            raise PDFParsingError(f"'{pdf_path}' contains no pages")

        return ParsedDocument(source_path=pdf_path, pages=pages)

    @staticmethod
    def _open(pdf_path: Path) -> fitz.Document:
        """Open a PDF file, wrapping any PyMuPDF failure in PDFParsingError."""
        try:
            return fitz.open(pdf_path)
        except Exception as exc:
            raise PDFParsingError(f"Could not open '{pdf_path}': {exc}") from exc
