"""Unit tests for ingestion.pdf_loader.

Uses a real PDF (tests/fixtures/sample_paper.pdf, the DrQA paper) as a
fixture rather than a synthetic file, so the tests exercise PyMuPDF
against a realistic multi-page academic PDF.
"""

from pathlib import Path

import pytest

from ingestion.pdf_loader import PageText, ParsedDocument, PDFLoader, PDFParsingError

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_paper.pdf"


@pytest.fixture
def loader() -> PDFLoader:
    return PDFLoader()


class TestPDFLoaderHappyPath:
    """Tests against a real, well-formed PDF."""

    def test_returns_a_parsed_document(self, loader: PDFLoader) -> None:
        result = loader.load(SAMPLE_PDF)
        assert isinstance(result, ParsedDocument)

    def test_source_path_is_preserved(self, loader: PDFLoader) -> None:
        result = loader.load(SAMPLE_PDF)
        assert result.source_path == SAMPLE_PDF

    def test_extracts_more_than_one_page(self, loader: PDFLoader) -> None:
        result = loader.load(SAMPLE_PDF)
        assert result.page_count > 1

    def test_page_numbers_are_1_indexed_and_sequential(self, loader: PDFLoader) -> None:
        result = loader.load(SAMPLE_PDF)
        page_numbers = [page.page_number for page in result.pages]
        assert page_numbers == list(range(1, result.page_count + 1))

    def test_pages_contain_page_text_instances(self, loader: PDFLoader) -> None:
        result = loader.load(SAMPLE_PDF)
        assert all(isinstance(page, PageText) for page in result.pages)

    def test_known_content_is_extracted(self, loader: PDFLoader) -> None:
        # This PDF is the DrQA paper; its own system name should appear
        # somewhere in the extracted text if extraction actually worked.
        result = loader.load(SAMPLE_PDF)
        assert "DrQA" in result.full_text

    def test_full_text_joins_all_pages(self, loader: PDFLoader) -> None:
        result = loader.load(SAMPLE_PDF)
        for page in result.pages:
            if page.text.strip():
                assert page.text in result.full_text


class TestPDFLoaderErrors:
    """Tests for the failure modes a caller must be able to rely on."""

    def test_missing_file_raises_file_not_found(self, loader: PDFLoader) -> None:
        missing_path = FIXTURES_DIR / "does_not_exist.pdf"
        with pytest.raises(FileNotFoundError):
            loader.load(missing_path)

    def test_corrupted_file_raises_pdf_parsing_error(
        self, loader: PDFLoader, tmp_path: Path
    ) -> None:
        fake_pdf = tmp_path / "not_really_a_pdf.pdf"
        fake_pdf.write_bytes(b"this is not a valid PDF file")

        with pytest.raises(PDFParsingError):
            loader.load(fake_pdf)

    def test_accepts_a_string_path_as_well_as_a_path_object(self, loader: PDFLoader) -> None:
        result = loader.load(str(SAMPLE_PDF))
        assert result.page_count > 0
