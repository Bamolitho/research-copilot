"""Unit tests for ingestion.downloader.

All tests mock the HTTP layer: unit tests should never depend on the
real arXiv API being reachable or fast. A real end-to-end check belongs
in a separate, explicitly marked integration test.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.downloader import ArxivDownloader, ArxivPaper

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1704.00051v2</id>
    <title>Reading Wikipedia to Answer Open-Domain Questions</title>
    <published>2017-04-01T00:00:00Z</published>
    <author><name>Danqi Chen</name></author>
    <author><name>Adam Fisch</name></author>
  </entry>
</feed>
"""


@pytest.fixture
def mock_session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def downloader(tmp_path: Path, mock_session: MagicMock) -> ArxivDownloader:
    return ArxivDownloader(download_dir=tmp_path, session=mock_session)


class TestSearch:
    def test_parses_papers_from_the_feed(
        self, downloader: ArxivDownloader, mock_session: MagicMock
    ) -> None:
        mock_response = MagicMock(text=SAMPLE_FEED)
        mock_session.get.return_value = mock_response

        papers = downloader.search("open domain question answering")

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, ArxivPaper)
        assert paper.arxiv_id == "1704.00051v2"
        assert paper.title == "Reading Wikipedia to Answer Open-Domain Questions"
        assert paper.authors == ["Danqi Chen", "Adam Fisch"]
        assert paper.pdf_url == "https://arxiv.org/pdf/1704.00051v2.pdf"

    def test_raises_on_http_error(
        self, downloader: ArxivDownloader, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value.raise_for_status.side_effect = RuntimeError("HTTP 500")
        with pytest.raises(RuntimeError):
            downloader.search("anything")

    def test_old_style_id_keeps_its_category_prefix(
        self, downloader: ArxivDownloader, mock_session: MagicMock
    ) -> None:
        # Regression test: pre-2007 arXiv ids look like "hep-ex/0307069v1".
        # Dropping the "hep-ex/" prefix builds a PDF URL that 404s.
        old_style_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/hep-ex/0307069v1</id>
    <title>Observation of Atmospheric Neutrino Oscillations in Soudan 2</title>
    <published>2003-07-01T00:00:00Z</published>
    <author><name>Someone</name></author>
  </entry>
</feed>
"""
        mock_session.get.return_value = MagicMock(text=old_style_feed)

        papers = downloader.search("neutrino oscillations")

        assert papers[0].arxiv_id == "hep-ex/0307069v1"
        assert papers[0].pdf_url == "https://arxiv.org/pdf/hep-ex/0307069v1.pdf"

    def test_plain_query_is_auto_prefixed_with_all(
        self, downloader: ArxivDownloader, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = MagicMock(text=SAMPLE_FEED)

        downloader.search("attention mechanism")

        sent_params = mock_session.get.call_args.kwargs["params"]
        assert sent_params["search_query"] == "all:attention mechanism"

    def test_already_qualified_query_is_not_double_prefixed(
        self, downloader: ArxivDownloader, mock_session: MagicMock
    ) -> None:
        # Regression test: "all:RAG AND all:attention" must NOT become
        # "all:all:RAG AND all:attention" (arXiv returns 400 Bad Request
        # for that malformed query).
        mock_session.get.return_value = MagicMock(text=SAMPLE_FEED)

        downloader.search("all:RAG AND all:attention")

        sent_params = mock_session.get.call_args.kwargs["params"]
        assert sent_params["search_query"] == "all:RAG AND all:attention"


class TestDownload:
    def _make_paper(self) -> ArxivPaper:
        return ArxivPaper(
            arxiv_id="1704.00051v2",
            title="Reading Wikipedia to Answer Open-Domain Questions",
            authors=["Danqi Chen"],
            published="2017-04-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/1704.00051v2.pdf",
        )

    def test_downloads_and_writes_the_file(
        self, downloader: ArxivDownloader, mock_session: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ingestion.downloader.time.sleep", lambda _seconds: None)
        mock_session.get.return_value = MagicMock(content=b"%PDF-1.4 fake pdf bytes")

        destination = downloader.download(self._make_paper())

        assert destination.exists()
        assert destination.read_bytes() == b"%PDF-1.4 fake pdf bytes"

    def test_old_style_id_does_not_create_a_subdirectory(
        self, downloader: ArxivDownloader, mock_session: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ingestion.downloader.time.sleep", lambda _seconds: None)
        mock_session.get.return_value = MagicMock(content=b"%PDF-1.4 fake pdf bytes")
        old_style_paper = ArxivPaper(
            arxiv_id="hep-ex/0307069v1",
            title="Observation of Atmospheric Neutrino Oscillations in Soudan 2",
            authors=["Someone"],
            published="2003-07-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/hep-ex/0307069v1.pdf",
        )

        destination = downloader.download(old_style_paper)

        assert destination.parent == downloader.download_dir
        assert destination.name == "hep-ex_0307069v1.pdf"

    def test_skips_download_if_file_already_exists(
        self, downloader: ArxivDownloader, mock_session: MagicMock
    ) -> None:
        paper = self._make_paper()
        existing = downloader.download_dir / f"{paper.arxiv_id}.pdf"
        existing.write_bytes(b"already here")

        destination = downloader.download(paper)

        mock_session.get.assert_not_called()
        assert destination.read_bytes() == b"already here"

    def test_overwrite_true_forces_a_fresh_download(
        self, downloader: ArxivDownloader, mock_session: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("ingestion.downloader.time.sleep", lambda _seconds: None)
        paper = self._make_paper()
        existing = downloader.download_dir / f"{paper.arxiv_id}.pdf"
        existing.write_bytes(b"stale content")
        mock_session.get.return_value = MagicMock(content=b"fresh content")

        destination = downloader.download(paper, overwrite=True)

        assert destination.read_bytes() == b"fresh content"


class TestCli:
    def test_requires_a_query_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ingestion.downloader import _run_cli

        monkeypatch.setattr("sys.argv", ["downloader.py"])  # no --query
        with pytest.raises(SystemExit):
            _run_cli()

    def test_searches_and_downloads_with_a_single_query(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from ingestion import downloader as downloader_module

        fake_paper = ArxivPaper(
            arxiv_id="1234.5678",
            title="A Fake Paper",
            authors=["Someone"],
            published="2024-01-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
        )
        monkeypatch.setattr(
            downloader_module.ArxivDownloader, "search", lambda self, q, max_results: [fake_paper]
        )
        monkeypatch.setattr(
            downloader_module.ArxivDownloader,
            "download",
            lambda self, paper, overwrite=False: tmp_path / "1234.5678.pdf",
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "downloader.py",
                "--query",
                "retrieval augmented generation",
                "--download-dir",
                str(tmp_path),
            ],
        )

        downloader_module._run_cli()

        output = capsys.readouterr().out
        assert "Found 1 paper(s)" in output
        assert "1234.5678" in output

    def test_multiple_topics_are_deduplicated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from ingestion import downloader as downloader_module

        shared_paper = ArxivPaper(
            arxiv_id="1111.1111",
            title="Attention Is All You Need",
            authors=["Someone"],
            published="2017-06-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/1111.1111.pdf",
        )
        rag_only_paper = ArxivPaper(
            arxiv_id="2222.2222",
            title="Retrieval-Augmented Generation",
            authors=["Someone Else"],
            published="2020-05-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/2222.2222.pdf",
        )

        # "rag" returns both papers, "attention mechanism" returns the
        # shared one again -- it must only be downloaded once.
        search_results = {
            "rag": [shared_paper, rag_only_paper],
            "attention mechanism": [shared_paper],
        }
        downloaded: list[str] = []

        def fake_search(self: object, query: str, max_results: int) -> list[ArxivPaper]:
            return search_results[query]

        def fake_download(self: object, paper: ArxivPaper, overwrite: bool = False) -> Path:
            downloaded.append(paper.arxiv_id)
            return tmp_path / f"{paper.arxiv_id}.pdf"

        monkeypatch.setattr(downloader_module.ArxivDownloader, "search", fake_search)
        monkeypatch.setattr(downloader_module.ArxivDownloader, "download", fake_download)
        monkeypatch.setattr(
            "sys.argv",
            [
                "downloader.py",
                "--query",
                "rag",
                "--query",
                "attention mechanism",
                "--download-dir",
                str(tmp_path),
            ],
        )

        downloader_module._run_cli()

        assert sorted(downloaded) == ["1111.1111", "2222.2222"]
        output = capsys.readouterr().out
        assert "2 unique paper(s) to download across 2 topic(s)" in output
