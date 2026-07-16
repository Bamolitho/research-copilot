# `ingestion/`

Turns source papers into clean, page-aware text ready for chunking. Two independent pieces: acquiring PDFs (`downloader.py`) and extracting their text (`pdf_loader.py`).

## What's here

| File | Purpose |
|---|---|
| `downloader.py` | Searches arXiv and downloads matching papers as PDFs. Has a CLI. |
| `pdf_loader.py` | Extracts page-by-page text from a PDF file using PyMuPDF. Library only, no CLI. |
| `chunker.py` | Splits a `ParsedDocument` into overlapping, page-aware chunks. Library only, no CLI. |
| `__init__.py` | Makes this folder an importable package (`ingestion.downloader`, `ingestion.pdf_loader`, `ingestion.chunker`). |

Corresponding tests live in `tests/test_downloader.py`, `tests/test_pdf_loader.py`, and `tests/test_chunker.py`, not in this folder — see [Running the tests](#running-the-tests).

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management and running commands
- Internet access to `export.arxiv.org` and `arxiv.org` (only needed for `downloader.py`; `pdf_loader.py` works fully offline)

Install all dependencies (including dev tools like `pytest` and `ruff`) from the repo root:

```bash
uv sync
```

## Usage

### Downloading papers

```bash
uv run python3 -m ingestion.downloader \
  --query "retrieval augmented generation" \
  --query "attention mechanism" \
  --max-results 20 \
  --download-dir data/papers
```

- `--query` is **required** and **repeatable** — pass it once per topic to build a multi-topic corpus in one run. A paper matching several topics is only downloaded once.
- `--max-results` applies **per topic** (default: 20).
- `--download-dir` is where PDFs are saved (default: `data/papers`). Created automatically if it doesn't exist.
- Files already present locally are **not** re-downloaded (idempotent by design).

Programmatic use (e.g. from a pipeline script):

```python
from ingestion.downloader import ArxivDownloader

downloader = ArxivDownloader(download_dir="data/papers")
papers = downloader.search("retrieval augmented generation", max_results=20)
for paper in papers:
    downloader.download(paper)          # add overwrite=True to force a re-download
```

### Parsing PDFs

```python
from ingestion.pdf_loader import PDFLoader, PDFParsingError

loader = PDFLoader()
document = loader.load("data/papers/2005.11401v4.pdf")

print(document.page_count)              # number of pages
print(document.pages[0].text)           # text of page 1
print(document.full_text)               # all pages joined, for quick lookups
```

`PDFLoader.load()` raises `FileNotFoundError` if the path doesn't exist, and `PDFParsingError` if the file exists but can't be read as a PDF (corrupted, wrong format, encrypted). Always catch both when looping over a directory of downloaded papers — one bad file shouldn't crash the whole ingestion run.

### Chunking

```python
from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import Chunker

document = PDFLoader().load("data/papers/2005.11401v4.pdf")
chunks = Chunker(chunk_size=200, overlap=40).split(document)

for chunk in chunks:
    print(chunk.chunk_id, chunk.page_start, chunk.page_end, chunk.text[:80])
```

- Chunking is by **word count**, not exact model tokens, to keep this module dependency-free — treat `chunk_size` as a proxy for roughly `chunk_size × 1.0–1.3` tokens.
- `overlap` must be strictly smaller than `chunk_size`; the constructor raises `ValueError` otherwise.
- Each `Chunk` records `page_start`/`page_end`, since a chunk can straddle a page break — this is what lets the retrieval layer cite "page X" later.
- The last chunk of a document is **not padded** and may be shorter than `chunk_size`.

## Running the tests

From the repo root:

```bash
uv run pytest tests/ -v
```

Only this folder's tests:

```bash
uv run pytest tests/test_downloader.py tests/test_pdf_loader.py -v
```

All downloader tests mock the HTTP layer — none of them touch the real arXiv API, so they run offline and in CI without rate-limit concerns. `test_pdf_loader.py` runs against a real PDF fixture (`tests/fixtures/sample_paper.pdf`).

## Linting

```bash
uv run ruff check ingestion/ tests/
```

## Design notes & gotchas

- **Old-style arXiv IDs.** Papers from before 2007 use ids like `hep-ex/0307069v1` (a category prefix, then a slash). The downloader preserves this prefix when building the PDF URL and flattens it (`hep-ex_0307069v1.pdf`) for the local filename, so it never tries to create an unexpected subdirectory.
- **Rate limiting.** `download()` sleeps 3 seconds after each request, per arXiv's own usage guideline. Don't remove this when scripting bulk downloads.
- **Query syntax.** Each `--query` value is sent as an arXiv `all:` field search. To require several terms in the *same* paper rather than searching several topics, use arXiv's boolean syntax directly, e.g. `--query "all:RAG AND all:attention"`.
- **`pdf_loader.py` has no CLI on purpose.** It's a library consumed by the chunking stage; wrapping it in its own CLI would just duplicate what a short pipeline script will do once `chunker.py` exists.

## Status

- [x] PDF parsing (`pdf_loader.py`)
- [x] arXiv downloader (`downloader.py`)
- [x] Chunking (`chunker.py`)
- [ ] Embeddings (`embeddings.py`) — next
