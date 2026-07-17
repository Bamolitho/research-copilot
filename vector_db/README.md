# `vector_db/`

Stores chunk embeddings for fast similarity search, and maps search results back to real, citable text.

## What's here

| File | Purpose |
|---|---|
| `faiss_store.py` | Wraps a FAISS index together with the chunk metadata it was built from. Library only, no CLI. |
| `__init__.py` | Makes this folder an importable package (`vector_db.faiss_store`). |

Corresponding tests live in `tests/test_faiss_store.py`, not in this folder — see [Running the tests](#running-the-tests).

> **Naming note:** this file is `faiss_store.py`, not `faiss.py`. A module that does `import faiss` must not share that name, or Python can resolve the import back to this file instead of the real library.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- No network access needed — FAISS is a local library, nothing here calls out to the internet

Install dependencies from the repo root:

```bash
uv sync
```

## Usage

```python
from ingestion.chunker import Chunker
from ingestion.embeddings import Embedder
from ingestion.pdf_loader import PDFLoader
from vector_db.faiss_store import FaissVectorStore

# Indexing (once per document)
document = PDFLoader().load("data/papers/2005.11401v4.pdf")
chunks = Chunker().split(document)

embedder = Embedder()
embedded_chunks = embedder.embed_chunks(chunks)

store = FaissVectorStore(dimension=1024)   # 1024 = BGE-M3's embedding size
store.add(chunks, embedded_chunks)
store.save("data/index")                   # writes index.faiss + chunks.json

# Querying (per question, no re-embedding of the corpus)
store = FaissVectorStore.load("data/index")
query_vector = embedder.embed_query("What are the challenges of ML-based IDS?")
results = store.search(query_vector, k=5)

for result in results:
    print(result.score, result.chunk.source_path, result.chunk.page_start, result.chunk.text[:80])
```

- `dimension` must match the embedding model's output size (1024 for BGE-M3) — it's fixed at construction, not inferred, so a mismatch fails immediately on `add()` rather than silently corrupting the index.
- `add()` validates that `chunks` and `embedded_chunks` line up (same length, same `chunk_id`s in order) before touching the index — building an index from two lists that don't actually correspond to each other is worse than crashing.
- `search()` returns fewer than `k` results if the index holds fewer than `k` vectors, and an empty list if the index is empty — it never raises just because there isn't enough data yet.

## Running the tests

From the repo root:

```bash
uv run pytest tests/test_faiss_store.py -v
```

These tests use FAISS for real (it's a local library, nothing to mock) and fake, hand-written embedding vectors — no dependency on the real BGE-M3 model.

## Linting

```bash
uv run ruff check vector_db/ tests/
```

## Design notes & gotchas

- **Scores are inner products, not raw cosine similarity by name** — but since `ingestion.embeddings` normalizes every vector to unit length before storing it, inner product *is* cosine similarity here. If you ever add vectors from a source that isn't normalized, scores stop being comparable to the rest of the corpus.
- **FAISS positions are not `chunk_id`s.** Internally, FAISS only knows sequential insertion positions (0, 1, 2, ...); `FaissVectorStore` keeps a parallel `_chunks_by_position` list to translate a result back into an actual `Chunk`. Don't call the private FAISS index directly and expect its ids to mean anything on their own.
- **One store, one embedding model.** Querying an index with a vector from a different embedding model than the one used to build it will run without error and return meaningless results — nothing here can detect that mistake, since a raw vector carries no record of which model produced it.
- **`save()` overwrites** `index.faiss` and `chunks.json` in the target directory without warning. Add versioning on top (e.g. a dated subdirectory) if you need rollback.

## Status

- [x] FAISS storage & search (`faiss_store.py`)
- [ ] Hybrid search (BM25 / OpenSearch) — later
- [ ] Reranking — later
