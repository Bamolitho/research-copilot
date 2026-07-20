# `api/`

Exposes the `ask()` pipeline over HTTP: the index, embedder, and LLM client are loaded once at startup, not on every request.

## What's here

| File | Purpose |
|---|---|
| `main.py` | The FastAPI app: `create_app(state)` (testable factory) and `get_app()` (real production entry point). |
| `schemas.py` | Pydantic request/response models (`AskRequest`, `AskResponse`, `Citation`, `HealthResponse`). |
| `__init__.py` | Makes this folder an importable package. |

Corresponding tests live in `tests/test_api.py`.

## Requirements

Same as the rest of the project — `uv sync` from the repo root. Running the real server needs a built index (`data/index/`) and a reachable LLM endpoint (Ollama locally, by default).

## Usage

```bash
make serve
# or directly:
uv run uvicorn api.main:get_app --factory --reload
```

Then, from another terminal:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main challenges of retrieval augmented generation?"}'
```

Interactive API docs (Swagger UI, generated automatically by FastAPI from `schemas.py`) are at `http://127.0.0.1:8000/docs` once the server is running.

### `GET /health`

```json
{"status": "ok", "indexed_chunks": 2623, "embedding_model": "BAAI/bge-m3"}
```

### `POST /ask`

Request:
```json
{"question": "What are the main challenges of RAG?", "top_k": 5}
```
`top_k` is optional; omit it to use the server's configured default (`TOP_K` in `.env`).

Response:
```json
{
  "answer": "The main challenges include data preprocessing... [1][2]",
  "citations": [
    {"number": 1, "source": "2508.14066v1.pdf", "page_start": 6, "page_end": 6, "score": 0.70, "excerpt": "..."}
  ]
}
```

Error responses: `422` for an invalid request body (empty question, non-positive `top_k` — handled automatically by Pydantic validation), `503` if the index has no chunks to search, `502` if the LLM backend is unreachable or errors out.

## Running the tests

```bash
uv run pytest tests/test_api.py -v
```

Every test uses `create_app(fake_state)` directly, never `get_app()` — no real index, embedding model, or LLM server is ever loaded during tests. `AppState` is built entirely from fakes (a tiny real `FaissVectorStore` with hand-inserted vectors, a mocked encoder, a mocked HTTP session), the same dependency-injection pattern used throughout `ingestion/`, `llm/`, and `scripts/`.

## Linting

```bash
uv run ruff check api/ tests/
```

## Design notes & gotchas

- **`get_app()` is never called at import time.** Importing `api.main` (e.g. from a test) only defines functions and classes — it never loads a model or an index as a side effect. Only uvicorn's factory mode (`--factory`) calls `get_app()`, and only when the server actually starts. This is what makes `create_app(fake_state)` possible in tests without monkeypatching anything.
- **No endpoint builds or rebuilds the index.** `build_index` can take hours on a large corpus; that has no place behind a synchronous HTTP request. Indexing stays a `scripts.build_index` CLI/batch job, run separately, out of band from the API's lifecycle.
- **State is loaded once, shared across all requests.** This is the whole reason this folder exists instead of just using `scripts.ask` — the CLI reloads the embedding model and index on every single invocation, which is fine for one-off questions and wasteful for a server answering many.
- **Citations expose only the filename, not the full server-side path.** `result.chunk.source_path.name`, not the full `Path` — avoids leaking server filesystem structure into API responses.

## Status

- [x] `POST /ask`, `GET /health`
- [ ] `frontend/` (a UI on top of this API) — next
- [ ] Auth / rate limiting — not needed for a single-user local project yet
