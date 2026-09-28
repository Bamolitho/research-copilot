# `frontend/`

A Streamlit chat UI on top of the API — asks questions, shows the answer, shows the citations, tells you plainly when something's wrong.

## What's here

| File | Purpose |
|---|---|
| `api_client.py` | Pure HTTP client for the API (`get_health`, `ask_question`). No Streamlit dependency. |
| `app.py` | The Streamlit UI itself. Imports only from `api_client`, renders what it returns. |
| `__init__.py` | Makes this folder an importable package. |

Corresponding tests live in `tests/test_frontend.py` — and only test `api_client.py`, never `app.py` directly (see gotchas below).

## Requirements

Same as the rest of the project — `uv sync` from the repo root. The API (`make serve`) must be running for the UI to show anything but an error banner.

## Usage

Two terminals:

```bash
# Terminal 1
make serve

# Terminal 2
make ui
```

Opens at `http://localhost:8501`. Ask a question in the chat box; the answer streams in with an expandable citation per source underneath it.

Override the API location if it's not running on the default port:

```bash
API_BASE_URL=http://127.0.0.1:9000 uv run streamlit run frontend/app.py
```

## Running the tests

```bash
uv run pytest tests/test_frontend.py -v
```

All mocked — no real API, index, or LLM server needed.

## Linting

```bash
uv run ruff check frontend/ tests/test_frontend.py
```

## Design notes & gotchas

- **`api_client.py` has zero Streamlit imports, on purpose.** This is what makes it unit-testable without a UI runtime, and it's why the tests only ever import from `api_client`, never `app`. Streamlit apps are a script that reruns top-to-bottom on every interaction — testing that flow directly needs `streamlit.testing.v1.AppTest`, which is more brittle and wasn't worth it for a project this size when the logic that actually matters (talking to the API correctly) is already covered.
- **`get_health` swallows connection errors into `None`, `ask_question` doesn't.** A dead API on page load should show a calm sidebar status, not a crash — but a failed question, mid-conversation, needs its own visible error message, which only makes sense inside the chat flow itself, not baked into the client.
- **The exact on-screen visual layout still hasn't been confirmed** — `make ui` was never run start-to-finish in the development environment this was built in (a `pip install streamlit` failure specific to that session made it impossible). The `ModuleNotFoundError` above was a real bug caught and fixed from your actual run; the rendering itself (sidebar, chat bubbles, citation expanders) is correct per Streamlit's documented API but not yet visually confirmed against a live app.
- **`app.py` uses a try/except fallback to import `api_client`**, not a plain import. This resolves a real conflict, not a style choice: mypy analyzes `app.py` as part of the `frontend` package and only resolves the qualified `frontend.api_client`; Streamlit runs `app.py` as a direct script, which puts `frontend/` itself (not the repo root) on `sys.path`, so only the bare `api_client` resolves at runtime. The `# type: ignore[import-not-found,no-redef]` and `# noqa: F811` on the fallback line are deliberate and scoped to this one line — they suppress mypy and ruff both correctly flagging the fallback branch as unresolvable/redundant from their respective single-tool point of view, which it only appears to be because neither tool sees how the other one runs this file.

## Status

- [x] Chat UI (`app.py`)
- [x] Pure, tested API client (`api_client.py`)
- [ ] Streaming token-by-token responses (currently waits for the full answer) — later
- [ ] Multi-session / conversation history persistence — later
