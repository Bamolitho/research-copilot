# `scripts/`

The walking skeleton: wires `ingestion/`, `vector_db/`, and `llm/` together into two runnable commands — build the index, then ask it a question.

## What's here

| File | Purpose |
|---|---|
| `build_index.py` | Parses, chunks, embeds, and indexes every PDF in a directory. Has a CLI. |
| `ask.py` | Embeds a question, searches the index, and generates a cited answer. Has a CLI. |
| `__init__.py` | Makes this folder an importable package (`scripts.build_index`, `scripts.ask`). |

Corresponding tests live in `tests/test_build_index.py` and `tests/test_ask.py`. Configuration (model names, paths, the LLM endpoint) comes from `config.py` at the repo root — see `.env.example` for every variable.

## Requirements

- Python 3.14+, [uv](https://docs.astral.sh/uv/)
- A local `.env` (optional — copy `.env.example` if you want to override any default)
- For `ask.py` to actually generate an answer: a running LLM server. Locally, that means [Ollama](https://ollama.com) installed and a model pulled:
  ```bash
  ollama pull llama3.1:8b-instruct-q4_K_M
  ```
  (Ollama's default `LLM_BASE_URL=http://localhost:11434/v1` already matches `.env.example`.)

Install dependencies from the repo root:

```bash
uv sync
```

## Usage

Build the index from every PDF in `data/papers/` (downloaded earlier with `ingestion.downloader`):

```bash
uv run python3 -m scripts.build_index
```

Ask a question against it:

```bash
uv run python3 -m scripts.ask "What are the main challenges of ML-based intrusion detection?"
```

Both commands accept flags that override `.env` for a single run, e.g.:

```bash
uv run python3 -m scripts.build_index --papers-dir data/papers --chunk-size 250
uv run python3 -m scripts.ask "..." --top-k 8
```

For a large corpus, checkpoint more or less often with `--save-every` (default: 10 files):

```bash
uv run python3 -m scripts.build_index --save-every 5
```

**Interrupting a long run (Ctrl+C) is safe.** `build_index` saves progress before exiting, so on CPU-only hardware where indexing 100+ papers can take hours, you can stop and resume without losing completed work. Resuming currently re-embeds everything from scratch though — see the gotcha below.

## Running the tests

```bash
uv run pytest tests/test_config.py tests/test_build_index.py tests/test_ask.py -v
```

`test_build_index.py` and `test_ask.py` use the real `sample_paper.pdf` fixture and real FAISS, but a fake encoder standing in for BGE-M3 — no real model download or LLM server is needed.

## Linting

```bash
uv run ruff check config.py scripts/ tests/
```

## Design notes & gotchas

- **`build_index` skips bad files instead of crashing.** A corrupted PDF or one with no extractable text is logged as a warning and the run continues — see `test_a_corrupted_pdf_is_skipped_without_crashing_the_run`.
- **Progress is checkpointed every `save_every` files (default 10), and once more on Ctrl+C.** Embedding on CPU is slow enough that losing hours of work to one interruption is a real cost, not a minor inconvenience.
- **Checkpointing is not the same as incremental re-runs.** `build_index` still re-parses, re-chunks, and re-embeds every PDF in `papers_dir` from scratch on every run — it has no notion of "already indexed." Re-running it after an interruption redoes the completed files too, it just guarantees you never lose more than `save_every` files of work to a crash. True incremental indexing (skip files already in the index) is a real gap, not yet built.
- **`ask` raises `ValueError` on an empty index**, rather than silently prompting the LLM with no context. Build the index first.
- **This was never run end-to-end against the real BGE-M3 model or a real LLM server in this development environment** (no access to huggingface.co or a local Ollama install in the sandbox this was built in). Run both commands for real, once, before trusting them.
- **Both scripts are also importable functions** (`build_index(...)`, `ask(...)`), not just CLIs — this is what a future `api/` FastAPI endpoint will call directly, instead of shelling out to these scripts.

## Status

- [x] Indexing pipeline (`build_index.py`)
- [x] Query pipeline (`ask.py`)
- [ ] `api/` (FastAPI wrapping `ask`) — next
- [ ] `frontend/` (Streamlit UI) — later
