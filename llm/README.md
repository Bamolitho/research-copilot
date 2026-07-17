# `llm/`

Turns retrieved chunks and a question into a grounded, citable answer. Two pieces: building the prompt (`prompt.py`) and calling the model (`generate.py`).

## What's here

| File | Purpose |
|---|---|
| `prompt.py` | Assembles a system instruction, numbered context, and the question into one prompt. Also builds the citation map. Library only, no CLI. |
| `generate.py` | Thin HTTP client for any OpenAI-compatible chat completions endpoint (vLLM, TGI, ...). Library only, no CLI. |
| `__init__.py` | Makes this folder an importable package (`llm.prompt`, `llm.generate`). |

Corresponding tests live in `tests/test_prompt.py` and `tests/test_generate.py`, not in this folder — see [Running the tests](#running-the-tests).

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A running OpenAI-compatible LLM server for actual generation (e.g. vLLM serving Llama 3 or Mistral) — not required to run this folder's tests, which are fully mocked

Install dependencies from the repo root:

```bash
uv sync
```

## Usage

```python
from vector_db.faiss_store import FaissVectorStore
from ingestion.embeddings import Embedder
from llm.prompt import build_prompt
from llm.generate import Generator

embedder = Embedder()
store = FaissVectorStore.load("data/index")

query_vector = embedder.embed_query("What are the challenges of ML-based IDS?")
results = store.search(query_vector, k=5)

prompt = build_prompt("What are the challenges of ML-based IDS?", results)

generator = Generator(base_url="http://localhost:8000/v1", model="meta-llama/Llama-3-8B-Instruct")
answer = generator.generate(prompt)

print(answer.text)
for number, result in answer.citations.items():
    print(f"[{number}] {result.chunk.source_path} (page {result.chunk.page_start}), score={result.score:.2f}")
```

- `build_prompt` raises `ValueError` if `results` is empty — decide explicitly what to tell the user when retrieval finds nothing (e.g. "no relevant documents found"), rather than silently prompting the model with no grounding at all.
- `Generator.generate` defaults to `temperature=0.0` for reproducible, low-hallucination answers. Pass a higher value only if you deliberately want more varied phrasing.
- `GeneratedAnswer.citations` is the exact same dict that was built by `build_prompt` — it lets a UI resolve any `[1]`, `[2]` the model writes back to a real chunk, source file, page, and similarity score, without re-parsing the answer text.

### Local dev vs. GCP production

`Generator` is provider-agnostic: only `base_url`, `model`, and `auth_header_provider` change between environments, the calling code doesn't.

**Local (CPU, no GPU) — Ollama:**
```python
generator = Generator(base_url="http://localhost:11434/v1", model="llama3.1:8b-instruct-q4")
```
No `auth_header_provider` needed — Ollama on localhost has no auth.

**Production — vLLM on Cloud Run (GPU):** a private Cloud Run service requires a Google-signed identity token per request, which expires and must be refreshed — not a fixed API key. `auth_header_provider` is called fresh on *every* request for exactly this reason:
```python
import google.auth.transport.requests
import google.oauth2.id_token

def _gcp_auth_header() -> str:
    token = google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), audience=CLOUD_RUN_SERVICE_URL
    )
    return f"Bearer {token}"

generator = Generator(
    base_url=f"{CLOUD_RUN_SERVICE_URL}/v1",
    model="meta-llama/Llama-3-8B-Instruct",
    auth_header_provider=_gcp_auth_header,
)
```

**A hosted API (e.g. Grok, or any other OpenAI-compatible provider)** works the same way, with a static key instead of a refreshed token:
```python
generator = Generator(
    base_url="https://api.x.ai/v1",
    model="grok-4.1-fast",
    auth_header_provider=lambda: f"Bearer {os.environ['XAI_API_KEY']}",
)
```

## Running the tests

From the repo root:

```bash
uv run pytest tests/test_prompt.py tests/test_generate.py -v
```

All `generate.py` tests mock the HTTP layer — none of them require a real LLM server. Run a manual check against your own vLLM instance before trusting this in a pipeline.

## Linting

```bash
uv run ruff check llm/ tests/
```

## Design notes & gotchas

- **The LLM never receives embeddings or scores**, only the original chunk text and the question — this is the "grounding" lesson from the course, and it's why `build_prompt` reads `result.chunk.text`, never `result.score` or a raw vector.
- **One prompt, one citation map.** `RagPrompt.citations` and `GeneratedAnswer.citations` are built once by `build_prompt` and carried through unchanged — `Generator` never re-derives or re-numbers them, so a citation number always means the same chunk from prompt to answer.
- **`Generator` is provider-agnostic on purpose.** It only assumes an OpenAI-compatible `/chat/completions` endpoint, which vLLM, TGI, and several others all expose — swapping the self-hosted model doesn't require touching this client, only `base_url` and `model`.
- **No retry logic yet.** A transient network error or a 503 from an overloaded LLM server currently just raises. Add retries with backoff before relying on this in a production request path (see the delivery roadmap's Epic H, observability & reliability).
- **`auth_header_provider` is called on every request, never cached.** This is deliberate: a GCP identity token expires (~1h) and a static API key doesn't, and re-fetching on every call is the one behavior that's correct for both without `Generator` needing to know which kind of credential it's holding.

## Status

- [x] Prompt construction (`prompt.py`)
- [x] Generation client (`generate.py`)
- [ ] Guardrails (PII redaction, prompt-injection filtering) — later
- [ ] Conversation memory — later
