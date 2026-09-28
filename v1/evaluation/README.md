# `evaluation/`

Measures retriever quality with real metrics against a hand-annotated gold set, instead of judging answers by eye.

## What's here

| File | Purpose |
|---|---|
| `metrics.py` | Pure Precision@k, Recall@k, MRR functions. No I/O, no FAISS. Library only, no CLI. |
| `retrieval_eval.py` | Loads the gold set + real index + real embedder, runs retrieval, reports metrics. Has a CLI. |
| `gold_set.json` | 17 hand-verified questions with known-relevant chunks, drawn from the real corpus. |
| `__init__.py` | Makes this folder an importable package (`evaluation.metrics`, `evaluation.retrieval_eval`). |

Corresponding tests live in `tests/test_metrics.py` and `tests/test_retrieval_eval.py`.

## Requirements

Same as the rest of the project — `uv sync` from the repo root. Running the real evaluation (not just the tests) needs a built index (`data/index/`, from `scripts.build_index`) and the real BGE-M3 model.

## Usage

```bash
uv run python3 -m evaluation.retrieval_eval
```

```
Retrieval evaluation (k=5), 17 question(s)

  P@5=0.40  R@5=1.00  RR=1.00  | What is a low-utility facet in Complex Answer Retrieval...
  P@5=0.20  R@5=0.50  RR=0.33  | What two estimators of facet utility does the paper propose...
  ...

--- Averages ---
  Mean Precision@5: 0.XXX
  Mean Recall@5:    0.XXX
  MRR:              0.XXX
```

Override `k`, the gold set, or the index directory for a single run:

```bash
uv run python3 -m evaluation.retrieval_eval --k 10 --gold-set evaluation/gold_set.json --index-dir data/index
```

Programmatic use (e.g. to compare two configurations in a script):

```python
from config import load_settings
from evaluation.retrieval_eval import evaluate, load_gold_set
from ingestion.embeddings import Embedder
from vector_db.faiss_store import FaissVectorStore

settings = load_settings()
gold_questions = load_gold_set()
store = FaissVectorStore.load(settings.index_dir)
embedder = Embedder(model_name=settings.embedding_model)

results = evaluate(gold_questions, store, embedder, k=5)
```

## The gold set format

```json
[
  {
    "question": "What are the main challenges of retrieval augmented generation?",
    "relevant": [
      {"source_path": "data/papers/2508.14066v1.pdf", "chunk_id": 6}
    ],
    "notes": "optional, why this chunk was judged relevant"
  }
]
```

- **`(source_path, chunk_id)` is the key, never `chunk_id` alone.** `ingestion.chunker.Chunker` numbers chunks per document starting at 0, so chunk_id 0 exists in every document in the corpus — comparing by chunk_id alone would silently match the wrong document. `evaluation.metrics.ChunkKey` is this pair, and `test_retrieval_eval.py` has a dedicated regression test for exactly this failure mode.
- Every entry in the current `gold_set.json` was built by reading the actual chunk text and verifying the claim, not generated from a title or an assumption about what a paper probably says.

## Running the tests

```bash
uv run pytest tests/test_metrics.py tests/test_retrieval_eval.py -v
```

`test_metrics.py` is fully pure (plain data in, float out). `test_retrieval_eval.py` uses a real, small `FaissVectorStore` plus a fake-encoder-backed `Embedder` — no real model download or index required.

## Linting

```bash
uv run ruff check evaluation/ tests/
```

## Design notes & gotchas

- **Precision@k and Recall@k, not the unbounded Precision/Recall.** A RAG pipeline only ever shows the LLM the top-k chunks, so the bounded versions are what's actually relevant to answer quality — see the course deck's Part 6 for the full argument.
- **No MAP or NDCG in this version, on purpose.** Both need graded relevance judgments (e.g. 0–3, not just relevant/not-relevant), which cost meaningfully more annotation effort than a 17-question binary gold set justifies right now. `mrr` alone already captures whether the *first* relevant result ranks highly.
- **The gold set is a small, honest sample, not a statistical audit.** 17 questions across 9 of the corpus's 50 documents is enough to catch a gross regression (e.g. an unrelated paper's chunks suddenly outranking the right one), not enough to certify overall corpus-wide retrieval quality. Expand it opportunistically; don't treat its current numbers as final.
- **This evaluates the retriever only.** A perfect Precision@k/Recall@k/MRR here says nothing about whether the LLM actually used the retrieved chunks well — that's generator evaluation, a separate, not-yet-built piece (Faithfulness, LLM-as-a-Judge; see the course deck's Part 7).

## Status

- [x] Retriever evaluation (`retrieval_eval.py`)
- [ ] Generator evaluation (Faithfulness, LLM-as-a-Judge) — next
- [ ] Expand `gold_set.json` beyond 17 questions / 9 documents — ongoing, opportunistic
