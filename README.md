# Research Copilot

A retrieval augmented assistant that answers technical questions using a corpus of scientific papers, and backs every answer with real citations you can check.

## Why this project exists

Most question answering systems trained on general web text will give you a confident sounding answer even when they don't actually know. For scientific work that's a real problem: you need to know which paper a claim comes from, not just whether the sentence sounds plausible.

Research Copilot is built around a simple rule: never answer from memory. Every response is grounded in a fixed collection of papers, and every claim points back to a specific source, an excerpt and a similarity score, so you can verify it yourself instead of trusting the model blindly.

Example query:

> What are the main challenges of machine learning based intrusion detection systems for software defined vehicles?

The system responds with a short synthesis in natural language, a numbered list of references, the exact excerpt supporting each reference, and a similarity score for each retrieved chunk.

## How it works

![Architecture diagram of Research Copilot](./Images/architecture.svg)

The pipeline is organized in three phases.

**Ingestion** turns raw PDFs into searchable vectors. Text is extracted with PyMuPDF, split into chunks sized for retrieval, then encoded with BGE-M3.

**Retrieval** finds the passages relevant to a question. Chunks live in a FAISS index, and a query embedding is compared against it to pull back the closest matching passages.

**Generation** turns retrieved passages into an answer. The LLM only sees the retrieved excerpts plus the question, and is instructed to answer strictly from that context and to cite where each part of the answer comes from.

## Example output

```
Q: What are the main challenges of AI based IDS for CAN networks?

According to the retrieved literature, the main challenges are:

- False positives in real driving conditions
- Poor quality or unbalanced datasets
- Concept drift as attack patterns evolve
- Limited compute resources on embedded ECUs

References
[1] Paper A, similarity 0.87
[2] Paper B, similarity 0.81
[3] Paper C, similarity 0.79
```

## Scope of the first version

The first version stays intentionally narrow, so we can ship something that works end to end before adding complexity on top.

Included in v1:

- Ingest a fixed set of PDF papers, a few hundred to start with
- Extract, chunk, embed and index them
- Answer a single question with a synthesized response, numbered references, source excerpts and similarity scores
- Run locally through Docker for a reproducible environment

Not in v1, planned for later:

- Conversation memory
- Hybrid search, combining BM25 with vector search
- Reranking of retrieved chunks
- Automatic query reformulation
- Bibliography and document export
- Kubernetes deployment and a full monitoring stack

## Roadmap

| Version | What it adds |
|---|---|
| v1 | Core RAG pipeline with citations, this version |
| v2 | Click a reference to see the original paragraph, page number and source PDF |
| v3 | Conversation memory across turns |
| v4 | Hybrid search, BM25 combined with vector search |
| v5 | Reranking of retrieved chunks, BGE reranker or Cohere rerank |
| v6 | Automatic multi query reformulation |
| v7 | Bibliography generation, BibTeX, APA, IEEE, ACM |
| v8 | Export to PDF, Markdown or LaTeX |
| Later | Kubernetes deployment and monitoring with Prometheus and Grafana |

## Success criteria for v1

- At least 2 to 3 correctly attributed citations for over 80% of a hand written test set of 10 to 20 questions
- No hallucinated references, every citation must map to a chunk that was actually retrieved
- Retrieval latency under 2 to 3 seconds for a corpus of a few hundred papers

## Tech stack

| Layer | Choice for v1 | Status |
|---|---|---|
| PDF extraction | PyMuPDF | Candidate, pdfplumber as backup for tricky layouts |
| Embeddings | BGE-M3 | Decided |
| Vector store | FAISS | Decided |
| LLM | Llama 3, Mistral, Qwen or GPT | Still open |
| Orchestration | LangChain or LlamaIndex | Still open |
| API | FastAPI | Candidate |
| Frontend | Streamlit | Candidate |
| Containerization | Docker | Decided |

BGE-M3 was picked over the classic bge-large-en-v1.5 or e5-large-v2 pair because it is MIT licensed, self-hostable on CPU for a corpus this size, and supports dense, sparse and multi-vector retrieval in a single model. That last point matters because hybrid search, BM25 combined with vector search, is already planned for v4, so starting on a model built for it avoids a migration later. FAISS was picked for its simplicity and maturity for a single-machine MVP, with ChromaDB and Qdrant staying as options once the project needs multi-user access or managed hosting.

## Status

This repository is at the design stage. The scope and architecture above are set for v1, and implementation starts with the ingestion module.

## License

MIT
