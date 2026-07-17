"""Central configuration for the Research Copilot pipeline.

Reads settings from environment variables (loaded from a local `.env`
file if one exists), with defaults suitable for local development
against a locally running Ollama server. See `.env.example` for every
variable this project reads and what it controls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # populates os.environ from a local .env file, if present


@dataclass(frozen=True)
class Settings:
    """All configurable pipeline settings.

    Attributes:
        embedding_model: HuggingFace model id for the embedder.
        embedding_dimension: Output size of that model's vectors (must
            match `embedding_model`; 1024 for BGE-M3).
        llm_base_url: OpenAI-compatible chat completions endpoint.
        llm_model: Model name as registered on that endpoint.
        papers_dir: Where downloaded PDFs are read from.
        index_dir: Where the FAISS index is read from / written to.
        chunk_size: Words per chunk (see ingestion.chunker.Chunker).
        chunk_overlap: Overlap, in words, between consecutive chunks.
        top_k: Number of chunks retrieved per question by default.
    """

    embedding_model: str
    embedding_dimension: int
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float
    papers_dir: Path
    index_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    save_every: int
    disable_llm_thinking: bool


def load_settings() -> Settings:
    """Build a Settings instance from environment variables.

    Every field has a working default for local development against a
    locally running Ollama server -- nothing needs to be set in `.env`
    to get started, only to override a default.
    """
    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", "qwen3:4b"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
        papers_dir=Path(os.getenv("PAPERS_DIR", "data/papers")),
        index_dir=Path(os.getenv("INDEX_DIR", "data/index")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
        top_k=int(os.getenv("TOP_K", "5")),
        save_every=int(os.getenv("SAVE_EVERY", "10")),
        disable_llm_thinking=os.getenv("DISABLE_LLM_THINKING", "false").lower() == "true",
    )
