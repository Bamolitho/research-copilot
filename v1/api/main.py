"""FastAPI wrapper around the ask() pipeline.

The index, embedder, and LLM client are loaded once -- not per request,
which is the entire point of an API over the CLI script (scripts.ask
reloads everything on every invocation).

`create_app` builds the app around an already-loaded AppState, and is
what tests use directly, with a fake state -- no real index, model, or
LLM server required. `get_app` is the production entry point: it loads
real settings and is only ever called by uvicorn (see the README for
the run command), never at import time, so importing this module never
has the side effect of loading a 2GB model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from config import load_settings
from fastapi import FastAPI, HTTPException
from scripts.ask import ask

from api.schemas import AskRequest, AskResponse, Citation, HealthResponse
from ingestion.embeddings import Embedder
from llm.generate import Generator
from vector_db.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Everything loaded once at startup and reused across every request.

    Attributes:
        store: The loaded FAISS index to search.
        embedder: Embedding model wrapper, must match the model used
            to build `store`.
        generator: LLM client used to generate answers.
        default_top_k: Number of chunks to retrieve when a request
            doesn't specify `top_k`.
        disable_llm_thinking: Passed through to llm.prompt.build_prompt.
    """

    store: FaissVectorStore
    embedder: Embedder
    generator: Generator
    default_top_k: int
    disable_llm_thinking: bool


def create_app(state: AppState) -> FastAPI:
    """Build the FastAPI app around a given, already-loaded AppState.

    Args:
        state: Everything the endpoints need, already constructed.
            Inject a fake one in tests to avoid needing a real index,
            embedding model, or LLM server.

    Returns:
        A ready-to-serve FastAPI app with /health and /ask registered.
    """
    app = FastAPI(
        title="Research Copilot API",
        description="Ask questions answered from a corpus of scientific papers, with citations.",
    )
    app.state.app_state = state

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report whether the index is loaded and how many chunks it holds."""
        return HealthResponse(
            status="ok",
            indexed_chunks=state.store.size,
            embedding_model=state.embedder.model_name,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_question(request: AskRequest) -> AskResponse:
        """Answer a question, grounded in the indexed papers.

        Raises:
            HTTPException: 503 if the index has nothing to search
                (not yet built); 502 if the LLM backend fails or times out.
        """
        top_k = request.top_k or state.default_top_k

        try:
            answer = ask(
                request.question,
                store=state.store,
                embedder=state.embedder,
                generator=state.generator,
                top_k=top_k,
                disable_thinking=state.disable_llm_thinking,
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc

        citations = [
            Citation(
                number=number,
                source=result.chunk.source_path.name,
                page_start=result.chunk.page_start,
                page_end=result.chunk.page_end,
                score=result.score,
                excerpt=result.chunk.text,
            )
            for number, result in answer.citations.items()
        ]
        return AskResponse(answer=answer.text, citations=citations)

    return app


def _load_state_from_settings() -> AppState:
    """Load a real AppState from environment settings, for production use."""
    settings = load_settings()
    logger.info("Loading index from %s", settings.index_dir)
    store = FaissVectorStore.load(settings.index_dir)
    logger.info("Loaded %d chunks", store.size)

    return AppState(
        store=store,
        embedder=Embedder(model_name=settings.embedding_model),
        generator=Generator(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        ),
        default_top_k=settings.top_k,
        disable_llm_thinking=settings.disable_llm_thinking,
    )


def get_app() -> FastAPI:
    """Uvicorn factory entry point -- only called at real server startup.

    Run with: uv run uvicorn api.main:get_app --factory
    Never call this at import time; it loads the real index and model.
    """
    return create_app(_load_state_from_settings())
