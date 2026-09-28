"""Request/response models for the API, validated by FastAPI via Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body of a POST /ask request."""

    question: str = Field(..., min_length=1, description="The question to answer.")
    top_k: int | None = Field(
        default=None, gt=0, description="Overrides the default number of retrieved chunks."
    )


class Citation(BaseModel):
    """One citation in an answer, resolved back to its source chunk."""

    number: int
    source: str
    page_start: int
    page_end: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    """Body of a successful POST /ask response."""

    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    """Body of a GET /health response."""

    status: str
    indexed_chunks: int
    embedding_model: str
