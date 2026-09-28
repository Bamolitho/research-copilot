# syntax=docker/dockerfile:1
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Only the frontend extra: no torch, no faiss, no sentence-transformers
# ever enter this image. This is the entire point of splitting
# pyproject.toml's dependencies by service instead of one shared list.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra frontend

COPY frontend/ ./frontend/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

# API_BASE_URL must point at wherever the API container is actually
# reachable from this one, never localhost, e.g.:
#   docker run -e API_BASE_URL=http://api-host:8000 ...
CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
