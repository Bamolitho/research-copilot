# syntax=docker/dockerfile:1
#
# Application image for the API service. Built automatically by the
# CD pipeline on every merge to main.
#
# Starts from the manually-built base image (api/Dockerfile.base),
# which already has dependencies and the embedding model baked in, so
# this build only ever copies source code and never touches the
# network. This is what makes it safe to rebuild on every push,
# unlike the base image.
ARG BASE_IMAGE=research-copilot-api-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY config.py ./
COPY api/ ./api/
COPY ingestion/ ./ingestion/
COPY vector_db/ ./vector_db/
COPY llm/ ./llm/
COPY scripts/ ./scripts/

EXPOSE 8000

# The FAISS index is deliberately NOT baked into this image: it is
# built from a corpus that grows independently of the application
# code, on its own schedule (see scripts/build_index.py). Mount it as
# a volume at runtime instead, e.g.:
#   docker run -v ./data/index:/app/data/index ...
# The API fails fast at startup if this path is empty or missing.
CMD ["uv", "run", "uvicorn", "api.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
