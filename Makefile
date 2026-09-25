.DEFAULT_GOAL := help

# ---- Configurable overrides, e.g. `make ask QUESTION="..."` ----------------
QUERY              ?= retrieval augmented generation
MAX_RESULTS        ?= 20
QUESTION           ?= What are the main challenges of retrieval augmented generation?
LLM_MODEL          ?= qwen3:4b
DOCKERHUB_USERNAME ?=

.PHONY: help install sync setup test lint format check \
        ollama-pull download index ask serve ui pipeline \
        docker-build-api-base docker-push-api-base docker-build-api docker-build-frontend \
        clean clean-cache clean-index

help: ## Show this help
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install/sync all dependencies, all services included (uv sync)
	uv sync --all-extras

sync: install ## Alias for install

setup: install ## First-time setup: install deps, create .env if missing
	@[ -f .env ] || cp .env.example .env
	@echo "Done. Now run: make ollama-pull"

test: ## Run the full test suite
	uv run pytest tests/ -v

lint: ## Lint the codebase with ruff
	uv run ruff check .

format: ## Auto-format the codebase with ruff
	uv run ruff format .

check: lint test ## Lint then test, run before every commit

ollama-pull: ## Pull the default local LLM via Ollama (override with LLM_MODEL=...)
	ollama pull $(LLM_MODEL)

download: ## Download papers from arXiv (override with QUERY=..., MAX_RESULTS=...)
	uv run python3 -m ingestion.downloader --query "$(QUERY)" --max-results $(MAX_RESULTS)

index: ## Build the FAISS index from downloaded papers
	uv run python3 -m scripts.build_index

ask: ## Ask a question against the index (override with QUESTION=...)
	uv run python3 -m scripts.ask "$(QUESTION)"

serve: ## Run the API locally (http://127.0.0.1:8000, auto-reload)
	uv run uvicorn api.main:get_app --factory --reload

ui: ## Run the Streamlit frontend (needs `make serve` running in another terminal)
	uv run streamlit run frontend/app.py

docker-build-api-base: ## Build the API base image (model baked in). Run rarely: only when the embedding model or api dependencies change.
	docker build -f api/Dockerfile.base -t research-copilot-api-base:latest .

docker-push-api-base: ## Push the API base image to Docker Hub. Requires DOCKERHUB_USERNAME=<your-username>
	@[ -n "$(DOCKERHUB_USERNAME)" ] || (echo "Set DOCKERHUB_USERNAME=<your-username>" && exit 1)
	docker tag research-copilot-api-base:latest $(DOCKERHUB_USERNAME)/research-copilot-api-base:latest
	docker push $(DOCKERHUB_USERNAME)/research-copilot-api-base:latest

docker-build-api: docker-build-api-base ## Build the API application image locally (builds the base first, cheap no-op if unchanged)
	docker build -f api/Dockerfile -t research-copilot-api:latest .

docker-build-frontend: ## Build the frontend Docker image
	docker build -f frontend/Dockerfile -t research-copilot-frontend:latest .

pipeline: download index ask ## Run everything end to end: download -> index -> ask
	@echo ""
	@echo "Pipeline complete."

clean-cache: ## Remove Python/tool caches (__pycache__, .pytest_cache, .ruff_cache)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-index: ## Delete the built FAISS index (does NOT delete downloaded papers)
	rm -rf data/index

clean: clean-cache ## Alias for clean-cache
