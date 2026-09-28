"""Unit tests for config.py."""

from pathlib import Path

import pytest
from config import load_settings

_ENV_VARS = [
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "PAPERS_DIR",
    "INDEX_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "SAVE_EVERY",
    "DISABLE_LLM_THINKING",
]


@pytest.fixture(autouse=True)
def _clear_relevant_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure defaults are actually tested, regardless of the local shell's env."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestDefaults:
    def test_defaults_target_a_local_ollama_server(self) -> None:
        settings = load_settings()
        assert settings.llm_base_url == "http://localhost:11434/v1"
        assert settings.llm_model == "llama3.2:3b"
        assert settings.llm_timeout_seconds == 180.0

    def test_defaults_use_bge_m3(self) -> None:
        settings = load_settings()
        assert settings.embedding_model == "BAAI/bge-m3"
        assert settings.embedding_dimension == 1024

    def test_default_paths_are_relative_to_data_dir(self) -> None:
        settings = load_settings()
        assert settings.papers_dir == Path("data/papers")
        assert settings.index_dir == Path("data/index")

    def test_default_chunking_and_retrieval_values(self) -> None:
        settings = load_settings()
        assert settings.chunk_size == 200
        assert settings.chunk_overlap == 40
        assert settings.top_k == 5
        assert settings.save_every == 10
        assert settings.disable_llm_thinking is False


class TestEnvironmentOverrides:
    def test_llm_settings_can_be_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "https://my-service.run.app/v1")
        monkeypatch.setenv("LLM_MODEL", "meta-llama/Llama-3-8B-Instruct")

        settings = load_settings()

        assert settings.llm_base_url == "https://my-service.run.app/v1"
        assert settings.llm_model == "meta-llama/Llama-3-8B-Instruct"

    def test_numeric_settings_are_parsed_as_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "300")
        monkeypatch.setenv("TOP_K", "10")

        settings = load_settings()

        assert settings.chunk_size == 300
        assert settings.top_k == 10

    def test_path_settings_are_parsed_as_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INDEX_DIR", "/tmp/custom-index")

        settings = load_settings()

        assert settings.index_dir == Path("/tmp/custom-index")

    def test_disable_llm_thinking_can_be_turned_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISABLE_LLM_THINKING", "true")

        settings = load_settings()

        assert settings.disable_llm_thinking is True
