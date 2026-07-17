"""Unit tests for llm.generate.

All tests mock the HTTP layer: no real vLLM (or any other) server is
required to run these. An end-to-end check against a real server
belongs in a separate, explicitly marked integration test.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.chunker import Chunk
from llm.generate import GeneratedAnswer, Generator
from llm.prompt import build_prompt
from vector_db.faiss_store import SearchResult

SOURCE = Path("fake.pdf")


def _prompt():
    chunk = Chunk(chunk_id=0, text="an excerpt", source_path=SOURCE, page_start=1, page_end=1)
    result = SearchResult(chunk=chunk, score=0.9)
    return build_prompt("a question", [result])


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


@pytest.fixture
def mock_session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def generator(mock_session: MagicMock) -> Generator:
    return Generator(base_url="http://localhost:8000/v1", model="test-model", session=mock_session)


class TestGenerate:
    def test_returns_the_model_answer_text(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("The main challenges are X and Y. [1]")

        answer = generator.generate(_prompt())

        assert isinstance(answer, GeneratedAnswer)
        assert answer.text == "The main challenges are X and Y. [1]"

    def test_carries_the_citation_map_through_unchanged(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        prompt = _prompt()
        mock_session.post.return_value = _mock_response("An answer. [1]")

        answer = generator.generate(prompt)

        assert answer.citations == prompt.citations

    def test_posts_to_the_chat_completions_endpoint(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        called_url = mock_session.post.call_args.args[0]
        assert called_url == "http://localhost:8000/v1/chat/completions"

    def test_sends_the_prompt_text_as_the_user_message(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("An answer.")
        prompt = _prompt()

        generator.generate(prompt)

        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["messages"] == [{"role": "user", "content": prompt.text}]
        assert payload["model"] == "test-model"

    def test_defaults_to_zero_temperature(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.0

    def test_custom_temperature_is_forwarded(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt(), temperature=0.7)

        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.7

    def test_raises_on_http_error(self, generator: Generator, mock_session: MagicMock) -> None:
        mock_session.post.return_value.raise_for_status.side_effect = RuntimeError("HTTP 500")
        with pytest.raises(RuntimeError):
            generator.generate(_prompt())

    def test_strips_a_trailing_slash_from_base_url(self, mock_session: MagicMock) -> None:
        generator = Generator(base_url="http://localhost:8000/v1/", model="m", session=mock_session)
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        called_url = mock_session.post.call_args.args[0]
        assert called_url == "http://localhost:8000/v1/chat/completions"

    def test_defaults_to_a_180_second_timeout(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        assert mock_session.post.call_args.kwargs["timeout"] == 180.0

    def test_custom_timeout_is_forwarded(self, mock_session: MagicMock) -> None:
        generator = Generator(
            base_url="http://localhost:11434/v1", model="m", session=mock_session, timeout=300.0
        )
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        assert mock_session.post.call_args.kwargs["timeout"] == 300.0


class TestThinkingBlockStripping:
    def test_strips_reasoning_before_a_closing_think_tag(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        # Matches the actual shape observed from qwen3:4b on Ollama: no
        # opening <think> tag in `content` (implicit in the chat
        # template), just the reasoning text, the closing tag, then
        # the real answer.
        raw = (
            'Okay, the user said "Say hello." So I need to respond...\n'
            'Let me go with "Hello! How can I assist you today?"\n'
            "</think>\n\n"
            "Hello! How can I assist you today? \U0001f60a"
        )
        mock_session.post.return_value = _mock_response(raw)

        answer = generator.generate(_prompt())

        assert answer.text == "Hello! How can I assist you today? \U0001f60a"
        assert "</think>" not in answer.text
        assert "Okay, the user said" not in answer.text

    def test_leaves_text_unchanged_when_no_think_tag(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        mock_session.post.return_value = _mock_response("A clean answer, no reasoning trace. [1]")

        answer = generator.generate(_prompt())

        assert answer.text == "A clean answer, no reasoning trace. [1]"

    def test_strips_an_explicit_opening_think_tag_too(
        self, generator: Generator, mock_session: MagicMock
    ) -> None:
        raw = "<think>\nSome internal reasoning.\n</think>\nThe final answer."
        mock_session.post.return_value = _mock_response(raw)

        answer = generator.generate(_prompt())

        assert answer.text == "The final answer."


class TestAuthHeaderProvider:
    def test_no_auth_header_by_default(self, generator: Generator, mock_session: MagicMock) -> None:
        # e.g. a local, unauthenticated Ollama server.
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        headers = mock_session.post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_static_api_key_is_sent_as_a_bearer_header(self, mock_session: MagicMock) -> None:
        # e.g. a fixed xAI/OpenAI-style API key.
        generator = Generator(
            base_url="https://api.x.ai/v1",
            model="grok-4.1-fast",
            auth_header_provider=lambda: "Bearer my-static-key",
            session=mock_session,
        )
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())

        headers = mock_session.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-static-key"

    def test_provider_is_called_fresh_on_every_request(self, mock_session: MagicMock) -> None:
        # e.g. a GCP identity token that expires and must be refreshed --
        # the provider must be re-invoked, never cached by Generator.
        tokens = iter(["Bearer token-1", "Bearer token-2"])
        generator = Generator(
            base_url="https://my-service.run.app/v1",
            model="llama-3",
            auth_header_provider=lambda: next(tokens),
            session=mock_session,
        )
        mock_session.post.return_value = _mock_response("An answer.")

        generator.generate(_prompt())
        generator.generate(_prompt())

        first_call_headers = mock_session.post.call_args_list[0].kwargs["headers"]
        second_call_headers = mock_session.post.call_args_list[1].kwargs["headers"]
        assert first_call_headers["Authorization"] == "Bearer token-1"
        assert second_call_headers["Authorization"] == "Bearer token-2"
