"""Unit tests for frontend.api_client.

No Streamlit dependency anywhere in this file, by design -- these
tests run regardless of whether Streamlit is installed.
"""

from unittest.mock import MagicMock, patch

import requests

from frontend.api_client import AnswerResult, Citation, HealthStatus, ask_question, get_health

BASE_URL = "http://127.0.0.1:8000"


class TestGetHealth:
    @patch("frontend.api_client.requests.get")
    def test_returns_health_status_on_success(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "ok",
            "indexed_chunks": 2623,
            "embedding_model": "BAAI/bge-m3",
        }

        result = get_health(BASE_URL)

        assert result == HealthStatus(indexed_chunks=2623, embedding_model="BAAI/bge-m3")

    @patch("frontend.api_client.requests.get")
    def test_returns_none_when_api_is_unreachable(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.ConnectionError("connection refused")

        assert get_health(BASE_URL) is None

    @patch("frontend.api_client.requests.get")
    def test_returns_none_on_timeout(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.Timeout("timed out")

        assert get_health(BASE_URL) is None

    @patch("frontend.api_client.requests.get")
    def test_calls_the_health_endpoint(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "indexed_chunks": 1,
            "embedding_model": "m",
        }

        get_health(BASE_URL)

        mock_get.assert_called_once_with(f"{BASE_URL}/health", timeout=10)


class TestAskQuestion:
    @patch("frontend.api_client.requests.post")
    def test_returns_answer_with_citations(self, mock_post: MagicMock) -> None:
        mock_post.return_value.json.return_value = {
            "answer": "RAG reduces hallucination. [1]",
            "citations": [
                {
                    "number": 1,
                    "source": "paper.pdf",
                    "page_start": 3,
                    "page_end": 3,
                    "score": 0.87,
                    "excerpt": "RAG grounds answers in retrieved text.",
                }
            ],
        }

        result = ask_question(BASE_URL, "What does RAG do?")

        assert result == AnswerResult(
            answer="RAG reduces hallucination. [1]",
            citations=[
                Citation(
                    number=1,
                    source="paper.pdf",
                    page_start=3,
                    page_end=3,
                    score=0.87,
                    excerpt="RAG grounds answers in retrieved text.",
                )
            ],
        )

    @patch("frontend.api_client.requests.post")
    def test_omits_top_k_from_payload_when_not_given(self, mock_post: MagicMock) -> None:
        mock_post.return_value.json.return_value = {"answer": "a", "citations": []}

        ask_question(BASE_URL, "a question")

        sent_payload = mock_post.call_args.kwargs["json"]
        assert "top_k" not in sent_payload

    @patch("frontend.api_client.requests.post")
    def test_includes_top_k_when_given(self, mock_post: MagicMock) -> None:
        mock_post.return_value.json.return_value = {"answer": "a", "citations": []}

        ask_question(BASE_URL, "a question", top_k=3)

        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["top_k"] == 3

    @patch("frontend.api_client.requests.post")
    def test_raises_on_http_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("503")

        try:
            ask_question(BASE_URL, "a question")
            raised = False
        except requests.HTTPError:
            raised = True
        assert raised

    @patch("frontend.api_client.requests.post")
    def test_handles_an_empty_citation_list(self, mock_post: MagicMock) -> None:
        mock_post.return_value.json.return_value = {
            "answer": "I don't know based on the retrieved documents.",
            "citations": [],
        }

        result = ask_question(BASE_URL, "an unanswerable question")

        assert result.citations == []
