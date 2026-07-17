"""Calls a self-hosted LLM to generate an answer from a RAG prompt.

Talks to any OpenAI-compatible chat completions endpoint over plain
HTTP -- this is what vLLM and Text Generation Inference both expose --
rather than depending on a vendor-specific SDK. The same client works
against a local vLLM server or any other compatible endpoint, just by
changing `base_url`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import requests

from llm.prompt import RagPrompt
from vector_db.faiss_store import SearchResult

DEFAULT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class GeneratedAnswer:
    """The model's answer, alongside the citations available to check it.

    Attributes:
        text: The generated answer text, as returned by the model.
        citations: The same citation map passed to `Generator.generate`,
            carried through so a caller can resolve any "[N]" the
            model cites back to a real chunk, source, and page.
    """

    text: str
    citations: dict[int, SearchResult]


class Generator:
    """A thin client for an OpenAI-compatible chat completions endpoint.

    Args:
        base_url: Base URL of the endpoint, e.g. "http://localhost:11434/v1"
            for a local Ollama server, or a Cloud Run service URL in
            production. A trailing slash is stripped.
        model: Model name as registered on that server (e.g.
            "llama3.1:8b-instruct-q4" for Ollama, or a Hugging Face
            model id for vLLM).
        auth_header_provider: Optional callable returning the value of
            the "Authorization" header, called fresh on every request.
            Leave as None for an unauthenticated local server (e.g.
            Ollama on localhost). Use a callable for anything that
            needs a token, whether static (an xAI/OpenAI-style API
            key: `lambda: f"Bearer {api_key}"`) or dynamically
            refreshed (a GCP-signed identity token for a private Cloud
            Run service, which expires and must be re-fetched) --
            calling it fresh each time supports both without this
            class needing to know which kind it is.
        session: Optional requests.Session to reuse. Inject a mock
            session in tests to avoid needing a real server running.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        auth_header_provider: Callable[[], str] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._auth_header_provider = auth_header_provider
        self._session = session or requests.Session()

    def generate(self, prompt: RagPrompt, temperature: float = 0.0) -> GeneratedAnswer:
        """Send a RAG prompt to the LLM and return its answer.

        Args:
            prompt: The assembled prompt, from llm.prompt.build_prompt.
            temperature: Sampling temperature. Defaults to 0.0 for
                reproducible, low-hallucination answers -- this is a
                grounded-answer assistant, not a creative one.

        Returns:
            The generated answer, carrying the same citation map as `prompt`.

        Raises:
            requests.HTTPError: If the request to the LLM server fails.
        """
        headers = {}
        if self._auth_header_provider is not None:
            headers["Authorization"] = self._auth_header_provider()

        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt.text}],
                "temperature": temperature,
            },
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer_text = response.json()["choices"][0]["message"]["content"]
        return GeneratedAnswer(text=answer_text, citations=prompt.citations)
