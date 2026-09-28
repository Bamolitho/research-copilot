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

DEFAULT_TIMEOUT_SECONDS = 180.0

# Some models (observed: qwen3:4b on Ollama) emit their internal
# reasoning trace inline in the response content regardless of API
# flags meant to disable it -- both a "/no_think" prompt instruction
# and Ollama's "think": false parameter were confirmed ignored. The
# reasoning is reliably terminated by a literal "</think>" tag even
# when no opening "<think>" tag is present in the returned content
# (it's implicit in the chat template). Stripping up to and including
# that tag keeps GeneratedAnswer.text as just the actual answer.
# This does NOT reduce latency -- the model still generates every
# reasoning token before this ever runs, it only cleans the output.
_THINK_END_TAG = "</think>"


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
        timeout: Seconds to wait for a response before giving up.
            Defaults to 180s -- CPU-only inference (e.g. Ollama without
            a GPU) can genuinely take over a minute for a full answer;
            60s was observed to be too aggressive in practice.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        auth_header_provider: Callable[[], str] | None = None,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._auth_header_provider = auth_header_provider
        self._session = session or requests.Session()
        self.timeout = timeout

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
            timeout=self.timeout,
        )
        response.raise_for_status()
        answer_text = response.json()["choices"][0]["message"]["content"]
        answer_text = self._strip_thinking_block(answer_text)
        return GeneratedAnswer(text=answer_text, citations=prompt.citations)

    @staticmethod
    def _strip_thinking_block(text: str) -> str:
        """Remove a leading reasoning trace, if the model emitted one.

        Looks for a literal "</think>" tag and keeps only what follows
        it. Text with no such tag (most models, most of the time) is
        returned unchanged.
        """
        if _THINK_END_TAG in text:
            return text.split(_THINK_END_TAG, 1)[1].strip()
        return text
