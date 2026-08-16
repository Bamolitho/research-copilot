"""Streamlit UI for Research Copilot.

A thin presentation layer: every actual API call goes through
api_client.py, which has no Streamlit dependency of its own. This
file only ever renders what api_client returns.

Run with: streamlit run frontend/app.py

Note the import below: it tries `frontend.api_client` first, then falls
back to a bare `api_client`. This isn't redundancy, it resolves a real
conflict between two tools that map this file to a module differently:

- mypy analyzes this file as part of the `frontend` package (because
  `frontend/__init__.py` exists), so it only resolves the qualified
  `frontend.api_client` -- and needs that branch to succeed to see the
  real types of AnswerResult, Citation, etc.
- Streamlit runs this file as a direct script, which puts this file's
  own directory (frontend/) on sys.path instead of the repo root -- so
  only the bare `api_client` import resolves at runtime.

Both names point at the exact same file; only the entry point differs.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

try:
    from frontend.api_client import AnswerResult, Citation, ask_question, get_health
except ImportError:  # pragma: no cover - only taken under `streamlit run`
    from api_client import (  # type: ignore[import-not-found,no-redef]
        AnswerResult,
        Citation,
        ask_question,
        get_health,
    )

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def render_citation(citation: Citation) -> None:
    """Render one citation as an expandable block."""
    with st.expander(f"[{citation.number}] {citation.source} (page {citation.page_start})"):
        st.caption(f"Similarity score: {citation.score:.3f}")
        st.write(citation.excerpt)


def render_answer(result: AnswerResult) -> None:
    """Render a full answer: the text, then every citation below it."""
    st.markdown(result.answer)
    if result.citations:
        st.markdown("**Sources**")
        for citation in result.citations:
            render_citation(citation)


def render_sidebar_status() -> None:
    """Show whether the API is reachable and how many chunks it holds."""
    st.sidebar.subheader("Status")
    health = get_health(API_BASE_URL)
    if health is None:
        st.sidebar.error(f"API unreachable at {API_BASE_URL}")
        st.sidebar.caption("Is `make serve` running?")
    else:
        st.sidebar.success("API reachable")
        st.sidebar.metric("Indexed chunks", health.indexed_chunks)
        st.sidebar.caption(f"Embedding model: {health.embedding_model}")


def main() -> None:
    st.set_page_config(page_title="Research Copilot", page_icon="\U0001f4da")
    st.title("Research Copilot")
    st.caption("Ask a question, get an answer grounded in the indexed papers, with citations.")

    render_sidebar_status()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["result"])
            else:
                st.write(message["content"])

    question = st.chat_input("Ask a question about the indexed papers...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating an answer..."):
            try:
                result = ask_question(API_BASE_URL, question)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 503:
                    st.error("The index has no documents yet. Build it first with `make index`.")
                elif status == 502:
                    st.error("The LLM backend is unreachable. Is Ollama running?")
                else:
                    st.error(f"Request failed: {exc}")
                return
            except requests.RequestException:
                st.error(f"Could not reach the API at {API_BASE_URL}. Is `make serve` running?")
                return

        render_answer(result)
        st.session_state.messages.append({"role": "assistant", "result": result})


if __name__ == "__main__":
    main()
