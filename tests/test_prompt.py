"""Unit tests for llm.prompt."""

from pathlib import Path

import pytest

from ingestion.chunker import Chunk
from llm.prompt import DEFAULT_SYSTEM_PROMPT, build_prompt
from vector_db.faiss_store import SearchResult

SOURCE = Path("fake.pdf")


def _result(chunk_id: int, text: str, score: float = 0.9) -> SearchResult:
    chunk = Chunk(chunk_id=chunk_id, text=text, source_path=SOURCE, page_start=1, page_end=1)
    return SearchResult(chunk=chunk, score=score)


class TestBuildPrompt:
    def test_rejects_empty_results(self) -> None:
        with pytest.raises(ValueError):
            build_prompt("a question", [])

    def test_includes_the_question(self) -> None:
        prompt = build_prompt("What causes concept drift?", [_result(0, "some excerpt")])
        assert "What causes concept drift?" in prompt.text

    def test_includes_the_default_system_prompt_by_default(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")])
        assert DEFAULT_SYSTEM_PROMPT in prompt.text

    def test_accepts_a_custom_system_prompt(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")], system_prompt="Be terse.")
        assert "Be terse." in prompt.text
        assert DEFAULT_SYSTEM_PROMPT not in prompt.text

    def test_numbers_context_entries_starting_at_1(self) -> None:
        prompt = build_prompt(
            "a question", [_result(0, "first excerpt"), _result(1, "second excerpt")]
        )
        assert "[1] first excerpt" in prompt.text
        assert "[2] second excerpt" in prompt.text

    def test_citation_map_matches_the_numbered_context(self) -> None:
        first = _result(0, "first excerpt")
        second = _result(1, "second excerpt")
        prompt = build_prompt("a question", [first, second])

        assert prompt.citations == {1: first, 2: second}

    def test_preserves_result_order_best_match_first(self) -> None:
        best = _result(0, "best match", score=0.95)
        worst = _result(1, "worst match", score=0.40)
        prompt = build_prompt("a question", [best, worst])

        assert prompt.text.index("[1] best match") < prompt.text.index("[2] worst match")

    def test_disable_thinking_appends_the_no_think_suffix(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")], disable_thinking=True)
        assert prompt.text.count("/no_think") == 1

    def test_disable_thinking_false_by_default(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")])
        assert "/no_think" not in prompt.text

    def test_includes_a_worked_citation_example_by_default(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")])
        assert "EXAMPLE" in prompt.text
        assert "[A]" in prompt.text
        assert "[B]" in prompt.text

    def test_example_can_be_disabled(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")], include_example=False)
        assert "EXAMPLE" not in prompt.text

    def test_example_letters_never_collide_with_real_numbered_citations(self) -> None:
        prompt = build_prompt("a question", [_result(0, "first"), _result(1, "second")])
        # the real context must still be numbered [1], [2], not lettered
        assert "[1] first" in prompt.text
        assert "[2] second" in prompt.text

    def test_system_prompt_instructs_per_sentence_citation(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")])
        assert "every factual sentence" in prompt.text

    def test_system_prompt_instructs_addressing_every_relevant_excerpt(self) -> None:
        prompt = build_prompt("a question", [_result(0, "some excerpt")])
        assert "not only the first one or two" in prompt.text

    def test_system_prompt_instructs_ignoring_source_citations(self) -> None:
        # Regression test: a real run leaked author-year citations from
        # the source text itself (e.g. "[Zobel, 1998]") into the answer,
        # mixed in with our own excerpt numbers.
        prompt = build_prompt("a question", [_result(0, "some excerpt")])
        assert "ignore these completely" in prompt.text
