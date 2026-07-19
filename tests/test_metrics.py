"""Unit tests for evaluation.metrics.

All pure functions, no fixtures beyond plain Python data needed.
"""

from evaluation.metrics import mean, precision_at_k, recall_at_k, reciprocal_rank

DOC_A = "data/papers/a.pdf"
DOC_B = "data/papers/b.pdf"


class TestPrecisionAtK:
    def test_basic_case(self) -> None:
        retrieved = [(DOC_A, 5), (DOC_A, 9), (DOC_A, 12), (DOC_B, 21), (DOC_A, 30), (DOC_B, 44)]
        relevant = {(DOC_A, 5), (DOC_A, 12), (DOC_A, 30)}

        assert precision_at_k(retrieved, relevant, k=6) == 0.5

    def test_perfect_precision(self) -> None:
        retrieved = [(DOC_A, 1), (DOC_A, 2)]
        relevant = {(DOC_A, 1), (DOC_A, 2)}

        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_zero_precision(self) -> None:
        retrieved = [(DOC_A, 1), (DOC_A, 2)]
        relevant = {(DOC_B, 99)}

        assert precision_at_k(retrieved, relevant, k=2) == 0.0

    def test_k_smaller_than_retrieved_list(self) -> None:
        retrieved = [(DOC_A, 1), (DOC_A, 2), (DOC_A, 3)]
        relevant = {(DOC_A, 1), (DOC_A, 2), (DOC_A, 3)}

        # only the first result is considered when k=1
        assert precision_at_k(retrieved, relevant, k=1) == 1.0

    def test_empty_retrieved_list(self) -> None:
        assert precision_at_k([], {(DOC_A, 1)}, k=5) == 0.0

    def test_k_zero_or_negative(self) -> None:
        retrieved = [(DOC_A, 1)]
        relevant = {(DOC_A, 1)}

        assert precision_at_k(retrieved, relevant, k=0) == 0.0
        assert precision_at_k(retrieved, relevant, k=-1) == 0.0


class TestRecallAtK:
    def test_basic_case(self) -> None:
        # ground truth has 3 relevant chunks total, only 2 appear in top-5
        retrieved = [(DOC_A, 5), (DOC_A, 9), (DOC_A, 12), (DOC_B, 21), (DOC_A, 30)]
        relevant = {(DOC_A, 5), (DOC_A, 12), (DOC_A, 90)}  # 90 is never retrieved

        assert recall_at_k(retrieved, relevant, k=5) == 2 / 3

    def test_all_relevant_found(self) -> None:
        retrieved = [(DOC_A, 1), (DOC_A, 2), (DOC_A, 3)]
        relevant = {(DOC_A, 1), (DOC_A, 2)}

        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_empty_relevant_set_returns_zero_not_error(self) -> None:
        retrieved = [(DOC_A, 1)]
        assert recall_at_k(retrieved, set(), k=5) == 0.0

    def test_k_smaller_than_where_relevant_items_are(self) -> None:
        # the only relevant chunk is retrieved at rank 3, but k=2 cuts it off
        retrieved = [(DOC_A, 99), (DOC_A, 98), (DOC_A, 1)]
        relevant = {(DOC_A, 1)}

        assert recall_at_k(retrieved, relevant, k=2) == 0.0


class TestReciprocalRank:
    def test_first_relevant_at_rank_1(self) -> None:
        retrieved = [(DOC_A, 1), (DOC_A, 2)]
        relevant = {(DOC_A, 1)}

        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_first_relevant_at_rank_4(self) -> None:
        retrieved = [(DOC_A, 9), (DOC_A, 8), (DOC_A, 7), (DOC_A, 1)]
        relevant = {(DOC_A, 1)}

        assert reciprocal_rank(retrieved, relevant) == 0.25

    def test_no_relevant_result_returns_zero(self) -> None:
        retrieved = [(DOC_A, 9), (DOC_A, 8)]
        relevant = {(DOC_B, 1)}

        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved_list(self) -> None:
        assert reciprocal_rank([], {(DOC_A, 1)}) == 0.0

    def test_only_the_first_hit_counts(self) -> None:
        # two relevant chunks in the list; only the earliest rank matters
        retrieved = [(DOC_A, 9), (DOC_A, 1), (DOC_A, 2)]
        relevant = {(DOC_A, 1), (DOC_A, 2)}

        assert reciprocal_rank(retrieved, relevant) == 0.5  # rank 2, not rank 3


class TestMean:
    def test_basic_average(self) -> None:
        assert mean([0.5, 1.0, 0.0]) == 0.5

    def test_empty_sequence_returns_zero(self) -> None:
        assert mean([]) == 0.0

    def test_single_value(self) -> None:
        assert mean([0.75]) == 0.75
