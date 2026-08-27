"""Unit tests for reciprocal rank fusion."""

from outpost.retrieval.hybrid import reciprocal_rank_fusion


def test_item_ranked_first_in_both_lists_wins() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0] == "a"


def test_item_missing_from_one_list_still_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b"]])
    assert set(fused) == {"a", "b"}


def test_empty_rankings_produce_empty_fusion() -> None:
    assert reciprocal_rank_fusion([]) == []
