"""Ranking validation requires one distinct candidate at every position."""

from itertools import permutations

import pytest

from src.evaluation import ValidationError, validate_ranking


@pytest.mark.parametrize("ranking", list(permutations(("A", "B", "C"))))
def test_every_valid_permutation_keeps_its_order(ranking):
    assert validate_ranking(ranking) == list(ranking)


@pytest.mark.parametrize("ranking", [
    ["A", "A", "C"], ["C", "B", "C"], ["A", "B"], ["A", "B", "C", "A"],
    ["A", "B", "D"], ["a", "b", "c"], ["A", "B", None], ["A", "B", []],
    [], "ABC", None, {"A", "B", "C"},
])
def test_duplicate_missing_or_unknown_rankings_are_rejected(ranking):
    with pytest.raises(ValidationError):
        validate_ranking(ranking)


def test_validation_returns_an_independent_list():
    ranking = ["A", "C", "B"]
    checked = validate_ranking(ranking)
    checked.reverse()
    assert ranking == ["A", "C", "B"]
