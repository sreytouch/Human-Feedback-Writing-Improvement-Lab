"""Ranking pairs remain grouped and mirrored copies never cross the holdout."""

import pandas as pd
import pytest

from src.evaluation import RUBRIC
from src.preference_model import (
    construct_preference_pairs,
    predict_comparison,
    train_preference_model,
    training_requirement,
)


def dataset(count=12):
    rows = []
    rankings = [("A", "C", "B"), ("C", "B", "A"), ("B", "A", "C")]
    for index in range(count):
        ranking = rankings[index % len(rankings)]
        record = {
            "evaluation_id": f"evaluation-{index:03}",
            "first_place": ranking[0], "second_place": ranking[1], "third_place": ranking[2],
        }
        for candidate, score in zip(ranking, (5, 3, 1)):
            for criterion in RUBRIC:
                record[f"{candidate}_{criterion}"] = score
        rows.append(record)
    return pd.DataFrame(rows)


def test_pair_construction_follows_ranking_not_alphabet():
    frame = dataset(1)
    pairs = construct_preference_pairs(frame)
    assert list(zip(pairs["left_candidate"], pairs["right_candidate"])) == [("A", "C"), ("A", "B"), ("C", "B")]
    assert list(pairs["preferred_left"]) == [1, 1, 1]
    assert list(pairs["meaning_preservation"]) == [2, 4, 2]
    assert list(pairs["evaluation_id"]) == ["evaluation-000"] * 3


def test_mirrored_pairs_have_opposite_features_and_labels():
    pairs = construct_preference_pairs(dataset(1), mirror=True)
    assert len(pairs) == 6
    for index in range(0, len(pairs), 2):
        original, mirrored = pairs.iloc[index], pairs.iloc[index + 1]
        assert original["left_candidate"] == mirrored["right_candidate"]
        assert original["right_candidate"] == mirrored["left_candidate"]
        assert original["preferred_left"] == 1
        assert mirrored["preferred_left"] == 0
        for criterion in RUBRIC:
            assert original[criterion] == -mirrored[criterion]


def test_empty_pairs_preserve_schema():
    pairs = construct_preference_pairs(pd.DataFrame())
    assert pairs.empty
    assert "evaluation_id" in pairs
    assert set(RUBRIC).issubset(pairs.columns)


def test_duplicate_rank_rejected():
    frame = dataset(1)
    frame.loc[0, "third_place"] = "A"
    with pytest.raises(ValueError, match="exactly once"):
        construct_preference_pairs(frame)


@pytest.mark.parametrize("value", [0, 6, 2.5, float("nan"), float("inf"), "bad"])
def test_invalid_score_rejected(value):
    frame = dataset(1).astype({"A_clarity": object})
    frame.loc[0, "A_clarity"] = value
    with pytest.raises(ValueError, match="whole number"):
        construct_preference_pairs(frame)


def test_insufficient_distinct_evaluations_explained():
    assert training_requirement(pd.DataFrame()) is not None
    assert training_requirement(dataset(7)) is not None
    assert training_requirement(dataset(8)) is None
    with pytest.raises(ValueError, match="at least 8"):
        train_preference_model(dataset(7))


def test_split_keeps_all_related_pairs_in_one_group():
    frame = dataset()
    result = train_preference_model(frame)
    assert len(result.training_ids) == 9
    assert len(result.held_out_ids) == 3
    assert set(result.training_ids).isdisjoint(result.held_out_ids)
    assert set(result.training_ids) | set(result.held_out_ids) == set(frame["evaluation_id"])
    assert result.training_pair_count == 27
    assert result.augmented_training_pair_count == 54
    assert result.held_out_pair_count == 9
    assert result.augmented_held_out_pair_count == 18
    training = construct_preference_pairs(frame.loc[frame["evaluation_id"].isin(result.training_ids)], mirror=True)
    held_out = construct_preference_pairs(frame.loc[frame["evaluation_id"].isin(result.held_out_ids)], mirror=True)
    assert set(training["evaluation_id"]).isdisjoint(held_out["evaluation_id"])
    assert set(result.model.classes_) == {0, 1}
    assert result.accuracy == 1.0
    assert len(result.coefficients) == 7


def test_group_split_is_reproducible_despite_row_order():
    frame = dataset()
    first = train_preference_model(frame)
    second = train_preference_model(frame.sample(frac=1, random_state=11))
    assert first.training_ids == second.training_ids
    assert first.held_out_ids == second.held_out_ids
    assert first.accuracy == second.accuracy


def test_duplicate_evaluation_ids_rejected():
    frame = dataset()
    frame.loc[1, "evaluation_id"] = frame.loc[0, "evaluation_id"]
    with pytest.raises(ValueError, match="unique"):
        train_preference_model(frame)


def test_prediction_uses_correct_group_and_reverses_with_pair():
    frame = dataset()
    result = train_preference_model(frame)
    for identifier, membership in ((result.training_ids[0], "Training evaluation"), (result.held_out_ids[0], "Held-out evaluation")):
        record = frame.loc[frame["evaluation_id"] == identifier].iloc[0].to_dict()
        first = predict_comparison(result, record, "A", "B")
        reversed_pair = predict_comparison(result, record, "B", "A")
        assert first["membership"] == membership
        assert first["probability_left"] == pytest.approx(reversed_pair["probability_right"])
        assert first["predicted_candidate"] == reversed_pair["predicted_candidate"]
        assert first["predicted_candidate"] == first["human_preferred_candidate"]
    with pytest.raises(ValueError, match="different"):
        predict_comparison(result, record, "A", "A")


def test_changing_held_out_scores_does_not_change_fitted_model():
    frame = dataset()
    original = train_preference_model(frame)
    changed = frame.copy()
    held_out_mask = changed["evaluation_id"].isin(original.held_out_ids)
    for candidate in ("A", "B", "C"):
        for criterion in RUBRIC:
            changed.loc[held_out_mask, f"{candidate}_{criterion}"] = 3
    updated = train_preference_model(changed)
    pd.testing.assert_frame_equal(original.coefficients, updated.coefficients)
    assert original.training_ids == updated.training_ids
    assert updated.accuracy == 0.5
