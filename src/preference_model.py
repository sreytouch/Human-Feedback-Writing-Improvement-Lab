"""A small preference classifier; it never trains the text-generation model.

Split whole evaluations before expanding their rankings into candidate pairs.
Mirrored pairs supply both classes without leaking related pairs into holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.evaluation import RUBRIC

MIN_EVALUATIONS = 8
CANDIDATES = ("A", "B", "C")
PAIR_METADATA = ["evaluation_id", "left_candidate", "right_candidate", "preferred_left"]


@dataclass
class PreferenceModelResult:
    """Keep the exact training/holdout groups alongside their trained model."""

    model: LogisticRegression
    accuracy: float
    coefficients: pd.DataFrame
    training_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    training_pair_count: int
    held_out_pair_count: int
    augmented_training_pair_count: int
    augmented_held_out_pair_count: int


def construct_preference_pairs(frame: pd.DataFrame, mirror: bool = False) -> pd.DataFrame:
    """Expand each strict ranking into three winning-minus-losing pairs.

    A > C > B becomes (A, C), (A, B), (C, B), with label 1. When
    mirror=True, add each reversed pair with negated features and label 0.
    The mirrored pairs are not additional independent human judgments.
    """
    columns = PAIR_METADATA + list(RUBRIC)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"evaluation_id", "first_place", "second_place", "third_place"}
    required.update(f"{candidate}_{key}" for candidate in CANDIDATES for key in RUBRIC)
    if not required.issubset(frame.columns):
        raise ValueError("Each evaluation needs its ID, complete ranking, and all seven scores for A, B, and C.")
    if frame["evaluation_id"].isna().any() or frame["evaluation_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Evaluation IDs must be non-empty.")
    if frame["evaluation_id"].astype(str).duplicated().any():
        raise ValueError("Evaluation IDs must be unique.")
    rows: list[dict[str, Any]] = []
    for _, evaluation in frame.iterrows():
        ranking = [evaluation["first_place"], evaluation["second_place"], evaluation["third_place"]]
        if len(set(ranking)) != 3 or set(ranking) != set(CANDIDATES):
            raise ValueError("Rank A, B, and C exactly once each.")
        scores = {
            candidate: pd.to_numeric(pd.Series({key: evaluation[f"{candidate}_{key}"] for key in RUBRIC}), errors="coerce")
            for candidate in CANDIDATES
        }
        values = np.concatenate([scores[candidate].to_numpy(dtype=float) for candidate in CANDIDATES])
        if not np.isfinite(values).all() or (values < 1).any() or (values > 5).any() or (values % 1 != 0).any():
            raise ValueError("Every rubric score must be a whole number from 1 to 5.")
        for better_index, worse_index in ((0, 1), (0, 2), (1, 2)):
            better, worse = ranking[better_index], ranking[worse_index]
            difference = {key: float(scores[better][key] - scores[worse][key]) for key in RUBRIC}
            rows.append({
                "evaluation_id": str(evaluation["evaluation_id"]), "left_candidate": better,
                "right_candidate": worse, "preferred_left": 1, **difference,
            })
            if mirror:
                rows.append({
                    "evaluation_id": str(evaluation["evaluation_id"]), "left_candidate": worse,
                    "right_candidate": better, "preferred_left": 0,
                    **{key: -value for key, value in difference.items()},
                })
    return pd.DataFrame(rows, columns=columns)


def training_requirement(frame: pd.DataFrame) -> str | None:
    """Explain the minimum number of independent evaluations for this demo."""
    count = frame["evaluation_id"].nunique() if "evaluation_id" in frame else 0
    if count < MIN_EVALUATIONS:
        return (
            f"Collected {count} distinct evaluations. Save at least {MIN_EVALUATIONS} complete rankings "
            "to fit this demo with separate training and held-out evaluation groups. "
            "Eight is a teaching minimum, not enough for a reliable benchmark."
        )
    return None


def train_preference_model(frame: pd.DataFrame) -> PreferenceModelResult:
    """Split evaluation groups 75/25, then construct and mirror pairs per split."""
    reason = training_requirement(frame)
    if reason:
        raise ValueError(reason)
    # Validate IDs without constructing pairs; related pairs must stay together.
    ids = frame["evaluation_id"].astype(str)
    if frame["evaluation_id"].isna().any() or ids.str.strip().eq("").any():
        raise ValueError("Evaluation IDs must be non-empty.")
    if ids.duplicated().any():
        raise ValueError("Evaluation IDs must be unique.")
    identifiers = sorted(ids)
    training_ids, held_out_ids = train_test_split(identifiers, test_size=0.25, random_state=42)
    training_pairs = construct_preference_pairs(frame.loc[ids.isin(training_ids)], mirror=True)
    held_out_pairs = construct_preference_pairs(frame.loc[ids.isin(held_out_ids)], mirror=True)
    features = list(RUBRIC)
    # No intercept means swapping left and right reverses the prediction.
    model = LogisticRegression(fit_intercept=False, max_iter=1000, random_state=42)
    model.fit(training_pairs[features], training_pairs["preferred_left"])
    accuracy = accuracy_score(held_out_pairs["preferred_left"], model.predict(held_out_pairs[features]))
    return PreferenceModelResult(
        model=model,
        accuracy=float(accuracy),
        coefficients=pd.DataFrame({"Criterion": [details["label"] for details in RUBRIC.values()], "Coefficient": model.coef_[0]}),
        training_ids=tuple(training_ids),
        held_out_ids=tuple(held_out_ids),
        training_pair_count=len(training_pairs) // 2,
        held_out_pair_count=len(held_out_pairs) // 2,
        augmented_training_pair_count=len(training_pairs),
        augmented_held_out_pair_count=len(held_out_pairs),
    )


def predict_comparison(
    result: PreferenceModelResult, evaluation: dict[str, Any], left: str = "A", right: str = "B"
) -> dict[str, Any]:
    """Predict a pair using the user's existing scores, not the response text."""
    if left not in CANDIDATES or right not in CANDIDATES or left == right:
        raise ValueError("Choose two different candidates from A, B, and C.")
    # Reuse pair construction to enforce complete, valid ratings and ranking.
    pairs = construct_preference_pairs(pd.DataFrame([evaluation]), mirror=True)
    selected = pairs.loc[(pairs["left_candidate"] == left) & (pairs["right_candidate"] == right)]
    features = selected[list(RUBRIC)]
    probability_left = float(result.model.predict_proba(features)[0, list(result.model.classes_).index(1)])
    predicted_left = bool(result.model.predict(features)[0])
    identifier = str(evaluation["evaluation_id"])
    membership = (
        "Training evaluation" if identifier in result.training_ids
        else "Held-out evaluation" if identifier in result.held_out_ids
        else "New evaluation"
    )
    return {
        "predicted_candidate": left if predicted_left else right,
        "probability_left": probability_left,
        "probability_right": 1.0 - probability_left,
        "human_preferred_candidate": left if int(selected["preferred_left"].iloc[0]) else right,
        "membership": membership,
    }
