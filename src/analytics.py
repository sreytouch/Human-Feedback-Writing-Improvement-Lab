"""Summaries of saved rewrite evaluations, including evaluation-level violations."""

from __future__ import annotations

import pandas as pd

from src.evaluation import RUBRIC

CANDIDATES = ("A", "B", "C")


def first_place_counts(frame: pd.DataFrame) -> pd.Series:
    """Return all candidate labels, including labels with no wins."""
    if frame.empty:
        return pd.Series(0, index=CANDIDATES, name="First-place count", dtype=int)
    return frame["first_place"].value_counts().reindex(CANDIDATES, fill_value=0)


def rubric_averages(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each candidate's average rating in every rubric category."""
    return pd.DataFrame(
        {
            candidate: [float(frame[f"{candidate}_{key}"].mean()) if not frame.empty else 0.0 for key in RUBRIC]
            for candidate in CANDIDATES
        },
        index=[details["label"] for details in RUBRIC.values()],
    )


def violation_mask(frame: pd.DataFrame) -> pd.Series:
    """Count an evaluation once if the evaluator flagged any candidate."""
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["constraint_violation_candidate"].isin(CANDIDATES)


def dashboard_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute rates using evaluations as the denominator, never candidates."""
    total = len(frame)
    counts = first_place_counts(frame)
    averages = rubric_averages(frame)
    combined = averages.mean(axis=1)
    violation_count = int(violation_mask(frame).sum())
    result: dict[str, float | int | str] = {
        "total": total,
        "average_score": float(combined.mean()) if total else 0.0,
        "violation_count": violation_count,
        "violation_rate": 100.0 * violation_count / total if total else 0.0,
        "weakest_criterion": "No evaluations yet",
    }
    if total:
        weakest = combined.min()
        result["weakest_criterion"] = ", ".join(combined.index[(combined - weakest).abs() < 1e-9])
    for candidate in CANDIDATES:
        result[f"first_percent_{candidate}"] = 100.0 * int(counts[candidate]) / total if total else 0.0
        result[f"average_{candidate}"] = float(averages[candidate].mean()) if total else 0.0
    for key, details in RUBRIC.items():
        result[f"average_{key}"] = float(combined.loc[details["label"]]) if total else 0.0
    return result
