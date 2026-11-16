"""Check aggregate meanings and denominators, including empty data."""

import pandas as pd
import pytest

from src.analytics import dashboard_summary, first_place_counts, rubric_averages
from src.evaluation import RUBRIC


def evaluation(first="A", violation="None", score=4):
    record = {"first_place": first, "constraint_violation_candidate": violation}
    for candidate in ("A", "B", "C"):
        for criterion in RUBRIC:
            record[f"{candidate}_{criterion}"] = score
    return record


def test_empty_dashboard():
    frame = pd.DataFrame()
    summary = dashboard_summary(frame)
    assert summary["total"] == 0
    assert summary["violation_rate"] == 0
    assert summary["weakest_criterion"] == "No evaluations yet"
    assert first_place_counts(frame).to_dict() == {"A": 0, "B": 0, "C": 0}
    assert rubric_averages(frame).shape == (7, 3)


def test_violation_rate_counts_evaluations_not_candidates():
    frame = pd.DataFrame([
        evaluation("A", "B", 5), evaluation("A", "C", 3),
        evaluation("B", "None", 4), evaluation("C", None, 4),
    ])
    summary = dashboard_summary(frame)
    assert summary["total"] == 4
    assert summary["violation_count"] == 2
    assert summary["violation_rate"] == 50
    assert summary["first_percent_A"] == 50
    assert summary["first_percent_B"] == 25
    assert summary["first_percent_C"] == 25
    assert summary["average_score"] == 4
    assert summary["average_meaning_preservation"] == 4
    assert summary["average_instruction_following"] == 4


def test_weakest_criterion_averages_across_all_candidates():
    record = evaluation()
    for candidate in ("A", "B", "C"):
        record[f"{candidate}_clarity"] = 1
    summary = dashboard_summary(pd.DataFrame([record]))
    assert summary["weakest_criterion"] == RUBRIC["clarity"]["label"]
    assert summary["average_score"] == pytest.approx(25 / 7)
    averages = rubric_averages(pd.DataFrame([record]))
    assert list(averages.loc[RUBRIC["clarity"]["label"]]) == [1, 1, 1]


def test_tied_weak_criteria_are_both_named():
    record = evaluation()
    for candidate in ("A", "B", "C"):
        record[f"{candidate}_clarity"] = 2
        record[f"{candidate}_tone"] = 2
    summary = dashboard_summary(pd.DataFrame([record]))
    assert RUBRIC["clarity"]["label"] in summary["weakest_criterion"]
    assert RUBRIC["tone"]["label"] in summary["weakest_criterion"]
