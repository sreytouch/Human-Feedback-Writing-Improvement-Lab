"""Validate complete human judgments and export them without network access."""

import csv
from datetime import date, datetime
import io
from uuid import UUID, uuid4

import pandas as pd
import pytest

from src.evaluation import (
    CANDIDATES, COMPARISON_COLUMNS, EVALUATION_COLUMNS, RUBRIC, ValidationError,
    calculate_average, export_csv, filter_evaluations, make_comparison,
    make_evaluation, validate_comparison, validate_evaluation,
)


def ranking_record(**changes):
    values = {
        "original_text": "I need your answer today.",
        "rewrite_instruction": "Make it more polite.", "constraints": "Keep the deadline.",
        "candidates": {"A": "Could you please reply today?", "B": "Please reply whenever you can.", "C": "Please answer today."},
        "scores": {candidate: dict.fromkeys(RUBRIC, score) for candidate, score in zip(CANDIDATES, [5, 2, 4])},
        "ranking": ["A", "C", "B"], "rationale": "A is polite and keeps the deadline.",
        "constraint_violation_candidate": "B", "constraint_violation_explanation": "B drops today's deadline.",
    }
    values.update(changes)
    return make_evaluation(**values)


def comparison_record(**changes):
    values = {
        "original_text": "Call me ASAP.", "rewrite_instruction": "Make it polite.",
        "constraints": "Preserve urgency.", "baseline_response": "Call me when you can.",
        "improved_response": "Could you please call me as soon as possible?",
        "preferred_response": "Improved", "rationale": "It keeps the urgency.",
    }
    values.update(changes)
    return make_comparison(**values)


def test_average_uses_all_seven_criteria_without_rounding():
    assert calculate_average(dict(zip(RUBRIC, [1, 2, 3, 4, 5, 4, 5]))) == pytest.approx(24 / 7)


@pytest.mark.parametrize("value", [0, 6, -1, 3.0, True, "4", None])
def test_ratings_must_be_integers_between_one_and_five(value):
    scores = dict.fromkeys(RUBRIC, 3)
    scores["clarity"] = value
    with pytest.raises(ValidationError):
        calculate_average(scores)


@pytest.mark.parametrize("scores", [{}, {"clarity": 3}, {**dict.fromkeys(RUBRIC, 3), "extra": 5}])
def test_all_and_only_the_rubric_criteria_are_required(scores):
    with pytest.raises(ValidationError):
        calculate_average(scores)


def test_ranking_record_has_exact_columns_uuid_utc_time_and_averages():
    record = ranking_record()
    assert list(record) == EVALUATION_COLUMNS
    assert str(UUID(record["evaluation_id"])) == record["evaluation_id"]
    assert datetime.fromisoformat(record["timestamp"]).utcoffset().total_seconds() == 0
    assert [record[f"{candidate}_average_score"] for candidate in CANDIDATES] == [5, 2, 4]
    assert [record[key] for key in ("first_place", "second_place", "third_place")] == ["A", "C", "B"]
    assert validate_evaluation(record) == record


@pytest.mark.parametrize("field", ["original_text", "rewrite_instruction", "rationale"])
@pytest.mark.parametrize("value", ["", " \n\t", None])
def test_required_evaluation_text_cannot_be_blank(field, value):
    with pytest.raises(ValidationError):
        ranking_record(**{field: value})


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_each_candidate_and_each_candidate_score_is_required(candidate):
    candidates = dict.fromkeys(CANDIDATES, "A rewrite")
    candidates[candidate] = " "
    with pytest.raises(ValidationError):
        ranking_record(candidates=candidates)
    scores = {name: dict.fromkeys(RUBRIC, 3) for name in CANDIDATES}
    del scores[candidate]["tone"]
    with pytest.raises(ValidationError):
        ranking_record(scores=scores)


@pytest.mark.parametrize("changes", [
    {"candidates": {}}, {"scores": {}}, {"ranking": ["A", "A", "B"]},
    {"constraint_violation_candidate": "D"}, {"constraints": 12},
    {"constraint_violation_explanation": 2}, {"evaluation_id": "bad-id"},
])
def test_invalid_evaluation_inputs_are_explained(changes):
    with pytest.raises(ValidationError):
        ranking_record(**changes)


def test_optional_text_and_reused_uuid_are_supported():
    identifier = str(uuid4())
    record = ranking_record(
        constraints=None, constraint_violation_candidate="None",
        constraint_violation_explanation="", evaluation_id=identifier,
    )
    assert record["constraints"] == ""
    assert record["constraint_violation_explanation"] == ""
    assert record["evaluation_id"] == identifier


@pytest.mark.parametrize("changes", [
    {"A_average_score": 1}, {"C_average_score": float("nan")},
    {"timestamp": None}, {"timestamp": "2026-11-13"},
    {"timestamp": "2026-11-13T15:00:00-08:00"}, {"evaluation_id": None},
    {"second_place": "A"}, {"unexpected": "field"},
])
def test_complete_ranking_record_is_revalidated_before_storage(changes):
    with pytest.raises(ValidationError):
        validate_evaluation({**ranking_record(), **changes})


def test_comparison_schema_and_optional_held_out_id():
    record = comparison_record(held_out_task_id="held-out-01")
    assert list(record) == COMPARISON_COLUMNS
    assert str(UUID(record["comparison_id"])) == record["comparison_id"]
    assert record["held_out_task_id"] == "held-out-01"
    assert record["protocol_version"] == "v1"
    assert validate_comparison(record) == record
    assert comparison_record()["held_out_task_id"] is None


@pytest.mark.parametrize("preference", ["Baseline", "Improved", "Tie"])
def test_each_comparison_preference_is_supported(preference):
    assert comparison_record(preferred_response=preference)["preferred_response"] == preference


@pytest.mark.parametrize("changes", [
    {"original_text": " "}, {"rewrite_instruction": ""}, {"baseline_response": ""},
    {"improved_response": None}, {"preferred_response": "A"}, {"rationale": " "},
    {"held_out_task_id": ""}, {"held_out_task_id": 1}, {"comparison_id": "bad-id"},
])
def test_comparison_requires_task_responses_choice_and_rationale(changes):
    with pytest.raises(ValidationError):
        comparison_record(**changes)


@pytest.mark.parametrize("changes", [
    {"comparison_id": None}, {"timestamp": "invalid"},
    {"protocol_version": "v2"}, {"unexpected": "field"},
])
def test_complete_comparison_record_is_revalidated(changes):
    with pytest.raises(ValidationError):
        validate_comparison({**comparison_record(), **changes})


@pytest.fixture
def history():
    rows = [
        ranking_record(original_text="Find [literal] text."),
        ranking_record(constraints="Keep [LITERAL] wording", ranking=["B", "A", "C"]),
        ranking_record(ranking=["C", "B", "A"]),
    ]
    for row, timestamp in zip(rows, [
        "2026-11-12T23:59:59+00:00", "2026-11-13T00:00:00+00:00", "2026-11-13T23:59:59+00:00",
    ]):
        row["timestamp"] = timestamp
    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)


def test_filters_combine_first_place_dates_and_literal_case_insensitive_search(history):
    selected = filter_evaluations(
        history, preferred_response="B", start_date=date(2026, 11, 13),
        end_date="2026-11-13", search="[literal]",
    )
    assert selected["first_place"].tolist() == ["B"]
    assert len(filter_evaluations(history, search="[literal]")) == 2
    assert len(filter_evaluations(history, start_date="2026-11-13", end_date="2026-11-13")) == 2
    selected.loc[:, "original_text"] = "changed"
    assert "changed" not in history["original_text"].tolist()


@pytest.mark.parametrize("column", [
    "original_text", "rewrite_instruction", "constraints", "candidate_a", "candidate_b",
    "candidate_c", "rationale", "constraint_violation_explanation",
])
def test_search_covers_every_free_text_field(history, column):
    history.loc[0, column] = "A needle.* to find"
    assert len(filter_evaluations(history, search="needle.*")) == 1


def test_empty_history_keeps_columns_and_invalid_filters_raise_clear_errors():
    empty = pd.DataFrame(columns=EVALUATION_COLUMNS)
    assert list(filter_evaluations(empty).columns) == EVALUATION_COLUMNS
    with pytest.raises(ValidationError):
        filter_evaluations(empty, start_date="2026-11-14", end_date="2026-11-13")
    with pytest.raises(ValidationError):
        filter_evaluations(empty, start_date="invalid")
    with pytest.raises(ValidationError):
        filter_evaluations(empty, preferred_response="All")


def test_csv_preserves_all_fields_unicode_quotes_and_multiline_text(history):
    original = 'A "quoted", café rewrite\nwith a second line.'
    history.loc[0, "candidate_a"] = original
    reader = csv.DictReader(io.StringIO(export_csv(history).decode("utf-8")))
    rows = list(reader)
    assert reader.fieldnames == EVALUATION_COLUMNS
    assert rows[0]["candidate_a"] == original
    assert rows[0]["A_tone"] == "5"
    assert len(rows) == 3


@pytest.mark.parametrize("value", ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "  =1+1", "\ttext", "\ntext", "\rtext"])
def test_csv_neutralizes_formulas_without_mutating_the_dataframe(value):
    frame = pd.DataFrame([ranking_record(original_text=value)])
    exported = next(csv.DictReader(io.StringIO(export_csv(frame).decode("utf-8"))))
    assert exported["original_text"] == "'" + value
    assert frame.loc[0, "original_text"] == value


def test_empty_csv_export_keeps_the_header():
    encoded = export_csv(pd.DataFrame(columns=EVALUATION_COLUMNS))
    assert encoded.decode("utf-8").strip().split(",") == EVALUATION_COLUMNS
