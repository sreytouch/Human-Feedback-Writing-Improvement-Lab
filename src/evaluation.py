"""Validate human rankings and comparison judgments, then prepare safe exports."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import math
from numbers import Integral, Real
from typing import Any
from uuid import UUID, uuid4

import pandas as pd


CANDIDATES = ("A", "B", "C")
COMPARISON_PREFERENCES = ("Baseline", "Improved", "Tie")
PROTOCOL_VERSION = "v1"
RUBRIC = {
    "meaning_preservation": {
        "label": "Meaning Preservation",
        "description": "Does the rewrite retain the important meaning of the original?",
    },
    "instruction_following": {
        "label": "Instruction Following",
        "description": "Does it perform the requested transformation?",
    },
    "constraint_following": {
        "label": "Constraint Following",
        "description": "Does it respect explicit restrictions?",
    },
    "clarity": {
        "label": "Clarity",
        "description": "Is it easy to understand?",
    },
    "tone": {
        "label": "Tone",
        "description": "Does it have the requested tone?",
    },
    "conciseness": {
        "label": "Conciseness",
        "description": "Is it appropriately concise?",
    },
    "overall_quality": {
        "label": "Overall Quality",
        "description": "How successful is the rewrite overall?",
    },
}
EVALUATION_COLUMNS = [
    "evaluation_id", "timestamp", "original_text", "rewrite_instruction", "constraints",
    "candidate_a", "candidate_b", "candidate_c",
    *[column for candidate in CANDIDATES for column in (
        *[f"{candidate}_{criterion}" for criterion in RUBRIC],
        f"{candidate}_average_score",
    )],
    "first_place", "second_place", "third_place", "rationale",
    "constraint_violation_candidate", "constraint_violation_explanation",
]
COMPARISON_COLUMNS = [
    "comparison_id", "timestamp", "original_text", "rewrite_instruction", "constraints",
    "baseline_response", "improved_response", "preferred_response", "rationale",
    "held_out_task_id", "protocol_version",
]
SEARCH_COLUMNS = [
    "original_text", "rewrite_instruction", "constraints", "candidate_a", "candidate_b",
    "candidate_c", "rationale", "constraint_violation_explanation",
]


class ValidationError(ValueError):
    """A judgment is incomplete or contains an invalid value."""


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} is required.")
    return value


def _optional_text(value: str | None, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be text.")
    return value


def _identifier(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(value))
    except (ValueError, AttributeError, TypeError) as error:
        raise ValidationError("The record ID must be a valid UUID string.") from error


def _validated_scores(scores: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(scores, Mapping) or set(scores) != set(RUBRIC):
        raise ValidationError("Score all seven rubric criteria, with no extra criteria.")
    validated = {}
    for criterion in RUBRIC:
        value = scores[criterion]
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 5:
            raise ValidationError(f"{RUBRIC[criterion]['label']} must be an integer from 1 to 5.")
        validated[criterion] = int(value)
    return validated


def calculate_average(scores: Mapping[str, int]) -> float:
    """Return the unrounded mean of seven validated integer rubric scores."""
    return sum(_validated_scores(scores).values()) / len(RUBRIC)


def validate_ranking(ranking: Sequence[str]) -> list[str]:
    """Require each of A, B, and C exactly once, in best-to-worst order."""
    if (
        isinstance(ranking, (str, bytes))
        or not isinstance(ranking, Sequence)
        or len(ranking) != 3
        or any(not isinstance(candidate, str) for candidate in ranking)
        or set(ranking) != set(CANDIDATES)
    ):
        raise ValidationError("Rank A, B, and C exactly once; duplicate or missing ranks are not allowed.")
    return list(ranking)


def make_evaluation(
    original_text: str,
    rewrite_instruction: str,
    constraints: str | None,
    candidates: Mapping[str, str],
    scores: Mapping[str, Mapping[str, int]],
    ranking: Sequence[str],
    rationale: str,
    constraint_violation_candidate: str = "None",
    constraint_violation_explanation: str = "",
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """Build one flat, validated ranking record with a UUID and UTC timestamp.

    Text is kept as supplied. Reuse the returned ID when retrying a save so the
    database can recognize the same evaluation rather than count it twice.
    """
    original_text = _required_text(original_text, "Original text")
    rewrite_instruction = _required_text(rewrite_instruction, "Rewrite instruction")
    constraints = _optional_text(constraints, "Constraints")
    rationale = _required_text(rationale, "Rationale")
    if not isinstance(candidates, Mapping) or set(candidates) != set(CANDIDATES):
        raise ValidationError("Provide all three candidates: A, B, and C.")
    if not isinstance(scores, Mapping) or set(scores) != set(CANDIDATES):
        raise ValidationError("Provide rubric scores for A, B, and C.")
    ranking = validate_ranking(ranking)
    if constraint_violation_candidate not in (*CANDIDATES, "None"):
        raise ValidationError("Constraint violation must identify A, B, C, or None.")
    explanation = _optional_text(constraint_violation_explanation, "Constraint violation explanation")
    record = {
        "evaluation_id": _identifier(evaluation_id),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "original_text": original_text,
        "rewrite_instruction": rewrite_instruction,
        "constraints": constraints,
        **{f"candidate_{candidate.lower()}": _required_text(candidates[candidate], f"Candidate {candidate}") for candidate in CANDIDATES},
    }
    for candidate in CANDIDATES:
        ratings = _validated_scores(scores[candidate])
        record.update({f"{candidate}_{criterion}": value for criterion, value in ratings.items()})
        record[f"{candidate}_average_score"] = calculate_average(ratings)
    record.update({
        "first_place": ranking[0], "second_place": ranking[1], "third_place": ranking[2],
        "rationale": rationale, "constraint_violation_candidate": constraint_violation_candidate,
        "constraint_violation_explanation": explanation,
    })
    return record


def make_comparison(
    original_text: str,
    rewrite_instruction: str,
    constraints: str | None,
    baseline_response: str,
    improved_response: str,
    preferred_response: str,
    rationale: str,
    comparison_id: str | None = None,
    held_out_task_id: str | None = None,
) -> dict[str, Any]:
    """Build a baseline/improved judgment, optionally for a fixed held-out task."""
    if preferred_response not in COMPARISON_PREFERENCES:
        raise ValidationError("Choose Baseline, Improved, or Tie.")
    if held_out_task_id is not None:
        held_out_task_id = _required_text(held_out_task_id, "Held-out task ID")
    return {
        "comparison_id": _identifier(comparison_id),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "original_text": _required_text(original_text, "Original text"),
        "rewrite_instruction": _required_text(rewrite_instruction, "Rewrite instruction"),
        "constraints": _optional_text(constraints, "Constraints"),
        "baseline_response": _required_text(baseline_response, "Baseline response"),
        "improved_response": _required_text(improved_response, "Improved response"),
        "preferred_response": preferred_response,
        "rationale": _required_text(rationale, "Rationale"),
        "held_out_task_id": held_out_task_id,
        "protocol_version": PROTOCOL_VERSION,
    }


def _validate_fields(record: Mapping[str, Any], columns: list[str], id_column: str) -> None:
    if not isinstance(record, Mapping) or set(record) != set(columns):
        raise ValidationError("The record must contain exactly the documented fields.")
    if not isinstance(record[id_column], str):
        raise ValidationError("A saved record must have a UUID string as its ID.")


def _validated_timestamp(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("UTC timezone required.")
    except (ValueError, AttributeError, TypeError) as error:
        raise ValidationError("Timestamp must be an ISO 8601 date and time in UTC.") from error
    return parsed.isoformat(timespec="microseconds")


def validate_evaluation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate a complete ranking before storage, including derived averages."""
    _validate_fields(record, EVALUATION_COLUMNS, "evaluation_id")
    validated = make_evaluation(
        original_text=record["original_text"], rewrite_instruction=record["rewrite_instruction"],
        constraints=record["constraints"],
        candidates={candidate: record[f"candidate_{candidate.lower()}"] for candidate in CANDIDATES},
        scores={candidate: {criterion: record[f"{candidate}_{criterion}"] for criterion in RUBRIC} for candidate in CANDIDATES},
        ranking=[record["first_place"], record["second_place"], record["third_place"]],
        rationale=record["rationale"], constraint_violation_candidate=record["constraint_violation_candidate"],
        constraint_violation_explanation=record["constraint_violation_explanation"],
        evaluation_id=record["evaluation_id"],
    )
    validated["timestamp"] = _validated_timestamp(record["timestamp"])
    for candidate in CANDIDATES:
        supplied = record[f"{candidate}_average_score"]
        expected = validated[f"{candidate}_average_score"]
        if isinstance(supplied, bool) or not isinstance(supplied, Real) or not math.isclose(
            supplied, expected, rel_tol=0, abs_tol=1e-9
        ):
            raise ValidationError(f"Candidate {candidate}'s average must match its rubric scores.")
    return validated


def validate_comparison(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a comparison before writing it, preserving its original time."""
    _validate_fields(record, COMPARISON_COLUMNS, "comparison_id")
    if record["protocol_version"] != PROTOCOL_VERSION:
        raise ValidationError(f"Comparison protocol_version must be {PROTOCOL_VERSION}.")
    validated = make_comparison(
        original_text=record["original_text"], rewrite_instruction=record["rewrite_instruction"],
        constraints=record["constraints"], baseline_response=record["baseline_response"],
        improved_response=record["improved_response"], preferred_response=record["preferred_response"],
        rationale=record["rationale"], comparison_id=record["comparison_id"],
        held_out_task_id=record["held_out_task_id"],
    )
    validated["timestamp"] = _validated_timestamp(record["timestamp"])
    return validated


def _as_date(value: date | str | None, label: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as error:
        raise ValidationError(f"{label} must be a date in YYYY-MM-DD format.") from error


def filter_evaluations(
    frame: pd.DataFrame,
    preferred_response: str | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Filter first-place candidate, inclusive UTC dates, and literal text."""
    if preferred_response is not None and preferred_response not in CANDIDATES:
        raise ValidationError("First-place filter must be A, B, C, or None.")
    start = _as_date(start_date, "Start date")
    end = _as_date(end_date, "End date")
    if start and end and start > end:
        raise ValidationError("Start date must be on or before end date.")
    if not isinstance(search, str):
        raise ValidationError("Search must be text.")
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    if preferred_response is not None:
        mask &= frame["first_place"].eq(preferred_response)
    if start is not None or end is not None:
        dates = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="ISO8601").dt.date
        if start is not None:
            mask &= dates.ge(start)
        if end is not None:
            mask &= dates.le(end)
    if search:
        matches = pd.Series(False, index=frame.index)
        for column in SEARCH_COLUMNS:
            matches |= frame[column].fillna("").astype(str).str.contains(search, case=False, regex=False)
        mask &= matches
    return frame.loc[mask].copy()


def export_csv(frame: pd.DataFrame) -> bytes:
    """Export all columns as UTF-8 CSV, without changing stored values.

    Formula-like strings beginning with =, +, -, or @ (even after whitespace),
    and strings beginning with tab/CR/LF, receive a leading apostrophe to prevent
    spreadsheet execution. Quotes, commas, and newlines use standard CSV quoting.
    The dataframe and database remain unchanged; account for the apostrophe when
    consuming this export as a machine-learning dataset.
    """
    def escape_formula(value: Any) -> Any:
        if isinstance(value, str) and (
            value.lstrip().startswith(("=", "+", "-", "@"))
            or value.startswith(("\t", "\r", "\n"))
        ):
            return "'" + value
        return value

    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(escape_formula)
    return safe.to_csv(index=False, lineterminator="\n").encode("utf-8")
