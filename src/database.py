"""SQLite persistence for ranking data and separate strategy comparisons."""

from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from src.evaluation import (
    CANDIDATES, COMPARISON_COLUMNS, EVALUATION_COLUMNS, RUBRIC,
    validate_comparison, validate_evaluation,
)


DatabasePath = str | Path | None


def _score_definitions(candidate: str) -> list[str]:
    definitions = [
        f'"{candidate}_{criterion}" INTEGER NOT NULL '
        f'CHECK (typeof("{candidate}_{criterion}") = \'integer\' AND "{candidate}_{criterion}" BETWEEN 1 AND 5)'
        for criterion in RUBRIC
    ]
    definitions.append(
        f'"{candidate}_average_score" REAL NOT NULL '
        f'CHECK ("{candidate}_average_score" BETWEEN 1 AND 5)'
    )
    return definitions


_SCORE_DEFINITIONS = ",\n".join(
    definition for candidate in CANDIDATES for definition in _score_definitions(candidate)
)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL CHECK (datetime(timestamp) IS NOT NULL),
    original_text TEXT NOT NULL CHECK (length(trim(original_text)) > 0),
    rewrite_instruction TEXT NOT NULL CHECK (length(trim(rewrite_instruction)) > 0),
    constraints TEXT NOT NULL,
    candidate_a TEXT NOT NULL CHECK (length(trim(candidate_a)) > 0),
    candidate_b TEXT NOT NULL CHECK (length(trim(candidate_b)) > 0),
    candidate_c TEXT NOT NULL CHECK (length(trim(candidate_c)) > 0),
    {_SCORE_DEFINITIONS},
    first_place TEXT NOT NULL CHECK (first_place IN ('A', 'B', 'C')),
    second_place TEXT NOT NULL CHECK (second_place IN ('A', 'B', 'C')),
    third_place TEXT NOT NULL CHECK (third_place IN ('A', 'B', 'C')),
    rationale TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    constraint_violation_candidate TEXT NOT NULL CHECK (constraint_violation_candidate IN ('A', 'B', 'C', 'None')),
    constraint_violation_explanation TEXT NOT NULL,
    CHECK (first_place != second_place AND first_place != third_place AND second_place != third_place)
);
CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL CHECK (datetime(timestamp) IS NOT NULL),
    original_text TEXT NOT NULL CHECK (length(trim(original_text)) > 0),
    rewrite_instruction TEXT NOT NULL CHECK (length(trim(rewrite_instruction)) > 0),
    constraints TEXT NOT NULL,
    baseline_response TEXT NOT NULL CHECK (length(trim(baseline_response)) > 0),
    improved_response TEXT NOT NULL CHECK (length(trim(improved_response)) > 0),
    preferred_response TEXT NOT NULL CHECK (preferred_response IN ('Baseline', 'Improved', 'Tie')),
    rationale TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    held_out_task_id TEXT UNIQUE CHECK (held_out_task_id IS NULL OR length(trim(held_out_task_id)) > 0),
    protocol_version TEXT NOT NULL CHECK (protocol_version = 'v1')
);
"""


def _resolve_path(path: DatabasePath) -> Path:
    if path is None:
        from src.config import get_settings

        path = get_settings().db_path
    if str(path) == ":memory:":
        raise ValueError("Use a file path so judgments persist across connections.")
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def init_db(path: DatabasePath = None) -> None:
    """Create both tables and the parent directory without deleting existing data."""
    with closing(sqlite3.connect(_resolve_path(path), timeout=30)) as connection:
        with connection:
            connection.executescript(_SCHEMA)


def _insert(table: str, record: dict[str, Any], columns: list[str], path: DatabasePath) -> bool:
    init_db(path)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    # Table/column names are internal constants; every submitted value is bound.
    query = f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    with closing(sqlite3.connect(_resolve_path(path), timeout=30)) as connection:
        with connection:
            cursor = connection.execute(query, [record[column] for column in columns])
            return cursor.rowcount == 1


def save_evaluation(record: Mapping[str, Any], path: DatabasePath = None) -> bool:
    """Save a ranking once; return False for an existing ID without overwriting."""
    return _insert("evaluations", validate_evaluation(record), EVALUATION_COLUMNS, path)


def get_evaluations(path: DatabasePath = None) -> pd.DataFrame:
    """Return all rankings, newest first, with the documented column order."""
    init_db(path)
    columns = ", ".join(f'"{column}"' for column in EVALUATION_COLUMNS)
    with closing(sqlite3.connect(_resolve_path(path), timeout=30)) as connection:
        return pd.read_sql_query(
            f"SELECT {columns} FROM evaluations ORDER BY timestamp DESC, rowid DESC", connection
        )


def get_evaluation(evaluation_id: str, path: DatabasePath = None) -> dict[str, Any] | None:
    """Return one full ranking, or None when its UUID is not present."""
    init_db(path)
    columns = ", ".join(f'"{column}"' for column in EVALUATION_COLUMNS)
    with closing(sqlite3.connect(_resolve_path(path), timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            f"SELECT {columns} FROM evaluations WHERE evaluation_id = ?", (evaluation_id,)
        ).fetchone()
        return dict(row) if row is not None else None


def save_comparison(record: Mapping[str, Any], path: DatabasePath = None) -> bool:
    """Save a comparison once, including at most one vote per held-out task.

    Duplicate comparison IDs or non-null held_out_task_id values return False.
    Existing votes are never overwritten; ordinary experiments with distinct
    UUIDs and no held-out task ID may be saved independently.
    """
    return _insert("comparisons", validate_comparison(record), COMPARISON_COLUMNS, path)


def get_comparisons(path: DatabasePath = None, held_out: bool | None = None) -> pd.DataFrame:
    """Read all comparisons, or only held-out (True) / ordinary (False) records."""
    if held_out is not None and not isinstance(held_out, bool):
        raise ValueError("held_out must be True, False, or None.")
    init_db(path)
    columns = ", ".join(f'"{column}"' for column in COMPARISON_COLUMNS)
    condition = ""
    if held_out is True:
        condition = " WHERE held_out_task_id IS NOT NULL"
    elif held_out is False:
        condition = " WHERE held_out_task_id IS NULL"
    with closing(sqlite3.connect(_resolve_path(path), timeout=30)) as connection:
        return pd.read_sql_query(
            f"SELECT {columns} FROM comparisons{condition} ORDER BY timestamp DESC, rowid DESC",
            connection,
        )
