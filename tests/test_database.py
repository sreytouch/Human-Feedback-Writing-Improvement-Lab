"""Check durable, isolated storage for rankings and strategy comparisons."""

import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.database import (
    get_comparisons, get_evaluation, get_evaluations, init_db,
    save_comparison, save_evaluation,
)
from src.evaluation import (
    CANDIDATES, COMPARISON_COLUMNS, EVALUATION_COLUMNS, RUBRIC,
    ValidationError, make_comparison, make_evaluation,
)


@pytest.fixture
def ranking():
    return make_evaluation(
        original_text="I need your answer today.", rewrite_instruction="Make it more polite.",
        constraints="Keep the deadline.", candidates={
            "A": "Could you please reply today?", "B": "Reply whenever you can.", "C": "Please reply today.",
        }, scores={candidate: dict.fromkeys(RUBRIC, 4) for candidate in CANDIDATES},
        ranking=["A", "C", "B"], rationale="A preserves the deadline politely.",
        constraint_violation_candidate="B", constraint_violation_explanation="The deadline was lost.",
    )


@pytest.fixture
def comparison():
    return make_comparison(
        original_text="Call me ASAP.", rewrite_instruction="Make this polite.",
        constraints="Preserve urgency.", baseline_response="Call me whenever you can.",
        improved_response="Could you please call me as soon as possible?",
        preferred_response="Improved", rationale="It preserves the urgency.",
    )


def test_initialization_creates_both_tables_and_parent_directory(tmp_path):
    path = tmp_path / "nested" / "feedback.db"
    init_db(path)
    init_db(path)
    assert path.is_file()
    assert get_evaluations(path).empty
    assert get_comparisons(path).empty
    assert list(get_evaluations(path).columns) == EVALUATION_COLUMNS
    assert list(get_comparisons(path).columns) == COMPARISON_COLUMNS
    with sqlite3.connect(path) as connection:
        for table, expected in (("evaluations", EVALUATION_COLUMNS), ("comparisons", COMPARISON_COLUMNS)):
            columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
            assert [column[1] for column in columns] == expected


def test_save_initializes_storage_and_round_trips_every_ranking_field(tmp_path, ranking):
    path = tmp_path / "feedback.db"
    assert save_evaluation(ranking, path) is True
    assert get_evaluation(ranking["evaluation_id"], path) == ranking
    assert get_evaluations(path).to_dict("records") == [ranking]
    assert get_evaluation(str(uuid4()), path) is None
    assert get_comparisons(path).empty


def test_duplicate_ranking_id_does_not_add_or_overwrite_a_judgment(tmp_path, ranking):
    path = tmp_path / "feedback.db"
    assert save_evaluation(ranking, path)
    assert save_evaluation(ranking, path) is False
    assert save_evaluation({**ranking, "rationale": "An edited rationale."}, path) is False
    assert get_evaluation(ranking["evaluation_id"], path) == ranking
    assert len(get_evaluations(path)) == 1


def test_comparison_is_separate_durable_and_idempotent(tmp_path, comparison):
    path = tmp_path / "feedback.db"
    assert save_comparison(comparison, path)
    assert save_comparison(comparison, path) is False
    assert get_comparisons(path).to_dict("records") == [comparison]
    assert get_evaluations(path).empty


def test_held_out_task_allows_only_one_vote_even_with_a_new_uuid(tmp_path, comparison):
    path = tmp_path / "feedback.db"
    first = {**comparison, "held_out_task_id": "held-out-01"}
    second = {**first, "comparison_id": str(uuid4()), "preferred_response": "Baseline"}
    assert save_comparison(first, path) is True
    assert save_comparison(second, path) is False
    assert get_comparisons(path, held_out=True).to_dict("records") == [first]


def test_ordinary_experiments_can_be_repeated_and_held_out_filter_is_exact(tmp_path, comparison):
    path = tmp_path / "feedback.db"
    assert save_comparison(comparison, path)
    assert save_comparison({**comparison, "comparison_id": str(uuid4())}, path)
    assert save_comparison({**comparison, "comparison_id": str(uuid4()), "held_out_task_id": "held-out-02"}, path)
    assert len(get_comparisons(path)) == 3
    assert len(get_comparisons(path, held_out=False)) == 2
    assert len(get_comparisons(path, held_out=True)) == 1
    with pytest.raises(ValueError):
        get_comparisons(path, held_out="false")


def test_records_are_returned_newest_first_even_when_saved_out_of_order(tmp_path, ranking, comparison):
    path = tmp_path / "feedback.db"
    newest = "2026-11-13T21:00:00.000000+00:00"
    oldest = "2026-11-12T21:00:00.000000+00:00"
    for save, record, id_column, read in (
        (save_evaluation, ranking, "evaluation_id", get_evaluations),
        (save_comparison, comparison, "comparison_id", get_comparisons),
    ):
        recent = {**record, "timestamp": newest}
        earlier = {**record, id_column: str(uuid4()), "timestamp": oldest}
        assert save(recent, path)
        assert save(earlier, path)
        assert read(path)[id_column].tolist() == [recent[id_column], earlier[id_column]]


def test_invalid_records_are_rejected_before_storage_is_created(tmp_path, ranking, comparison):
    path = tmp_path / "feedback.db"
    for changes in ({"second_place": "A"}, {"A_tone": 6}, {"A_average_score": 1}, {"rationale": " "}, {"evaluation_id": None}):
        with pytest.raises(ValidationError):
            save_evaluation({**ranking, **changes}, path)
    with pytest.raises(ValidationError):
        save_comparison({**comparison, "preferred_response": "A"}, path)
    assert not path.exists()


def test_sqlite_also_enforces_scores_rankings_and_preferences(tmp_path, ranking, comparison):
    path = tmp_path / "feedback.db"
    save_evaluation(ranking, path)
    save_comparison(comparison, path)
    with sqlite3.connect(path) as connection:
        for statement in (
            "UPDATE evaluations SET A_tone = 6",
            "UPDATE evaluations SET B_clarity = 1.5",
            "UPDATE evaluations SET second_place = first_place",
            "UPDATE evaluations SET constraint_violation_candidate = 'D'",
            "UPDATE comparisons SET preferred_response = 'A'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)
    assert get_evaluation(ranking["evaluation_id"], path) == ranking


def test_submitted_sql_text_does_not_change_the_schema(tmp_path, ranking, comparison):
    path = tmp_path / "feedback.db"
    text = "Robert'); DROP TABLE evaluations; --"
    ranking["original_text"] = text
    comparison["rationale"] = text
    save_evaluation(ranking, path)
    save_comparison(comparison, path)
    assert get_evaluation(ranking["evaluation_id"], path)["original_text"] == text
    assert get_comparisons(path).iloc[0]["rationale"] == text
    assert get_evaluation("' OR 1=1 --", path) is None


def test_default_path_is_read_from_settings(tmp_path, monkeypatch, ranking, comparison):
    path = tmp_path / "configured.db"
    monkeypatch.setattr("src.config.get_settings", lambda: SimpleNamespace(db_path=path))
    assert save_evaluation(ranking)
    assert save_comparison(comparison)
    assert get_evaluation(ranking["evaluation_id"]) == ranking
    assert len(get_evaluations()) == len(get_comparisons()) == 1
    assert path.is_file()
