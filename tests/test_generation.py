"""Check independent candidate calls and the fixed experimental protocol."""

from unittest.mock import Mock

import pytest

from src import generation
from src.config import Settings
from src.generation import (
    BASELINE_INSTRUCTIONS, IMPROVED_INSTRUCTIONS, PROTOCOL_VERSION,
    generate_candidates, generate_strategy_comparison,
)
from src.openai_client import GenerationError
from src.sample_tasks import HELD_OUT_PROTOCOL_VERSION, HELD_OUT_TASKS, SAMPLE_TASKS


@pytest.fixture
def settings():
    return Settings(api_key="test-only-key")


def test_three_independent_candidates_share_task_and_settings(monkeypatch, settings):
    fake = Mock(side_effect=["Candidate one", "Candidate two", "Candidate three"])
    monkeypatch.setattr(generation, "generate_text", fake)
    result = generate_candidates("Original text", "Rewrite politely", "Keep the deadline", settings)
    assert result == {"A": "Candidate one", "B": "Candidate two", "C": "Candidate three"}
    assert fake.call_count == 3
    strategies = set()
    for call in fake.call_args_list:
        assert call.args[:3] == ("Original text", "Rewrite politely", "Keep the deadline")
        assert call.kwargs["settings"] is settings
        strategies.add(call.args[3])
        for answer in result.values():
            assert answer not in str(call)
    assert len(strategies) == 3


def test_duplicate_candidate_gets_one_independent_retry(monkeypatch, settings):
    fake = Mock(side_effect=["First", "First", "Second", "Third"])
    monkeypatch.setattr(generation, "generate_text", fake)
    assert generate_candidates("Original", "Rewrite", settings=settings) == {"A": "First", "B": "Second", "C": "Third"}
    assert fake.call_count == 4
    assert fake.call_args_list[2].args[:3] == fake.call_args_list[1].args[:3]
    assert "First" not in fake.call_args_list[2].args[3]


def test_duplicate_c_is_checked_against_b_as_well_as_a(monkeypatch, settings):
    fake = Mock(side_effect=["First", "Second", "Second", "Third"])
    monkeypatch.setattr(generation, "generate_text", fake)
    assert generate_candidates("Original", "Rewrite", settings=settings)["C"] == "Third"
    assert fake.call_count == 4


def test_persistent_duplicates_fail_without_partial_candidates(monkeypatch, settings):
    fake = Mock(side_effect=["Same", "Same", "Same"])
    monkeypatch.setattr(generation, "generate_text", fake)
    with pytest.raises(GenerationError, match="repeated a candidate"):
        generate_candidates("Original", "Rewrite", settings=settings)
    assert fake.call_count == 3


def test_later_request_failure_is_not_returned_as_partial_success(monkeypatch, settings):
    fake = Mock(side_effect=["First", GenerationError("API unavailable")])
    monkeypatch.setattr(generation, "generate_text", fake)
    with pytest.raises(GenerationError, match="API unavailable"):
        generate_candidates("Original", "Rewrite", settings=settings)


def test_strategy_comparison_changes_only_system_instructions(monkeypatch, settings):
    fake = Mock(side_effect=["Baseline result", "Improved result"])
    monkeypatch.setattr(generation, "generate_text", fake)
    result = generate_strategy_comparison("Original", "Rewrite", "Keep constraints", settings)
    assert result == {"baseline_response": "Baseline result", "improved_response": "Improved result"}
    first, second = fake.call_args_list
    assert first.args[:3] == second.args[:3] == ("Original", "Rewrite", "Keep constraints")
    assert first.args[3] == BASELINE_INSTRUCTIONS
    assert second.args[3] == IMPROVED_INSTRUCTIONS
    assert first.kwargs["settings"] is second.kwargs["settings"] is settings
    assert "Baseline result" not in str(second)
    assert fake.call_count == 2


def test_identical_strategy_outputs_are_a_valid_tie(monkeypatch, settings):
    fake = Mock(return_value="Same rewrite")
    monkeypatch.setattr(generation, "generate_text", fake)
    result = generate_strategy_comparison("Original", "Rewrite", settings=settings)
    assert result["baseline_response"] == result["improved_response"]
    assert fake.call_count == 2


def test_comparison_failure_returns_no_partial_result(monkeypatch, settings):
    monkeypatch.setattr(generation, "generate_text", Mock(side_effect=["Baseline", GenerationError("Failure")]))
    with pytest.raises(GenerationError, match="Failure"):
        generate_strategy_comparison("Original", "Rewrite", settings=settings)


@pytest.mark.parametrize("method", [generate_candidates, generate_strategy_comparison])
def test_empty_input_is_rejected_before_any_generation(monkeypatch, settings, method):
    fake = Mock()
    monkeypatch.setattr(generation, "generate_text", fake)
    with pytest.raises(GenerationError, match="rewrite instruction"):
        method("Original", " ", settings=settings)
    fake.assert_not_called()


def test_fixed_task_libraries_are_complete_distinct_and_versioned():
    assert len(SAMPLE_TASKS) >= 10 and len(HELD_OUT_TASKS) >= 10
    required = {"id", "title", "original_text", "rewrite_instruction", "constraints"}
    all_tasks = SAMPLE_TASKS + HELD_OUT_TASKS
    assert len({task["id"] for task in all_tasks}) == len(all_tasks)
    assert all(required.issubset(task) for task in all_tasks)
    for task in all_tasks:
        assert all(isinstance(task[key], str) and task[key].strip() for key in required - {"constraints"})
        assert isinstance(task["constraints"], str)
    practice_text = {task["original_text"] for task in SAMPLE_TASKS}
    test_text = {task["original_text"] for task in HELD_OUT_TASKS}
    assert practice_text.isdisjoint(test_text)
    assert PROTOCOL_VERSION == HELD_OUT_PROTOCOL_VERSION == "v1"
    assert all(task["protocol_version"] == "v1" for task in HELD_OUT_TASKS)


def test_held_out_examples_do_not_leak_into_frozen_strategy_prompts():
    for task in HELD_OUT_TASKS:
        assert task["original_text"] not in BASELINE_INSTRUCTIONS
        assert task["original_text"] not in IMPROVED_INSTRUCTIONS
        assert task["id"] not in IMPROVED_INSTRUCTIONS
    assert "Preserve" in IMPROVED_INSTRUCTIONS
    assert "constraint" in IMPROVED_INSTRUCTIONS
    assert "unsupported" in IMPROVED_INSTRUCTIONS
    assert "check" in IMPROVED_INSTRUCTIONS
