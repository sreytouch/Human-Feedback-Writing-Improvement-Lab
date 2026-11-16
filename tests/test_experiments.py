"""Protect the separation between development feedback and held-out results."""

import pandas as pd

from src.experiments import comparison_rates, development_originals, eligible_test_results


def test_development_texts_are_normalized_across_both_collections():
    rankings = pd.DataFrame({"original_text": [" Keep  the DEADLINE. "]})
    comparisons = pd.DataFrame({"original_text": ["Call\nme now."]})
    assert development_originals(rankings, comparisons) == {"keep the deadline.", "call me now."}


def test_held_out_results_exclude_reused_unknown_and_wrong_protocol_tasks():
    tasks = [{"id": "one", "original_text": "New task"}, {"id": "two", "original_text": "Reused text"}]
    rows = pd.DataFrame([
        {"held_out_task_id": "one", "protocol_version": "v1", "preferred_response": "Tie"},
        {"held_out_task_id": "one", "protocol_version": "v1", "preferred_response": "Baseline"},
        {"held_out_task_id": "two", "protocol_version": "v1", "preferred_response": "Improved"},
        {"held_out_task_id": "unknown", "protocol_version": "v1", "preferred_response": "Improved"},
        {"held_out_task_id": "one", "protocol_version": "v2", "preferred_response": "Improved"},
    ])
    eligible = eligible_test_results(rows, tasks, {"reused text"}, "v1")
    assert len(eligible) == 1
    assert eligible.iloc[0]["preferred_response"] == "Tie"
    assert len(rows) == 5


def test_rates_include_ties_and_support_an_empty_dataset():
    rates = comparison_rates(pd.DataFrame({"preferred_response": ["Baseline", "Improved", "Tie", "Tie"]}))
    assert rates["total"] == 4
    assert rates["Baseline_rate"] == rates["Improved_rate"] == 0.25
    assert rates["Tie_rate"] == 0.5
    assert comparison_rates(pd.DataFrame())["Tie_rate"] == 0
    assert eligible_test_results(pd.DataFrame(), [], set(), "v1").empty
