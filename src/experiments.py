"""Small, testable helpers for the separate strategy experiment datasets."""

import pandas as pd

TASK_FIELDS = ("original_text", "rewrite_instruction", "constraints")
OUTCOMES = ("Baseline", "Improved", "Tie")


def normalized_original(text: str) -> str:
    """Ignore casing and repeated whitespace when spotting reused test texts."""
    return " ".join(text.casefold().split())


def development_originals(evaluations: pd.DataFrame, comparisons: pd.DataFrame) -> set[str]:
    texts = set()
    for frame in (evaluations, comparisons):
        if not frame.empty:
            texts.update(normalized_original(str(value)) for value in frame["original_text"])
    return texts


def eligible_test_results(frame: pd.DataFrame, tasks: list[dict], used_originals: set[str], protocol: str) -> pd.DataFrame:
    """Exclude results for unknown, reused, or differently versioned test tasks."""
    if frame.empty:
        return frame.copy()
    eligible = {task["id"] for task in tasks if normalized_original(task["original_text"]) not in used_originals}
    return frame.loc[frame["held_out_task_id"].isin(eligible) & frame["protocol_version"].eq(protocol)].drop_duplicates("held_out_task_id").copy()


def comparison_rates(frame: pd.DataFrame) -> dict[str, float | int]:
    """Use every completed judgment, including ties, as the rate denominator."""
    total = len(frame)
    result = {"total": total}
    for label in OUTCOMES:
        count = int(frame["preferred_response"].eq(label).sum()) if total else 0
        result[label] = count
        result[f"{label}_rate"] = count / total if total else 0.0
    return result
