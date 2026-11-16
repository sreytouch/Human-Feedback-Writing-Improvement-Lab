"""Judge fixed strategies on a separate, versioned set of ten writing tasks."""

import matplotlib.pyplot as plt
import streamlit as st

from src.experiment_ui import render_comparison
from src.experiments import comparison_rates, development_originals, eligible_test_results, normalized_original
from src.generation import PROTOCOL_VERSION
from src.sample_tasks import HELD_OUT_TASKS
from src.ui import load_comparisons, load_evaluations, settings_or_stop, show_task

st.title("Improvement Test")
st.write("Test the same two strategies on ten tasks that are separate from the practice library.")
st.info("Keep these examples out of strategy design. Once you inspect and tune against them, they are no longer unseen. Freeze the strategy before judging; use fresh tasks and a new protocol for later experiments.")
settings = settings_or_stop()
rankings = load_evaluations()
development = load_comparisons(held_out=False)
held_out = load_comparisons(held_out=True)
used_originals = development_originals(rankings, development)
valid = eligible_test_results(held_out, HELD_OUT_TASKS, used_originals, PROTOCOL_VERSION)
rates = comparison_rates(valid)
st.caption(f"Fixed protocol: {PROTOCOL_VERSION} · {len(valid)} / {len(HELD_OUT_TASKS)} eligible tasks judged · One saved judgment per test task")
for column, label in zip(st.columns(3), ("Baseline", "Improved", "Tie")):
    column.metric(f"{label} {'rate' if label == 'Tie' else 'win rate'}", f"{rates[f'{label}_rate']:.0%}" if rates["total"] else "—")
st.caption("Rates use completed, eligible task judgments as the denominator and include ties. With ten tasks these estimates are illustrative, not proof of a general improvement.")
if not valid.empty:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(["Baseline", "Improved", "Tie"], [rates[label] for label in ("Baseline", "Improved", "Tie")], color=["#6B7B93", "#2563EB", "#17A398"])
    ax.set_ylabel("Task judgments")
    ax.set_ylim(0, max(1, max(rates[label] for label in ("Baseline", "Improved", "Tie"))) + 1)
    ax.set_yticks(range(max(rates[label] for label in ("Baseline", "Improved", "Tie")) + 2))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
excluded = len(held_out) - len(valid)
if excluded:
    st.warning(f"{excluded} saved test result(s) are excluded because their task text was also used in development, or their task/protocol is not in this test set.")
index = st.selectbox("Test task", range(len(HELD_OUT_TASKS)), format_func=lambda item: f"{item + 1:02d} · {HELD_OUT_TASKS[item]['title']}")
task = HELD_OUT_TASKS[index]
show_task(task)
existing = held_out.loc[held_out["held_out_task_id"].eq(task["id"])] if not held_out.empty else held_out
if not existing.empty:
    record = existing.iloc[0]
    st.success(f"This task already has a saved judgment: {record['preferred_response']}.")
    left, right = st.columns(2)
    left.markdown("**Baseline Response**")
    left.text(record["baseline_response"])
    right.markdown("**Improved Response**")
    right.text(record["improved_response"])
    st.markdown("**Rationale**")
    st.text(record["rationale"])
elif normalized_original(task["original_text"]) in used_originals:
    st.warning("This original text already appears in your development data. It cannot count as unseen; choose another test task.")
else:
    render_comparison(task, settings, f"test_{task['id']}", held_out_task_id=task["id"])
st.caption("The interface shows strategy labels, so evaluator expectations may influence choices. The test set is an educational example, not a blinded benchmark.")
