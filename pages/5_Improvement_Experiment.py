"""Explore a targeted prompt improvement using development tasks."""

import streamlit as st

from src.experiment_ui import render_comparison
from src.experiments import comparison_rates
from src.generation import BASELINE_INSTRUCTIONS, IMPROVED_INSTRUCTIONS, PROTOCOL_VERSION
from src.ui import load_comparisons, settings_or_stop, task_inputs

st.title("Improvement Experiment")
st.write("Evaluate → identify a weakness → make a targeted change → evaluate again.")
st.info("The improved strategy is a fixed, stronger prompt. It emphasizes meaning, explicit constraints, tone, and checking the rewrite. It is not learned from your feedback and is not guaranteed to win.")
with st.expander("Inspect the two generation strategies"):
    st.markdown("**Baseline Strategy**")
    st.text(BASELINE_INSTRUCTIONS)
    st.markdown("**Improved Strategy**")
    st.text(IMPROVED_INSTRUCTIONS)
    st.caption(f"Protocol {PROTOCOL_VERSION}. Both strategies see all task fields, including constraints; the baseline is not deprived of the user's request.")
st.write("Use this page to explore weaknesses from your dataset. Keep the Improvement Test tasks separate while designing your changes.")
task = task_inputs("experiment")
render_comparison(task, settings_or_stop(), "experiment_comparison")
results = load_comparisons(held_out=False)
if not results.empty:
    st.subheader("Development results")
    rates = comparison_rates(results)
    for column, label in zip(st.columns(3), ("Baseline", "Improved", "Tie")):
        column.metric(f"{label} rate", f"{rates[f'{label}_rate']:.0%}")
    st.caption(f"{rates['total']} saved development comparisons. These are exploratory results, not held-out evidence.")
