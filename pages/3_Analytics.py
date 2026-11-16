"""Patterns in human rankings and scores, summarized without API calls."""

import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.analytics import dashboard_summary, first_place_counts, rubric_averages
from src.ui import load_evaluations

st.title("Analytics")
st.write("See which rewrites people prefer, where ratings are weakest, and how often an important constraint was missed.")
frame = load_evaluations()
summary = dashboard_summary(frame)
columns = st.columns(4)
columns[0].metric("Evaluations", summary["total"])
for column, label in zip(columns[1:], ("A", "B", "C")):
    column.metric(f"Candidate {label} ranked first", f"{summary[f'first_percent_{label}']:.1f}%")

if frame.empty:
    st.info("Save your first ranking on Generate & Evaluate to start collecting these insights.")
    st.stop()

columns = st.columns(3)
columns[0].metric("Average rubric score", f"{summary['average_score']:.2f} / 5")
columns[1].metric("Flagged evaluations", summary["violation_count"])
columns[2].metric("Constraint-violation rate", f"{summary['violation_rate']:.1f}%")
st.caption("Violation rate = evaluations with a flagged candidate ÷ all saved evaluations. Each evaluation can flag one candidate, so this is not a count of every individual violation.")
columns = st.columns(2)
columns[0].metric("Meaning preservation", f"{summary['average_meaning_preservation']:.2f} / 5")
columns[1].metric("Instruction following", f"{summary['average_instruction_following']:.2f} / 5")
st.write(f"**Lowest-rated criterion across candidates:** {summary['weakest_criterion']}")
st.caption("Weakest means lowest mean rating across all candidates and saved evaluations; tied criteria are all listed. It does not count individual failures.")

left, right = st.columns(2)
with left:
    st.subheader("First-place counts")
    counts = first_place_counts(frame)
    fig = Figure(figsize=(5.5, 4.2))
    ax = fig.subplots()
    ax.bar([f"Candidate {label}" for label in counts.index], counts.values, color=["#315DC1", "#238A7B", "#9560B6"])
    ax.set_ylabel("Evaluations")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(1.5, float(counts.max()) * 1.25))
    for index, count in enumerate(counts.values):
        ax.annotate(str(count), (index, count), xytext=(0, 5), textcoords="offset points", ha="center")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
with right:
    st.subheader("Ratings by criterion")
    averages = rubric_averages(frame)
    fig = Figure(figsize=(6.5, 4.6))
    ax = fig.subplots()
    averages.plot.barh(ax=ax, color=["#315DC1", "#238A7B", "#9560B6"])
    ax.set_xlim(0, 5)
    ax.set_xlabel("Mean human rating (1–5)")
    ax.legend(title="Candidate")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
with st.expander("View the chart data"):
    st.dataframe(averages.round(2), width="stretch")
    st.dataframe(counts.rename_axis("Candidate").reset_index(name="First-place count"), hide_index=True, width="stretch")
st.info("Use these patterns to propose a specific improvement, then evaluate it on fresh tasks. Candidate labels identify positions, not different underlying models; first-place rates alone do not prove one model or strategy is better.")
