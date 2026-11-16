"""Inspect and export ranked evaluations and separate strategy comparisons."""

import pandas as pd
import streamlit as st

from src.evaluation import RUBRIC, export_csv
from src.ui import load_comparisons, load_evaluations, show_task

st.title("Dataset")
collection = st.radio("Dataset", ["Ranked rewrites", "Development comparisons", "Held-out comparisons"], horizontal=True)
is_ranked = collection == "Ranked rewrites"
frame = load_evaluations() if is_ranked else load_comparisons(held_out=collection == "Held-out comparisons")
if frame.empty:
    st.info("No saved records in this dataset yet. Generate responses and save your judgment on the corresponding page.")
    st.stop()
choice_column = "first_place" if is_ranked else "preferred_response"
id_column = "evaluation_id" if is_ranked else "comparison_id"
choice = st.selectbox("Top candidate" if is_ranked else "Preferred strategy", ["All", *(list("ABC") if is_ranked else ["Baseline", "Improved", "Tie"])])
search = st.text_input("Search text", placeholder="Search original text, instructions, responses, or rationale")
mask = pd.Series(True, index=frame.index)
if choice != "All":
    mask &= frame[choice_column].eq(choice)
if search:
    text_columns = [column for column in frame.columns if column in {"original_text", "rewrite_instruction", "constraints", "candidate_a", "candidate_b", "candidate_c", "baseline_response", "improved_response", "rationale", "constraint_violation_explanation"}]
    search_mask = pd.Series(False, index=frame.index)
    for column in text_columns:
        search_mask |= frame[column].fillna("").astype(str).str.contains(search, case=False, regex=False)
    mask &= search_mask
if st.checkbox("Filter by date (UTC)"):
    dates = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601").dt.date
    limits = st.date_input("Date range", value=(dates.min(), dates.max()), min_value=dates.min(), max_value=dates.max())
    if len(limits) != 2:
        st.info("Select both a start date and an end date.")
        st.stop()
    mask &= dates.ge(limits[0]) & dates.le(limits[1])
filtered = frame.loc[mask]
st.caption(f"{len(filtered)} / {len(frame)} records · UTC timestamps")
st.dataframe(filtered, hide_index=True, width="stretch")
st.download_button("Download as CSV", export_csv(filtered), file_name=collection.lower().replace(" ", "_") + ".csv", mime="text/csv", disabled=filtered.empty)
st.caption("Exports include all fields for the filtered records. Formula-like text receives an apostrophe for spreadsheet safety; database values are unchanged.")
if filtered.empty:
    st.info("No records match these filters.")
    st.stop()
records = {row[id_column]: row for row in filtered.to_dict("records")}
selected = st.selectbox("Inspect a record", list(records), format_func=lambda key: f"{records[key]['timestamp'][:19]} · {records[key]['original_text'][:65]} · {key[:8]}")
record = records[selected]
show_task(record)
if is_ranked:
    for label, column in zip("ABC", st.columns(3)):
        column.markdown(f"**Candidate {label}**")
        column.text(record[f"candidate_{label.lower()}"])
        column.metric("Mean rubric score", f"{record[f'{label}_average_score']:.2f} / 5")
    st.dataframe(pd.DataFrame({"Criterion": [details["label"] for details in RUBRIC.values()], **{label: [record[f"{label}_{criterion}"] for criterion in RUBRIC] for label in "ABC"}}), hide_index=True, width="stretch")
    st.write(f"Ranking: {record['first_place']} > {record['second_place']} > {record['third_place']}")
    st.write(f"Constraint violation flag: {record['constraint_violation_candidate']}")
    if record["constraint_violation_explanation"]:
        st.text(record["constraint_violation_explanation"])
else:
    left, right = st.columns(2)
    left.markdown("**Baseline Response**")
    left.text(record["baseline_response"])
    right.markdown("**Improved Response**")
    right.text(record["improved_response"])
    st.write(f"Preference: {record['preferred_response']} · Protocol: {record['protocol_version']}")
st.markdown("**Rationale**")
st.text(record["rationale"])
