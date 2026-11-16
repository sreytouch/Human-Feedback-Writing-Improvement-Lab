"""A grouped-holdout preference-model demonstration using rubric differences."""

import streamlit as st

from src.preference_model import predict_comparison, train_preference_model, training_requirement
from src.ui import load_evaluations

st.title("Preference model")
st.write("Can a small model learn which rating differences tend to accompany human preferences?")
st.info("This is a simplified educational preference model, not a production RLHF reward model.")
st.write("The classifier sees seven human rubric-score differences. It does not read candidate text, change the OpenAI model, or run a reinforcement-learning algorithm.")
frame = load_evaluations()
with st.expander("From rankings to training pairs", expanded=True):
    st.markdown(
        "For a ranking **A > C > B**, we create **A over C**, **A over B**, and **C over B**. "
        "For each pair, the features are the left candidate's ratings minus the right candidate's ratings.\n\n"
        "We also reverse each pair and label it 0 instead of 1. This supplies both outcomes without treating "
        "the left-hand position as always preferred. These mirrors are **not new human judgments**.\n\n"
        "We split **whole evaluation IDs** into 75% training and 25% held out **before** making pairs. "
        "All pairs and their mirrors from one evaluation stay together."
    )
reason = training_requirement(frame)
if reason:
    st.warning(reason)
    st.stop()
try:
    result = train_preference_model(frame)
except ValueError as error:
    st.error(str(error))
    st.stop()

columns = st.columns(4)
columns[0].metric("Training evaluations", len(result.training_ids))
columns[1].metric("Held-out evaluations", len(result.held_out_ids))
columns[2].metric("Training ranking pairs", result.training_pair_count)
columns[3].metric("Held-out pair accuracy", f"{result.accuracy:.1%}")
st.caption(
    f"Training: {result.training_pair_count} ranking pairs → {result.augmented_training_pair_count} rows after mirroring. "
    f"Held out: {result.held_out_pair_count} ranking pairs → {result.augmented_held_out_pair_count} rows after mirroring. "
    "Accuracy uses the mirrored held-out rows; an always-left or always-right prediction scores 50%."
)
st.caption("The split is reproducible. This model stays fitted only on its training groups, so the reported accuracy and example predictions refer to the same model. Small or repetitive datasets can make this estimate unreliable; similar tasks in both sets can still make the score optimistic.")

st.subheader("What the classifier learned")
st.dataframe(result.coefficients, hide_index=True, width="stretch")
st.write("A positive coefficient means a higher rating for that criterion tends to favor the candidate. A negative coefficient goes the other way. All differences use the same 1–5 rating scale. The model has no intercept, so reversing a pair reverses its estimated preference.")
st.caption("Coefficients describe associations in your collected ratings, not objective importance or causation. Ratings are already human judgments, so this demo cannot evaluate a new rewrite without someone first scoring it.")

st.subheader("Inspect a prediction")
records = {str(row["evaluation_id"]): row.to_dict() for _, row in frame.iterrows()}
selected = st.selectbox(
    "Saved evaluation",
    options=list(records),
    format_func=lambda key: f"{str(records[key]['timestamp'])[:16]} · {records[key]['original_text'][:65]} · {key[:8]}",
)
record = records[selected]
left_column, right_column = st.columns(2)
left = left_column.selectbox("Left candidate", ["A", "B", "C"])
right = right_column.selectbox("Right candidate", [candidate for candidate in ("A", "B", "C") if candidate != left])
prediction = predict_comparison(result, record, left, right)
st.write(f"**{prediction['membership']}** · Human preferred **{prediction['human_preferred_candidate']}** in this pair.")
if abs(prediction["probability_left"] - 0.5) < 1e-9:
    st.write("The model assigns **50% to each candidate**; this pair has no estimated advantage.")
else:
    st.write(f"Model prefers **Candidate {prediction['predicted_candidate']}**.")
columns = st.columns(2)
columns[0].metric(f"Estimated P({left} preferred)", f"{prediction['probability_left']:.1%}")
columns[1].metric(f"Estimated P({right} preferred)", f"{prediction['probability_right']:.1%}")
st.caption("These are uncalibrated estimates of preference, not probabilities that the rewrite is correct.")
if prediction["membership"] == "Training evaluation":
    st.caption("This evaluation helped fit the classifier. Its prediction is not an independent test.")
with st.expander("Read the evaluation"):
    st.write("**Original text**")
    st.text(record["original_text"])
    st.write("**Instruction**")
    st.text(record["rewrite_instruction"])
    if record["constraints"]:
        st.write("**Constraints**")
        st.text(record["constraints"])
    columns = st.columns(2)
    for column, label in zip(columns, (left, right)):
        column.write(f"**Candidate {label}**")
        column.text(record[f"candidate_{label.lower()}"])
    st.write("**Human rationale**")
    st.text(record["rationale"])
