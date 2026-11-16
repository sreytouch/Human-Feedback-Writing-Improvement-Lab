"""An interactive illustration of how incomplete rewards miss the real goal."""

import pandas as pd
import streamlit as st

st.title("Reward hacking demo")
st.write("A score can improve while the actual task gets worse. Explore how the choice of metric changes which rewrite wins.")
st.write("**Original:** I need your answer today.")
st.write("**Goal:** Make the message more polite without changing the deadline.")
left, right = st.columns(2)
with left:
    st.subheader("Candidate A")
    st.write("Could you please send me your answer by the end of today?")
    st.caption("Polite and preserves today's deadline.")
with right:
    st.subheader("Candidate B")
    st.write("Whenever you have a chance, I would really appreciate hearing from you.")
    st.caption("Very polite, but changes the deadline into an open-ended request.")
st.write("These **hypothetical ratings** are invented for the demonstration, not generated or saved evaluations:")
ratings = pd.DataFrame({
    "Candidate": ["A", "B"], "Politeness": [4, 5], "Meaning preservation": [5, 2], "Constraint following": [5, 1]
})
st.dataframe(ratings, hide_index=True, width="stretch")

st.subheader("Scoring system 1 · Politeness only")
left, right = st.columns(2)
left.metric("Candidate A", "4.00 / 5")
right.metric("Candidate B", "5.00 / 5")
st.warning("Politeness alone rewards B, even though it removes the deadline and fails an explicit requirement.")

st.subheader("Scoring system 2 · Include the full task")
st.write("Adjust the relative weights. The app normalizes them so the final reward stays on a 1–5 scale.")
columns = st.columns(3)
politeness = columns[0].slider("Politeness weight", 0, 10, 2)
meaning = columns[1].slider("Meaning-preservation weight", 0, 10, 4)
constraints = columns[2].slider("Constraint-following weight", 0, 10, 4)
total = politeness + meaning + constraints
if total == 0:
    st.info("Set at least one weight above zero to calculate a reward.")
else:
    score_a = (politeness * 4 + meaning * 5 + constraints * 5) / total
    score_b = (politeness * 5 + meaning * 2 + constraints * 1) / total
    st.caption(f"Normalized weights: politeness {politeness / total:.0%}, meaning {meaning / total:.0%}, constraints {constraints / total:.0%}.")
    left, right = st.columns(2)
    left.metric("Candidate A reward", f"{score_a:.2f} / 5")
    right.metric("Candidate B reward", f"{score_b:.2f} / 5")
    if abs(score_a - score_b) < 1e-9:
        st.warning("This weighting ties the candidates, although B loses the required deadline.")
    elif score_a > score_b:
        st.success("This weighting prefers A, which satisfies the actual request.")
    else:
        st.warning("This weighting still overvalues politeness: B wins the score but misses the task.")
st.subheader("What this illustrates")
st.write("Optimizing an incomplete metric can improve the score while making actual task performance worse. If an optimizer learned to remove deadlines because doing so earned more politeness points, that would be reward hacking. It does not require the model to consciously cheat.")
st.write("This page demonstrates reward misalignment; it does not train an optimizer. A stronger evaluation checks meaning and every explicit constraint. Some requirements should be pass/fail gates instead of points that better style can cancel out.")
st.caption("The controls only change this illustration. They do not change saved ratings or the preference model.")
