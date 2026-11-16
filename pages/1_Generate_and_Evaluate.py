"""Generate three independent rewrites and collect one validated ranking."""

import sqlite3
from uuid import uuid4

import streamlit as st

from src.database import save_evaluation
from src.evaluation import RUBRIC, calculate_average, make_evaluation
from src.generation import generate_candidates
from src.openai_client import GenerationError
from src.sample_tasks import SAMPLE_TASKS
from src.ui import generation_ready, remember_feedback, settings_or_stop, show_task, task_inputs


def load_sample(sample: dict) -> None:
    task = {key: sample[key] for key in ("original_text", "rewrite_instruction", "constraints")}
    st.session_state["rewrite_draft"] = task
    for key, value in task.items():
        st.session_state[f"rewrite_{key}"] = value


def remember_score(label: str, criterion: str, key: str) -> None:
    st.session_state["evaluation_draft"]["scores"][label][criterion] = st.session_state[key]


def remember_rank(position: int, key: str) -> None:
    st.session_state["evaluation_draft"]["ranking"][position] = st.session_state[key]


st.title("Generate & Evaluate")
st.write("Compare three rewrites of the same task. Judge meaning and requirements as carefully as tone.")
settings = settings_or_stop()
with st.expander("Sample-task library"):
    selected = st.selectbox("Sample task", range(len(SAMPLE_TASKS)), format_func=lambda index: SAMPLE_TASKS[index]["title"])
    sample = SAMPLE_TASKS[selected]
    show_task(sample)
    st.button("Use this sample", on_click=load_sample, args=(sample,))
task = task_inputs("rewrite")
ready = generation_ready(settings)
st.caption("Three separate requests generate A, B, and C. Duplicate candidates may be retried once; failed requests may also be retried by the SDK.")
if st.button("Generate Candidates", type="primary", disabled=not ready):
    try:
        with st.spinner("Generating three independent rewrites…"):
            candidates = generate_candidates(**task, settings=settings)
        st.session_state["evaluation_draft"] = {
            "id": str(uuid4()), **task, "candidates": candidates,
            "scores": {label: dict.fromkeys(RUBRIC, 3) for label in "ABC"},
            "ranking": [None, None, None], "rationale": "",
            "constraint_violation_candidate": "None", "constraint_violation_explanation": "", "saved": False,
        }
    except GenerationError as error:
        st.error(str(error))

current = st.session_state.get("evaluation_draft")
if not current:
    st.info("Your three candidates will appear here. Nothing is saved until you submit an evaluation.")
    st.stop()
st.divider()
with st.expander("Task used for these candidates"):
    show_task(current)
stale = any(task[field] != current[field] for field in task)
if stale:
    st.warning("The input differs from the task used for these rewrites. Restore it or generate new candidates before saving.")
st.caption("Scores start at 3. Review every criterion: 1 = poor, 3 = adequate, 5 = excellent. When no explicit constraint applies, judge compliance with the task's stated requirements.")
for label, column in zip("ABC", st.columns(3, gap="medium")):
    with column:
        st.subheader(f"Candidate {label}")
        with st.container(border=True):
            st.text(current["candidates"][label])
        for criterion, details in RUBRIC.items():
            key = f"{current['id']}_{label}_{criterion}"
            st.slider(details["label"], 1, 5, current["scores"][label][criterion], key=key, help=details["description"], disabled=current["saved"], on_change=remember_score, args=(label, criterion, key))
        st.metric("Mean rubric score", f"{calculate_average(current['scores'][label]):.2f} / 5")
st.subheader("Rank the candidates")
st.caption("Rank all three once. The ranking can differ from the score averages; explain your reasoning below.")
chosen = []
for position, (label, column) in enumerate(zip(["1st place", "2nd place", "3rd place"], st.columns(3))):
    available = [candidate for candidate in "ABC" if candidate not in chosen]
    value = current["ranking"][position]
    if value not in available:
        current["ranking"][position] = value = None
    key = f"{current['id']}_rank_{position}_{''.join(chosen)}"
    with column:
        rank = st.selectbox(label, available, index=available.index(value) if value else None, key=key, disabled=current["saved"], placeholder="Choose candidate", on_change=remember_rank, args=(position, key))
    if rank:
        chosen.append(rank)
rationale_key = f"{current['id']}_rationale"
st.text_area("Why did you choose the top response?", value=current["rationale"], key=rationale_key, max_chars=5000, disabled=current["saved"], on_change=remember_feedback, args=("evaluation_draft", "rationale", rationale_key))
violation_key = f"{current['id']}_violation"
choices = ["None", "A", "B", "C"]
st.selectbox("Did any response violate an important constraint?", choices, index=choices.index(current["constraint_violation_candidate"]), key=violation_key, format_func=lambda item: item if item == "None" else f"Candidate {item}", disabled=current["saved"], on_change=remember_feedback, args=("evaluation_draft", "constraint_violation_candidate", violation_key))
st.caption("Record one candidate with a key violation. If several violate a requirement, mention the others in the explanation.")
if current["constraint_violation_candidate"] != "None":
    explanation_key = f"{current['id']}_violation_explanation"
    st.text_area("Constraint violation explanation (optional)", value=current["constraint_violation_explanation"], key=explanation_key, max_chars=5000, disabled=current["saved"], on_change=remember_feedback, args=("evaluation_draft", "constraint_violation_explanation", explanation_key))
if st.button("Save Evaluation", type="primary", disabled=current["saved"] or stale):
    try:
        record = make_evaluation(
            **{field: current[field] for field in task}, candidates=current["candidates"], scores=current["scores"],
            ranking=current["ranking"], rationale=current["rationale"],
            constraint_violation_candidate=current["constraint_violation_candidate"],
            constraint_violation_explanation=current["constraint_violation_explanation"] if current["constraint_violation_candidate"] != "None" else "",
            evaluation_id=current["id"],
        )
        save_evaluation(record, settings.db_path)
        current["saved"] = True
        st.rerun()
    except ValueError as error:
        st.error(str(error))
    except (sqlite3.Error, OSError):
        st.error("Could not save the evaluation. Check the database folder permissions and try again. Your draft is still here.")
if current["saved"]:
    st.success("Evaluation saved. Generate another set to collect more feedback.")
    st.page_link("pages/2_Dataset.py", label="Inspect the dataset", icon="🗂️")
