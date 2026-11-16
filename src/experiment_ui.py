"""Shared baseline/improved interaction for development and held-out pages."""

import sqlite3
from uuid import uuid4

import streamlit as st

from src.config import Settings
from src.database import save_comparison
from src.evaluation import make_comparison
from src.generation import PROTOCOL_VERSION, generate_strategy_comparison
from src.openai_client import GenerationError
from src.ui import generation_ready, remember_feedback, show_task


def render_comparison(task: dict, settings: Settings, state_key: str, held_out_task_id: str | None = None) -> None:
    """Keep each judgment attached to the exact task and response pair."""
    fields = ("original_text", "rewrite_instruction", "constraints")
    current = st.session_state.get(state_key)
    ready = generation_ready(settings)
    fixed_pair_exists = held_out_task_id is not None and current is not None
    st.caption("Both strategies receive the same task, model, sampling settings, and token budget. Only the generation instructions differ. Two independent API calls are made; equal responses are a valid result.")
    if st.button("Generate Baseline & Improved", key=f"{state_key}_generate", type="primary", disabled=not ready or fixed_pair_exists):
        try:
            with st.spinner("Generating baseline and improved rewrites…"):
                outputs = generate_strategy_comparison(**{field: task[field] for field in fields}, settings=settings)
            current = {
                "id": str(uuid4()), **{field: task[field] for field in fields}, **outputs,
                "preferred_response": None, "rationale": "", "saved": False,
                "held_out_task_id": held_out_task_id, "protocol_version": PROTOCOL_VERSION,
            }
            st.session_state[state_key] = current
            st.rerun()
        except GenerationError as error:
            st.error(str(error))
    if not current:
        return
    stale = any(task[field].strip() != current[field].strip() for field in fields)
    if stale:
        st.warning("The task has changed. Generate a fresh pair, or restore the original task, before saving.")
    with st.expander("Task used for this pair"):
        show_task(current)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Baseline Response")
        with st.container(border=True):
            st.text(current["baseline_response"])
    with right:
        st.subheader("Improved Response")
        with st.container(border=True):
            st.text(current["improved_response"])
    if current["baseline_response"] == current["improved_response"]:
        st.info("Both strategies produced the same text. A tie is a valid judgment.")
    preference_key = f"{current['id']}_preference"
    choices = ["Baseline", "Improved", "Tie"]
    preference = current["preferred_response"]
    st.radio("Which response is better?", choices, index=choices.index(preference) if preference else None, horizontal=True, key=preference_key, disabled=current["saved"], on_change=remember_feedback, args=(state_key, "preferred_response", preference_key))
    rationale_key = f"{current['id']}_rationale"
    st.text_area("Explain your choice", value=current["rationale"], key=rationale_key, max_chars=5000, disabled=current["saved"], on_change=remember_feedback, args=(state_key, "rationale", rationale_key))
    if st.button("Save Comparison", key=f"{state_key}_save", type="primary", disabled=current["saved"] or stale):
        try:
            record = make_comparison(
                **{field: current[field] for field in fields},
                baseline_response=current["baseline_response"], improved_response=current["improved_response"],
                preferred_response=current["preferred_response"], rationale=current["rationale"],
                comparison_id=current["id"], held_out_task_id=held_out_task_id,
            )
            inserted = save_comparison(record, settings.db_path)
            current["saved"] = True
            current["already_saved"] = not inserted
            st.rerun()
        except ValueError as error:
            st.error(str(error))
        except (sqlite3.Error, OSError):
            st.error("Could not save this comparison. Check database permissions and try again. Your draft is retained.")
    if current["saved"]:
        if current.get("already_saved"):
            st.info("A judgment was already saved for this comparison or test task. The existing saved vote was retained.")
        else:
            st.success("Comparison saved in the separate strategy-comparison dataset.")
