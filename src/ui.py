"""Shared, small Streamlit helpers for storage, inputs, and session drafts."""

import sqlite3

import pandas as pd
import streamlit as st

from src.config import ConfigurationError, Settings, get_settings
from src.database import get_comparisons, get_evaluations


def settings_or_stop() -> Settings:
    try:
        return get_settings()
    except ConfigurationError as error:
        st.error(str(error))
        st.stop()


def load_evaluations() -> pd.DataFrame:
    settings = settings_or_stop()
    try:
        return get_evaluations(settings.db_path)
    except (sqlite3.Error, OSError, pd.errors.DatabaseError):
        st.error("Could not open the evaluation database. Check WRITING_LAB_DB_PATH and folder permissions.")
        st.stop()


def load_comparisons(held_out: bool | None = None) -> pd.DataFrame:
    settings = settings_or_stop()
    try:
        return get_comparisons(settings.db_path, held_out=held_out)
    except (sqlite3.Error, OSError, pd.errors.DatabaseError):
        st.error("Could not open strategy comparisons. Check the database file and folder permissions.")
        st.stop()


def generation_ready(settings: Settings) -> bool:
    ready = bool(settings.api_key)
    if not ready:
        st.info("To generate text, copy .env.example to .env and set OPENAI_API_KEY to your own key. The learning pages and saved datasets work without a key.")
    st.caption(f"Model: {settings.model} · Generation sends the task to OpenAI and may incur API charges. Configure model and sampling in .env.")
    return ready


def remember_draft(key: str, state_key: str, field: str) -> None:
    st.session_state.setdefault(state_key, {})[field] = st.session_state[key]


def task_inputs(prefix: str) -> dict[str, str]:
    """Keep user inputs when navigating to other pages in the same session."""
    state_key = f"{prefix}_draft"
    draft = st.session_state.setdefault(state_key, {})
    task = {}
    for field, label, height in (
        ("original_text", "Original text", 110),
        ("rewrite_instruction", "Rewrite instruction", 80),
        ("constraints", "Optional constraints", 80),
    ):
        key = f"{prefix}_{field}"
        st.session_state.setdefault(key, draft.get(field, ""))
        task[field] = st.text_area(label, key=key, height=height, max_chars=20000 if field == "original_text" else 5000, on_change=remember_draft, args=(key, state_key, field)).strip()
    return task


def show_task(task: dict) -> None:
    for field, label in (("original_text", "Original text"), ("rewrite_instruction", "Rewrite instruction"), ("constraints", "Constraints")):
        st.markdown(f"**{label}**")
        st.text(task[field] or "None")


def remember_feedback(state_key: str, field: str, widget_key: str) -> None:
    st.session_state[state_key][field] = st.session_state[widget_key]
