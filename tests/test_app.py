"""Exercise real Streamlit page interactions with isolated storage and mocked generation."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.database import get_comparisons, get_evaluations, save_comparison, save_evaluation
from src.evaluation import RUBRIC, make_comparison, make_evaluation
from src.openai_client import GenerationError
from src.sample_tasks import HELD_OUT_TASKS, SAMPLE_TASKS

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {"A": "Could you please reply today?", "B": "Please reply when you can.", "C": "Please reply today."}
PAIR = {"baseline_response": "Reply when convenient.", "improved_response": "Could you please reply today?"}


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    monkeypatch.setenv("WRITING_LAB_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr("src.generation.generate_candidates", lambda **kwargs: CANDIDATES.copy())
    monkeypatch.setattr("src.experiment_ui.generate_strategy_comparison", lambda **kwargs: PAIR.copy())
    return tmp_path / "test.db"


def page(filename):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    return app.switch_page(f"pages/{filename}.py").run()


def element(elements, label):
    return next(item for item in elements if item.label == label)


def click(app, label):
    element(app.button, label).click().run()
    assert not app.exception


def sample_candidates():
    app = page("1_Generate_and_Evaluate")
    click(app, "Use this sample")
    assert app.text_area(key="rewrite_original_text").value == SAMPLE_TASKS[0]["original_text"]
    click(app, "Generate Candidates")
    return app


@pytest.mark.parametrize("filename", [
    "1_Generate_and_Evaluate", "2_Dataset", "3_Analytics", "4_Preference_Model",
    "5_Improvement_Experiment", "6_Improvement_Test", "7_Reward_Hacking", "8_Concepts",
])
def test_every_page_renders_without_a_key_or_data(filename, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = page(filename)
    assert not app.exception
    for button in app.button:
        if button.label.startswith("Generate"):
            assert button.disabled


def test_rank_save_navigation_and_dataset_filters(isolated_app):
    app = sample_candidates()
    assert len(app.slider) == 21
    click(app, "Save Evaluation")
    assert app.error
    assert get_evaluations(isolated_app).empty
    for label, choice in [("1st place", "A"), ("2nd place", "C"), ("3rd place", "B")]:
        element(app.selectbox, label).select(choice).run()
    assert element(app.selectbox, "3rd place").options == ["B"]
    evaluation_id = app.session_state["evaluation_draft"]["id"]
    app.slider(key=f"{evaluation_id}_A_meaning_preservation").set_value(5).run()
    element(app.text_area, "Why did you choose the top response?").set_value("A keeps the deadline politely.").run()
    element(app.selectbox, "Did any response violate an important constraint?").select("B").run()
    element(app.text_area, "Constraint violation explanation (optional)").set_value("Lost the deadline.").run()
    app.switch_page("pages/8_Concepts.py").run()
    app.switch_page("pages/1_Generate_and_Evaluate.py").run()
    assert app.slider(key=f"{evaluation_id}_A_meaning_preservation").value == 5
    assert element(app.selectbox, "2nd place").value == "C"
    click(app, "Save Evaluation")
    records = get_evaluations(isolated_app)
    assert len(records) == 1
    assert records.iloc[0]["first_place"] == "A"
    assert records.iloc[0]["constraint_violation_candidate"] == "B"
    assert records.iloc[0]["A_meaning_preservation"] == 5
    assert element(app.button, "Save Evaluation").disabled
    app.switch_page("pages/2_Dataset.py").run()
    assert not app.exception
    assert len(app.get("download_button")) == 1
    element(app.text_input, "Search text").set_value("keeps the deadline").run()
    assert len(app.dataframe[0].value) == 1
    element(app.selectbox, "Top candidate").select("B").run()
    assert app.dataframe[0].value.empty


def test_changing_first_rank_cannot_keep_a_duplicate_and_stale_inputs_block_save():
    app = sample_candidates()
    for label, choice in [("1st place", "A"), ("2nd place", "B"), ("3rd place", "C")]:
        element(app.selectbox, label).select(choice).run()
    element(app.selectbox, "1st place").select("B").run()
    assert element(app.selectbox, "2nd place").value is None
    selected = [x for x in app.session_state["evaluation_draft"]["ranking"] if x]
    assert len(selected) == len(set(selected))
    app.text_area(key="rewrite_original_text").set_value("A different request.").run()
    assert element(app.button, "Save Evaluation").disabled


def test_generation_failure_preserves_existing_candidate_draft(monkeypatch):
    app = sample_candidates()
    original_id = app.session_state["evaluation_draft"]["id"]

    def fail(**kwargs):
        raise GenerationError("The request failed. Please try again.")

    monkeypatch.setattr("src.generation.generate_candidates", fail)
    click(app, "Generate Candidates")
    assert app.error
    assert app.session_state["evaluation_draft"]["id"] == original_id


def test_strategy_comparison_requires_judgment_and_saves_separately(isolated_app):
    app = page("5_Improvement_Experiment")
    app.text_area(key="experiment_original_text").set_value("Reply today.")
    app.text_area(key="experiment_rewrite_instruction").set_value("Be polite.").run()
    click(app, "Generate Baseline & Improved")
    click(app, "Save Comparison")
    assert app.error
    element(app.radio, "Which response is better?").set_value("Improved").run()
    element(app.text_area, "Explain your choice").set_value("It preserves the deadline.").run()
    click(app, "Save Comparison")
    assert len(get_comparisons(isolated_app, held_out=False)) == 1
    assert get_evaluations(isolated_app).empty
    assert element(app.button, "Save Comparison").disabled


def test_duplicate_comparison_does_not_report_a_new_save(monkeypatch):
    app = page("5_Improvement_Experiment")
    app.text_area(key="experiment_original_text").set_value("Reply today.")
    app.text_area(key="experiment_rewrite_instruction").set_value("Be polite.").run()
    click(app, "Generate Baseline & Improved")
    element(app.radio, "Which response is better?").set_value("Tie").run()
    element(app.text_area, "Explain your choice").set_value("Both are equally effective.").run()
    monkeypatch.setattr("src.experiment_ui.save_comparison", lambda *args: False)
    click(app, "Save Comparison")
    assert any("existing saved vote was retained" in item.value for item in app.info)


def test_held_out_task_has_one_vote_and_uses_tie_denominator(isolated_app):
    app = page("6_Improvement_Test")
    click(app, "Generate Baseline & Improved")
    assert element(app.button, "Generate Baseline & Improved").disabled
    element(app.radio, "Which response is better?").set_value("Tie").run()
    element(app.text_area, "Explain your choice").set_value("Both meet the task equally.").run()
    click(app, "Save Comparison")
    assert len(get_comparisons(isolated_app, held_out=True)) == 1
    assert get_comparisons(isolated_app, held_out=False).empty
    # A new browser session must also see the durable vote and cannot vote again.
    app = page("6_Improvement_Test")
    assert not app.button
    assert element(app.metric, "Tie rate").value == "100%"
    assert any("already has a saved judgment" in item.value for item in app.success)


def test_development_task_is_blocked_from_held_out_generation(isolated_app):
    task = HELD_OUT_TASKS[0]
    record = make_comparison(**{field: task[field] for field in ("original_text", "rewrite_instruction", "constraints")}, **PAIR, preferred_response="Improved", rationale="Development judgment.")
    save_comparison(record, isolated_app)
    app = page("6_Improvement_Test")
    assert not app.exception
    assert not app.button
    assert any("already appears in your development data" in item.value for item in app.warning)


def test_analytics_and_model_render_with_saved_rankings(isolated_app):
    for index in range(8):
        record = make_evaluation(
            original_text=f"Please send update {index} today.", rewrite_instruction="Be polite.", constraints="Keep deadline.",
            candidates=CANDIDATES, scores={label: dict.fromkeys(RUBRIC, score) for label, score in [("A", 5), ("B", 2), ("C", 4)]},
            ranking=["A", "C", "B"], rationale="A preserves meaning and tone.",
        )
        save_evaluation(record, isolated_app)
    for filename in ("3_Analytics", "4_Preference_Model"):
        app = page(filename)
        assert not app.exception
        assert app.metric
