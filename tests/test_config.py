"""Configuration loading is isolated from real environment secrets."""

import pytest

from src import config
from src.config import ConfigurationError, get_settings

ENV_NAMES = (
    "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_USE_TEMPERATURE", "OPENAI_TEMPERATURE",
    "OPENAI_MAX_OUTPUT_TOKENS", "WRITING_LAB_DB_PATH",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_do_not_require_api_key(tmp_path):
    settings = get_settings()
    assert settings.api_key is None
    assert settings.model == "gpt-4.1-mini"
    assert settings.temperature == 0.7
    assert settings.use_temperature is True
    assert settings.max_output_tokens == 1024
    assert settings.db_path == tmp_path / "data" / "writing_lab.db"


def test_project_relative_env_and_process_priority(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-only-key\nOPENAI_MODEL=file-model\n"
        "WRITING_LAB_DB_PATH=custom/data.db\nOPENAI_USE_TEMPERATURE=false\n"
    )
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    settings = get_settings()
    assert settings.api_key == "test-only-key"
    assert "test-only-key" not in repr(settings)
    assert settings.model == "process-model"
    assert settings.use_temperature is False
    assert settings.db_path == tmp_path / "custom" / "data.db"


def test_settings_are_read_fresh_without_mutating_environment(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_MODEL=first-model\n")
    assert get_settings().model == "first-model"
    env.write_text("OPENAI_MODEL=second-model\n")
    assert get_settings().model == "second-model"


def test_placeholder_is_not_a_usable_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")
    assert get_settings().api_key is None


def test_absolute_db_path(tmp_path, monkeypatch):
    path = tmp_path / "other" / "lab.db"
    monkeypatch.setenv("WRITING_LAB_DB_PATH", str(path))
    assert get_settings().db_path == path


@pytest.mark.parametrize("name,value", [
    ("OPENAI_MODEL", " "), ("OPENAI_USE_TEMPERATURE", "sometimes"),
    ("OPENAI_TEMPERATURE", "hot"), ("OPENAI_TEMPERATURE", "nan"),
    ("OPENAI_TEMPERATURE", "inf"), ("OPENAI_TEMPERATURE", "2.1"),
    ("OPENAI_MAX_OUTPUT_TOKENS", "1.5"), ("OPENAI_MAX_OUTPUT_TOKENS", "0"),
    ("WRITING_LAB_DB_PATH", ""),
])
def test_invalid_settings_have_actionable_errors(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        get_settings()
