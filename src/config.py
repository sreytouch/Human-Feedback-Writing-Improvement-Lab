"""Read project configuration without exposing keys or mutating the environment."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """A configuration setting needs correction before it can be used."""


@dataclass(frozen=True)
class Settings:
    """Settings shared by all generation strategies; keys stay out of repr."""

    api_key: str | None = field(default=None, repr=False)
    model: str = "gpt-4.1-mini"
    use_temperature: bool = True
    temperature: float = 0.7
    max_output_tokens: int = 1024
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "writing_lab.db")


def get_settings() -> Settings:
    """Read .env relative to this project; existing environment values win."""
    values = {**dotenv_values(PROJECT_ROOT / ".env"), **os.environ}

    def read(name: str, default: str) -> str:
        value = values.get(name, default)
        return str(value).strip() if value is not None else ""

    key = read("OPENAI_API_KEY", "")
    if key in {"", "your_api_key_here"}:
        key = None
    model = read("OPENAI_MODEL", "gpt-4.1-mini")
    if not model:
        raise ConfigurationError("OPENAI_MODEL must contain a model name.")
    temperature_enabled = read("OPENAI_USE_TEMPERATURE", "true").lower()
    if temperature_enabled not in {"true", "false", "yes", "no", "1", "0"}:
        raise ConfigurationError("OPENAI_USE_TEMPERATURE must be true or false.")
    try:
        temperature = float(read("OPENAI_TEMPERATURE", "0.7"))
    except ValueError as exc:
        raise ConfigurationError("OPENAI_TEMPERATURE must be a number from 0 to 2.") from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ConfigurationError("OPENAI_TEMPERATURE must be a number from 0 to 2.")
    try:
        max_tokens = int(read("OPENAI_MAX_OUTPUT_TOKENS", "1024"))
    except ValueError as exc:
        raise ConfigurationError("OPENAI_MAX_OUTPUT_TOKENS must be an integer from 64 to 16384.") from exc
    if not 64 <= max_tokens <= 16384:
        raise ConfigurationError("OPENAI_MAX_OUTPUT_TOKENS must be an integer from 64 to 16384.")
    db_value = read("WRITING_LAB_DB_PATH", "data/writing_lab.db")
    if not db_value:
        raise ConfigurationError("WRITING_LAB_DB_PATH must contain a database file path.")
    db_path = Path(db_value).expanduser()
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Settings(
        api_key=key, model=model,
        use_temperature=temperature_enabled in {"true", "yes", "1"},
        temperature=temperature, max_output_tokens=max_tokens,
        db_path=db_path.resolve(),
    )
