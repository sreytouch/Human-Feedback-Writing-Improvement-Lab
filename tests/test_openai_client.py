"""Mocked SDK tests never issue a live API request."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
    InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError,
)

from src import openai_client
from src.config import Settings
from src.openai_client import GenerationError, generate_text


def response(text="A rewrite", status="completed", refusal=False):
    return SimpleNamespace(
        output_text=text, status=status,
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")] if refusal else [])],
    )


@pytest.fixture
def fake_sdk(monkeypatch):
    constructor = MagicMock()
    client = constructor.return_value.__enter__.return_value
    client.responses.create.return_value = response()
    monkeypatch.setattr(openai_client, "OpenAI", constructor)
    return constructor, client


@pytest.fixture
def settings():
    return Settings(api_key="test-only-key")


def test_request_includes_all_inputs_and_settings(fake_sdk, settings):
    constructor, client = fake_sdk
    result = generate_text("I need this today.", "Make this polite.", "Keep today.", "Return a rewrite.", settings)
    assert result == "A rewrite"
    constructor.assert_called_once_with(api_key=settings.api_key, timeout=30.0, max_retries=1)
    request = client.responses.create.call_args.kwargs
    assert request["input"] == [{"role": "user", "content": "Original text:\nI need this today.\n\nRewrite instruction:\nMake this polite.\n\nConstraints:\nKeep today."}]
    assert request["instructions"] == "Return a rewrite."
    assert request["model"] == settings.model
    assert request["temperature"] == 0.7
    assert request["max_output_tokens"] == 1024
    assert request["store"] is False
    assert "previous_response_id" not in request
    assert "conversation" not in request


def test_unsupported_temperature_can_be_omitted(fake_sdk):
    _, client = fake_sdk
    generate_text("Original", "Rewrite", "", "Strategy", Settings(api_key="test", use_temperature=False))
    assert "temperature" not in client.responses.create.call_args.kwargs


@pytest.mark.parametrize("key", [None, "", "your_api_key_here"])
def test_missing_key_has_setup_help_without_request(fake_sdk, key):
    constructor, _ = fake_sdk
    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        generate_text("Original", "Rewrite", "", "Strategy", Settings(api_key=key))
    constructor.assert_not_called()


@pytest.mark.parametrize("original,instruction,constraints,match", [
    (" ", "Rewrite", "", "original text"),
    ("Original", " ", "", "rewrite instruction"),
    ("Original", "Rewrite", None, "Constraints"),
])
def test_invalid_inputs_do_not_call_api(fake_sdk, settings, original, instruction, constraints, match):
    constructor, _ = fake_sdk
    with pytest.raises(GenerationError, match=match):
        generate_text(original, instruction, constraints, "Strategy", settings)
    constructor.assert_not_called()


@pytest.mark.parametrize("answer,match", [
    (response(""), "no rewrite text"),
    (response("Partial", "incomplete"), "did not finish"),
    (response("Refusal", refusal=True), "declined"),
])
def test_unusable_api_output_is_rejected(fake_sdk, settings, answer, match):
    _, client = fake_sdk
    client.responses.create.return_value = answer
    with pytest.raises(GenerationError, match=match):
        generate_text("Original", "Rewrite", "", "Strategy", settings)


@pytest.mark.parametrize("error_class,status,match", [
    (AuthenticationError, 401, "API key"), (RateLimitError, 429, "rate or usage"),
    (BadRequestError, 400, "temperature"), (NotFoundError, 404, "unavailable"),
    (PermissionDeniedError, 403, "unavailable"), (InternalServerError, 500, "Try again later"),
])
def test_safe_api_error_messages(fake_sdk, settings, error_class, status, match):
    _, client = fake_sdk
    http_response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    client.responses.create.side_effect = error_class("unsafe test-only-key detail", response=http_response, body=None)
    with pytest.raises(GenerationError, match=match) as error:
        generate_text("Original", "Rewrite", "", "Strategy", settings)
    assert settings.api_key not in str(error.value)
    assert "unsafe" not in str(error.value)


@pytest.mark.parametrize("kind,match", [(APIConnectionError, "connection"), (APITimeoutError, "too long")])
def test_connection_errors_are_readable(fake_sdk, settings, kind, match):
    _, client = fake_sdk
    client.responses.create.side_effect = kind(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    with pytest.raises(GenerationError, match=match):
        generate_text("Original", "Rewrite", "", "Strategy", settings)
