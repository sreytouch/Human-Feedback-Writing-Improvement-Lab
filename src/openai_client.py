"""One independent rewrite request through the official OpenAI Python SDK."""

import math
from typing import Any

from openai import (
    APIConnectionError, APIError, APITimeoutError, AuthenticationError,
    BadRequestError, NotFoundError, OpenAI, PermissionDeniedError, RateLimitError,
)

from src.config import Settings, get_settings


class GenerationError(RuntimeError):
    """A generation failure with a message safe to display in the interface."""


def validate_task_inputs(
    original_text: str, rewrite_instruction: str, constraints: str
) -> tuple[str, str, str]:
    """Require an original and instruction, while allowing blank constraints."""
    if not isinstance(original_text, str) or not original_text.strip():
        raise GenerationError("Enter the original text before generating rewrites.")
    if not isinstance(rewrite_instruction, str) or not rewrite_instruction.strip():
        raise GenerationError("Enter a rewrite instruction before generating rewrites.")
    if not isinstance(constraints, str):
        raise GenerationError("Constraints must be text.")
    return original_text.strip(), rewrite_instruction.strip(), constraints.strip()


def _answer_text(response: Any) -> str:
    """Accept only complete, nonempty answers, never a partial response or refusal."""
    if response.status != "completed":
        raise GenerationError(
            "The model did not finish its rewrite. Try shorter input or increase "
            "OPENAI_MAX_OUTPUT_TOKENS in .env. No complete result was created."
        )
    for item in response.output:
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "refusal":
                raise GenerationError(
                    "The model declined the request. Rephrase the original text or "
                    "instruction and try again. No complete result was created."
                )
    answer = response.output_text.strip()
    if not answer:
        raise GenerationError("The model returned no rewrite text. Try another request or model.")
    return answer


def generate_text(
    original_text: str,
    rewrite_instruction: str,
    constraints: str,
    instructions: str,
    settings: Settings | None = None,
) -> str:
    """Request one rewrite with no conversation history and no other outputs.

    Model, output budget, and sampling settings are shared across experimental
    strategies. The SDK may retry a transport/API failure once; this is separate
    from the candidate-level duplicate retry in generation.py.
    """
    original, instruction, constraint_text = validate_task_inputs(
        original_text, rewrite_instruction, constraints
    )
    settings = settings or get_settings()
    if not settings.api_key or settings.api_key == "your_api_key_here":
        raise GenerationError(
            "Add OPENAI_API_KEY to your project's .env file. Copy .env.example "
            "to .env, then replace your_api_key_here with your own API key."
        )
    if not isinstance(instructions, str) or not instructions.strip():
        raise GenerationError("The generation strategy must contain instructions.")
    if not isinstance(settings.use_temperature, bool):
        raise GenerationError("OPENAI_USE_TEMPERATURE must be true or false.")
    if settings.use_temperature and (
        isinstance(settings.temperature, bool)
        or not isinstance(settings.temperature, (int, float))
        or not math.isfinite(settings.temperature)
        or not 0 <= settings.temperature <= 2
    ):
        raise GenerationError("OPENAI_TEMPERATURE must be a number from 0 to 2.")
    user_input = (
        f"Original text:\n{original}\n\n"
        f"Rewrite instruction:\n{instruction}\n\n"
        f"Constraints:\n{constraint_text or 'No additional constraints.'}"
    )
    request: dict[str, Any] = {
        "model": settings.model,
        "instructions": instructions,
        "input": [{"role": "user", "content": user_input}],
        "max_output_tokens": settings.max_output_tokens,
        "store": False,
    }
    if settings.use_temperature:
        request["temperature"] = float(settings.temperature)
    try:
        with OpenAI(api_key=settings.api_key, timeout=30.0, max_retries=1) as client:
            return _answer_text(client.responses.create(**request))
    except AuthenticationError as exc:
        raise GenerationError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.") from exc
    except RateLimitError as exc:
        raise GenerationError(
            "OpenAI's rate or usage limit was reached. Check API billing and limits, "
            "then try again later."
        ) from exc
    except APITimeoutError as exc:
        raise GenerationError("OpenAI took too long to respond. Please try again.") from exc
    except APIConnectionError as exc:
        raise GenerationError("Could not connect to OpenAI. Check your connection and try again.") from exc
    except (PermissionDeniedError, NotFoundError) as exc:
        raise GenerationError(
            "The configured model is unavailable to this API key. Check OPENAI_MODEL and model access."
        ) from exc
    except BadRequestError as exc:
        raise GenerationError(
            "OpenAI rejected the generation settings. Check OPENAI_MODEL and the token limit. "
            "For a model that does not support temperature, set OPENAI_USE_TEMPERATURE=false in .env."
        ) from exc
    except APIError as exc:
        raise GenerationError("OpenAI could not complete the request. Try again later.") from exc
