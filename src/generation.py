"""Independent candidate generation and a fixed baseline/improved experiment."""

from src.config import Settings, get_settings
from src.openai_client import GenerationError, generate_text, validate_task_inputs

PROTOCOL_VERSION = "v1"

# Keep these strategies fixed while collecting the v1 held-out results. Changing
# a strategy requires a new protocol version and a new evaluation run.
BASELINE_INSTRUCTIONS = (
    "Rewrite the supplied original text according to the user's rewrite request. "
    "Return only the rewritten text."
)
IMPROVED_INSTRUCTIONS = (
    "Rewrite the original text according to the user's instruction. Preserve its "
    "important meaning and satisfy every explicit constraint, including deadlines, "
    "urgency, word limits, and restrictions on promises. Use the requested tone. "
    "Do not add unsupported facts, commitments, or information. Prioritize meaning, "
    "instruction following, and constraints over unnecessary stylistic changes. "
    "Before returning the rewrite, check it against the original meaning, rewrite "
    "instruction, and each constraint, and correct any violations. Return only "
    "the final rewritten text; do not include the checking process."
)
CANDIDATE_INSTRUCTIONS = (
    "Rewrite the original text according to the user's rewrite instruction and "
    "all explicit constraints. Preserve important meaning and do not add "
    "unsupported information. Return only the rewritten text. "
    "The writing approach below is secondary to the user's requirements."
)
CANDIDATE_APPROACHES = {
    "A": "Prefer a direct, natural phrasing when the requested tone allows it.",
    "B": "Consider an alternative sentence structure appropriate to the requested tone.",
    "C": "Consider a concise, carefully worded formulation appropriate to the request.",
}


def generate_candidates(
    original_text: str,
    rewrite_instruction: str,
    constraints: str = "",
    settings: Settings | None = None,
) -> dict[str, str]:
    """Generate A/B/C independently and retry each duplicate once.

    No prior candidate is included in another request. A failure or persistent
    duplicate raises GenerationError rather than returning a partial set.
    """
    original, instruction, constraint_text = validate_task_inputs(
        original_text, rewrite_instruction, constraints
    )
    settings = settings or get_settings()
    candidates: dict[str, str] = {}
    for label, approach in CANDIDATE_APPROACHES.items():
        strategy = f"{CANDIDATE_INSTRUCTIONS}\nWriting approach: {approach}"
        candidate = generate_text(original, instruction, constraint_text, strategy, settings=settings)
        if candidate in candidates.values():
            candidate = generate_text(
                original, instruction, constraint_text,
                strategy + " Use a fresh wording while still satisfying every requirement.",
                settings=settings,
            )
        if candidate in candidates.values():
            raise GenerationError(
                "The model repeated a candidate, including on one retry. Try a "
                "more open-ended rewrite task or different generation settings. "
                "No complete set of candidates was created."
            )
        candidates[label] = candidate
    return candidates


def generate_strategy_comparison(
    original_text: str,
    rewrite_instruction: str,
    constraints: str = "",
    settings: Settings | None = None,
) -> dict[str, str]:
    """Compare fixed v1 strategies using identical task and model settings.

    Only the system instructions differ. Outputs are never shared between calls;
    identical answers are allowed because a tie is a valid experimental result.
    """
    original, instruction, constraint_text = validate_task_inputs(
        original_text, rewrite_instruction, constraints
    )
    settings = settings or get_settings()
    baseline = generate_text(
        original, instruction, constraint_text, BASELINE_INSTRUCTIONS, settings=settings
    )
    improved = generate_text(
        original, instruction, constraint_text, IMPROVED_INSTRUCTIONS, settings=settings
    )
    return {"baseline_response": baseline, "improved_response": improved}
