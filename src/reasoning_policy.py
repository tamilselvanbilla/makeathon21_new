"""Local reasoning capability and graceful fallback policy."""

import re

UNCERTAINTY_MARKERS = (
    "i cannot answer",
    "i do not know",
    "not enough information",
    "unavailable",
    "unable to determine",
)


def should_fallback(response: str) -> bool:
    """Detect honest uncertainty or non-answer responses."""
    normalized = response.casefold()
    return any(marker in normalized for marker in UNCERTAINTY_MARKERS) or bool(
        re.search(r"\b(?:cannot|can't|unable)\b", normalized)
    )


def require_local_answer(response: str) -> str:
    """Return a safe fallback when the local model declines or is uncertain."""
    if should_fallback(response):
        return "I cannot answer this accurately with the local model. I can offer a general explanation, or you can consult a qualified professional."
    return response
