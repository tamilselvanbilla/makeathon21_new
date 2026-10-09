"""Strict local-only data policy and graceful fallback for local answers."""

import re

WEATHER_TERMS = ("weather", "forecast", "temperature", "rain", "raining", "rainy", "umbrella")
ALLOWED_LOOKUP_TERMS = (*WEATHER_TERMS, "news", "search", "current event", "current events")
DISALLOWED_CONTEXT_TERMS = (
    "financial",
    "medical",
    "document",
    "recording",
    "personal data",
    "private",
)

UNCERTAINTY_MARKERS = (
    "i cannot answer",
    "i do not know",
    "not enough information",
    "unavailable",
    "unable to determine",
)

LOCAL_FALLBACK_ANSWER = (
    "I cannot answer this accurately with the local model. I can offer a "
    "general explanation, or you can consult a qualified professional."
)


def is_allowed_cloud_lookup(question: str) -> bool:
    """Only permit explicit factual lookup requests; reasoning stays local."""
    normalized = question.casefold()
    # Whole words only, so "train" or "brain" never count as "rain".
    if not any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in ALLOWED_LOOKUP_TERMS):
        return False
    return not any(term in normalized for term in DISALLOWED_CONTEXT_TERMS)


def is_uncertain(response: str) -> bool:
    """Empty, uncertain, or non-answer responses. They are spoken as they are, but not
    remembered as facts, and uncertain weather advice is dropped."""
    normalized = response.casefold()
    if not normalized.strip():
        return True
    return any(marker in normalized for marker in UNCERTAINTY_MARKERS) or bool(
        re.search(r"\b(?:cannot|can't|unable)\b", normalized)
    )


def require_local_answer(response: str) -> str:
    """Replace an empty reply with the fallback. A refusal in the model's own words
    ("...not recorded in the provided information") is kept: it says more than the
    generic fallback, which used to overwrite any reply containing "cannot"."""
    return response if response.strip() else LOCAL_FALLBACK_ANSWER


def redact_for_log(text: str) -> str:
    """Remove credentials from diagnostic logs."""
    text = re.sub(r"\b(?:password|token|secret)\b", "[REDACTED]", text, flags=re.I)
    return text[:500]
