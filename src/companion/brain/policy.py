"""Strict local-only data policy and graceful fallback for local answers."""

import re

ALLOWED_LOOKUP_TERMS = (
    "weather",
    "forecast",
    "news",
    "search",
    "current event",
)
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
    if not any(term in normalized for term in ALLOWED_LOOKUP_TERMS):
        return False
    return not any(term in normalized for term in DISALLOWED_CONTEXT_TERMS)


def should_fallback(response: str) -> bool:
    """Detect empty, uncertain, or non-answer responses."""
    normalized = response.casefold()
    if not normalized.strip():
        return True
    return any(marker in normalized for marker in UNCERTAINTY_MARKERS) or bool(
        re.search(r"\b(?:cannot|can't|unable)\b", normalized)
    )


def require_local_answer(response: str) -> str:
    """Return a safe fallback when the local model declines or is uncertain."""
    if should_fallback(response):
        return LOCAL_FALLBACK_ANSWER
    return response


def redact_for_log(text: str) -> str:
    """Remove credentials from diagnostic logs."""
    text = re.sub(r"\b(?:password|token|secret)\b", "[REDACTED]", text, flags=re.I)
    return text[:500]
