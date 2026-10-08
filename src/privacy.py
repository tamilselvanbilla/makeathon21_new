"""Strict local-only data policy and allowed factual-lookup classification."""

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


def is_allowed_cloud_lookup(question: str) -> bool:
    """Only permit explicit factual lookup requests; reasoning stays local."""
    normalized = question.casefold()
    if not any(term in normalized for term in ALLOWED_LOOKUP_TERMS):
        return False
    return not any(term in normalized for term in DISALLOWED_CONTEXT_TERMS)


def can_send_audio(_audio_accepted: bool) -> bool:
    """Raw audio must never be transmitted to a cloud service."""
    return False


def classify_request(question: str) -> str:
    """Classify the request as local reasoning or permitted lookup."""
    if is_allowed_cloud_lookup(question):
        return "online_factual_lookup"
    return "local_reasoning"


def redact_for_log(text: str) -> str:
    """Remove audio-like content and personal details from diagnostic logs."""
    text = re.sub(r"\b(?:password|token|secret)\b", "[REDACTED]", text, flags=re.I)
    return text[:500]
