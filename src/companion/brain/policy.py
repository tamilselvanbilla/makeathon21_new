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


# The small model sometimes answers as the user ("I am 175 cm tall", "I am taking ...").
# Telling it not to in the prompt made its other answers worse, so the opening is fixed
# here, except where the assistant speaks of itself ("I am not sure", "I am sorry").
SPOKEN_AS_USER = re.compile(
    r"^I(?: am|'m)\b(?!\s+(?:not|unable|sorry|afraid|an?|your|here|happy|glad|just|only|the)\b)|"
    r"^I (?=(?:take|invested|bought|purchased|joined|worked|studied|took|started)\b)"
)


def as_second_person(answer: str) -> str:
    """ "I am 175 cm tall" -> "You are 175 cm tall", "I invested ..." -> "You invested ..."."""
    return SPOKEN_AS_USER.sub(lambda m: "You " if m.group() == "I " else "You are", answer)


# A sentence end is punctuation, maybe closing quotes or brackets, then a space or the
# end, so the "." in "25.5" is not one.
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def full_sentences(text: str) -> str:
    """Drop an unfinished last sentence (a model reply cut off by its token limit); a
    reply with no full sentence is ended with "." instead."""
    text = text.strip()
    ends = [match.end() for match in SENTENCE_END.finditer(text)]
    if ends and ends[-1] == len(text) or not text:
        return text
    return text[: ends[-1]] if ends else text.rstrip(",;:- ") + "."


def drop_repeated(answer: str, previous_answer: str) -> str:
    """Remove sentences copied from the previous answer: shown the last exchange for
    context, the small model often repeats it before answering the follow-up."""
    seen = {s.strip().casefold() for s in SENTENCE_SPLIT.split(previous_answer) if s.strip()}
    kept = [s for s in SENTENCE_SPLIT.split(answer) if s.strip() and s.strip().casefold() not in seen]
    return " ".join(kept) if kept else answer


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def limit_words(text: str, max_words: int) -> str:
    """At most `max_words` words, ending at the last full sentence that fits."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return full_sentences(" ".join(words[:max_words]))


def redact_for_log(text: str) -> str:
    """Remove credentials from diagnostic logs."""
    text = re.sub(r"\b(?:password|token|secret)\b", "[REDACTED]", text, flags=re.I)
    return text[:500]
