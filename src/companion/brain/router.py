"""Decide how a transcribed request is handled.

Rule-based for now; a grammar-constrained LLM router can replace `route()`
without changing callers.
"""

import re
from enum import Enum

from ..online_gateway import LookupRequest
from .policy import WEATHER_TERMS, is_allowed_cloud_lookup

EXIT_PHRASES = frozenset({"quit", "exit", "stop", "goodbye"})

# "in Bengaluru", "for New York tomorrow", "at Chennai?" -> the place name.
PLACE_AFTER_PREPOSITION = re.compile(
    r"\b(?:in|at|for|of)\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3}?)"
    r"(?=\s+(?:today|tomorrow|tonight|now|right|this|and|or|but|so|should|will|is|to|with|like)\b|\s*[?.!,]|\s*$)",
    re.IGNORECASE,
)
NOT_A_PLACE = frozenset({"a", "an", "the", "my", "me", "us", "you", "it", "this", "that", "today", "tomorrow", "now"})


class Intent(str, Enum):
    LOCAL_REASONING = "local_reasoning"
    ONLINE_LOOKUP = "online_factual_lookup"
    EXIT = "exit"


def route(text: str) -> Intent:
    if text.casefold().strip(" .!?") in EXIT_PHRASES:
        return Intent.EXIT
    if is_allowed_cloud_lookup(text):
        return Intent.ONLINE_LOOKUP
    return Intent.LOCAL_REASONING


def extract_place(text: str) -> str | None:
    for match in PLACE_AFTER_PREPOSITION.finditer(text):
        place = match.group(1).strip(" .'-")
        if place and place.split()[0].casefold() not in NOT_A_PLACE:
            return place.title()
    return None


def parse_lookup(text: str, default_place: str) -> LookupRequest:
    """Build the minimal request that may leave the device: kind, place, day."""
    lowered = text.casefold()
    if any(re.search(rf"\b{term}\b", lowered) for term in WEATHER_TERMS):
        kind = "weather"
    elif re.search(r"\bnews\b", lowered):
        kind = "news"
    else:
        kind = "search"
    day = "tomorrow" if re.search(r"\btomorrow\b", lowered) else "today"
    return LookupRequest(kind=kind, place=extract_place(text) or default_place, day=day)
