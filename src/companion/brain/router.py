"""Decide how a transcribed request is handled.

Rule-based for now; a grammar-constrained LLM router can replace `route()`
without changing callers.
"""

from enum import Enum

from .policy import is_allowed_cloud_lookup

EXIT_PHRASES = frozenset({"quit", "exit", "stop", "goodbye"})


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
