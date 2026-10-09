"""Visible listening indicator states and the console implementation.

A GPIO/LED-ring implementation only needs to provide `show(state)`.
"""

from enum import Enum
from typing import Callable, Protocol


class IndicatorState(Enum):
    MUTED = "muted"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    ONLINE = "online"
    SPEAKING = "speaking"


class Indicator(Protocol):
    def show(self, state: IndicatorState) -> None: ...


class ConsoleIndicator:
    """Prints state changes; used when no LED hardware is attached."""

    def __init__(self, output: Callable[[str], None] = print):
        self.output = output
        self.state: IndicatorState | None = None

    def show(self, state: IndicatorState) -> None:
        if state is self.state:
            return
        self.state = state
        self.output(f"[LED] {state.name}")
