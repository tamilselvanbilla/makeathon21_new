"""Mute switch abstraction.

A GPIO-backed switch only needs to provide `is_muted()`.
"""

from typing import Protocol


class MuteSwitch(Protocol):
    def is_muted(self) -> bool: ...


class SoftwareMuteSwitch:
    """In-memory mute state for development machines without GPIO."""

    def __init__(self, initial_muted: bool = False):
        self.muted = bool(initial_muted)

    def toggle(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)

    def is_muted(self) -> bool:
        return self.muted
