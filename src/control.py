"""Local device controls and mute state."""

from typing import Protocol


class ListenerState(Protocol):
    def set_listening(self, active: bool) -> None: ...


def is_muted(muted: bool) -> bool:
    """Normalize the physical mute state."""
    return bool(muted)


class ConsoleListenerState:
    """Visible local listening indicator suitable for a desk prototype."""

    def __init__(self, output=None):
        self.output = output or print

    def set_listening(self, active: bool) -> None:
        state = "LISTENING" if active else "IDLE"
        self.output(f"[LOCAL] Listening indicator: {state}")

    def set_status(self, status: str) -> None:
        self.output(f"[LOCAL] Listening indicator: {status}")


class PhysicalMuteSwitch:
    """A physical mute switch abstraction with optional GPIO support."""

    def __init__(self, initial_muted: bool = False):
        self.muted = initial_muted

    def toggle(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def set_muted(self, muted: bool) -> None:
        self.muted = is_muted(muted)

    def is_muted(self) -> bool:
        return self.muted
