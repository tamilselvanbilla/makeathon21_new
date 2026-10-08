"""Privacy-aware timing and operational logs for the voice assistant."""

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "assistant.log"


def log_event(event: str, *, duration_ms: float | None = None, **details: Any) -> None:
    """Append one JSON record without recording raw user prompts."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
    }
    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 3)
    record.update(details)

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, default=str) + "\n")


def timed_event(event: str, **details: Any):
    """Context manager that logs elapsed time and status."""
    return _TimedEvent(event, details)


class _TimedEvent:
    def __init__(self, event: str, details: dict[str, Any]):
        self.event = event
        self.details = details
        self.started_at = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration_ms = (time.perf_counter() - self.started_at) * 1000
        status = "error" if exc_type is not None else "ok"
        self.details["status"] = status
        log_event(
            self.event,
            duration_ms=duration_ms,
            **self.details,
        )
        return False
