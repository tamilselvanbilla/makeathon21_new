"""Traces of each conversation turn, stored on the device.

A trace is one turn: listening, transcription, routing, retrieval, the online
lookup, the language model, guards and speech, each a timed span with a few
attributes (counts, token numbers, the route taken). They go to a local SQLite
file (`data/traces.sqlite3`) and are shown on the console after each turn, by
`scripts/traces.py`, and on its local dashboard.

What is stored: timings, statuses, counts and the question and reply text (the
same text memory keeps; `TRACE_CONTENT=0` leaves it out). Never audio, prompts
or the contents of personal records. "Forget everything" also deletes traces.

`span()` and `annotate()` can be called anywhere; outside a turn they do nothing,
so modules don't need to know whether tracing is on.
"""

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass
class Span:
    seq: int
    parent: int | None
    name: str
    start: float  # perf_counter seconds
    end: float | None = None
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000


@dataclass
class Trace:
    id: str
    started: datetime
    start: float
    spans: list[Span] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    discarded: bool = False
    stack: list[Span] = field(default_factory=list)

    def discard(self) -> None:
        """Nothing was heard: don't keep this trace."""
        self.discarded = True

    def first(self, name: str) -> Span | None:
        return next((span for span in self.spans if span.name == name), None)

    def response_ms(self) -> float | None:
        """From the end of the user's speech (or the typed request) to the reply starting."""
        speak = self.first("speak")
        if speak is None:
            return None
        listen = self.first("listen")
        heard = listen.end if listen and listen.end else self.start
        return (speak.start - heard) * 1000


_active: ContextVar[Trace | None] = ContextVar("active_trace", default=None)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span | None]:
    """Time a step of the current turn; a no-op outside a turn."""
    trace = _active.get()
    if trace is None:
        yield None
        return
    parent = trace.stack[-1].seq if trace.stack else None
    current = Span(len(trace.spans), parent, name, time.perf_counter(), attrs=dict(attrs))
    trace.spans.append(current)
    trace.stack.append(current)
    try:
        yield current
    except BaseException as exc:
        current.status = "error"
        current.attrs["error"] = type(exc).__name__
        raise
    finally:
        current.end = time.perf_counter()
        trace.stack.pop()


def annotate(**attrs: Any) -> None:
    """Add attributes to the innermost open span (or the turn, if none is open)."""
    trace = _active.get()
    if trace is not None:
        (trace.stack[-1].attrs if trace.stack else trace.attrs).update(attrs)


def set_turn(**attrs: Any) -> None:
    """Attributes of the whole turn: route, input, reply."""
    trace = _active.get()
    if trace is not None:
        trace.attrs.update(attrs)


def current_trace() -> Trace | None:
    return _active.get()


class Tracer:
    def __init__(
        self,
        path: Path | None = None,
        enabled: bool = True,
        content: bool = True,
        console: bool = False,
        retention_days: int = 30,
        clock: Callable[[], datetime] = datetime.now,
    ):
        """`path=None` keeps traces in RAM (tests, or memory persistence off)."""
        self.enabled = enabled
        self.content = content
        self.console = console
        self._clock = clock
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(":memory:" if path is None else str(path))
        self._db.executescript(SCHEMA)
        cutoff = (clock() - timedelta(days=retention_days)).isoformat(timespec="seconds")
        old = "SELECT id FROM traces WHERE started < ?"
        self._db.execute(f"DELETE FROM spans WHERE trace_id IN ({old})", (cutoff,))
        self._db.execute("DELETE FROM traces WHERE started < ?", (cutoff,))
        self._db.commit()

    @contextmanager
    def turn(self, **attrs: Any) -> Iterator[Trace | None]:
        if not self.enabled:
            yield None
            return
        trace = Trace(secrets.token_hex(4), self._clock(), time.perf_counter(), attrs=dict(attrs))
        token = _active.set(trace)
        try:
            yield trace
        except BaseException as exc:
            trace.status = "error"
            trace.attrs["error"] = type(exc).__name__
            raise
        finally:
            _active.reset(token)
            if not trace.discarded:
                self.save(trace)
                if self.console:
                    print(summary_line(trace))

    def save(self, trace: Trace) -> None:
        total = (time.perf_counter() - trace.start) * 1000
        attrs = dict(trace.attrs)
        question, reply = attrs.pop("input", ""), attrs.pop("reply", "")
        if not self.content:
            question, reply = "", ""
        if any(s.status == "error" for s in trace.spans) and trace.status == "ok":
            trace.status = "error"
        self._db.execute(
            "INSERT INTO traces (id, started, total_ms, response_ms, route, status, input, reply, attrs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace.id, trace.started.isoformat(timespec="seconds"), round(total, 1),
                _round(trace.response_ms()), attrs.pop("route", ""), trace.status, question, reply,
                json.dumps(attrs, default=str),
            ),
        )
        self._db.executemany(
            "INSERT INTO spans (trace_id, seq, parent, name, start_ms, duration_ms, status, attrs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (trace.id, s.seq, s.parent, s.name, round((s.start - trace.start) * 1000, 1),
                 round(s.duration_ms, 1), s.status, json.dumps(s.attrs, default=str))
                for s in trace.spans
            ],
        )
        self._db.commit()

    def forget_all(self) -> int:
        count = self._db.execute("SELECT count(*) FROM traces").fetchone()[0]
        self._db.execute("DELETE FROM spans")
        self._db.execute("DELETE FROM traces")
        self._db.commit()
        return count

    # Reading, for scripts/traces.py and its dashboard.

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, started, total_ms, response_ms, route, status, input, reply, attrs "
            "FROM traces ORDER BY started DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_trace_row(row) for row in rows]

    def get(self, trace_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT id, started, total_ms, response_ms, route, status, input, reply, attrs FROM traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        trace = _trace_row(row)
        trace["spans"] = [
            {"seq": s[0], "parent": s[1], "name": s[2], "start_ms": s[3], "duration_ms": s[4],
             "status": s[5], "attrs": json.loads(s[6])}
            for s in self._db.execute(
                "SELECT seq, parent, name, start_ms, duration_ms, status, attrs FROM spans "
                "WHERE trace_id = ? ORDER BY seq", (trace_id,),
            )
        ]
        return trace

    def stats(self, since: datetime | None = None) -> dict:
        """Turn counts by route and status, and latency percentiles per step."""
        start = (since or datetime.min).isoformat(timespec="seconds")
        traces = self._db.execute(
            "SELECT route, status, response_ms FROM traces WHERE started >= ?", (start,)
        ).fetchall()
        durations: dict[str, list[float]] = {}
        for name, duration in self._db.execute(
            "SELECT s.name, s.duration_ms FROM spans s JOIN traces t ON t.id = s.trace_id WHERE t.started >= ?",
            (start,),
        ):
            durations.setdefault(name, []).append(duration)
        tokens = [
            json.loads(a) for (a,) in self._db.execute(
                "SELECT s.attrs FROM spans s JOIN traces t ON t.id = s.trace_id "
                "WHERE t.started >= ? AND s.name = 'llm_generation'", (start,),
            )
        ]
        speeds = [t["tokens_per_s"] for t in tokens if t.get("tokens_per_s")]
        routes: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for route, status, _ in traces:
            routes[route or "-"] = routes.get(route or "-", 0) + 1
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "turns": len(traces),
            "routes": dict(sorted(routes.items(), key=lambda kv: -kv[1])),
            "statuses": statuses,
            "response_ms": _percentiles([r for _, _, r in traces if r is not None]),
            "steps": {name: _percentiles(values) for name, values in sorted(durations.items())},
            "llm": {
                "calls": len(tokens),
                "prompt_tokens": _percentiles([t["prompt_tokens"] for t in tokens if "prompt_tokens" in t]),
                "completion_tokens": _percentiles([t["completion_tokens"] for t in tokens if "completion_tokens" in t]),
                "tokens_per_s": _percentiles(speeds),
            },
        }


SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY, started TEXT NOT NULL, total_ms REAL, response_ms REAL,
    route TEXT, status TEXT, input TEXT, reply TEXT, attrs TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL, seq INTEGER NOT NULL, parent INTEGER, name TEXT NOT NULL,
    start_ms REAL, duration_ms REAL, status TEXT, attrs TEXT,
    PRIMARY KEY (trace_id, seq)
);
CREATE INDEX IF NOT EXISTS traces_started ON traces (started);
"""

# Steps shown on the console line, in pipeline order.
CONSOLE_STEPS = ["listen", "stt", "knowledge", "memory", "online_lookup", "llm_generation", "speak"]
LABELS = {"llm_generation": "llm", "online_lookup": "online"}


def summary_line(trace: Trace) -> str:
    """[TRACE 3f2a91c0] llm | stt 4.1s · memory 3ms · llm 8.6s (351 prompt -> 18 reply tokens, 2.1 tok/s overall) · speak 2.0s | reply after 12.6s"""
    parts = []
    for name in CONSOLE_STEPS:
        spans = [s for s in trace.spans if s.name == name]
        if not spans:
            continue
        text = f"{LABELS.get(name, name)} {_seconds(sum(s.duration_ms for s in spans))}"
        llm = spans[-1].attrs
        if name == "llm_generation" and "completion_tokens" in llm:
            text += f" ({llm.get('prompt_tokens')} prompt -> {llm['completion_tokens']} reply tokens, {llm.get('tokens_per_s')} tok/s overall)"
        if any(s.status == "error" for s in spans):
            text += " FAILED"
        parts.append(text)
    response = trace.response_ms()
    tail = f" | reply after {_seconds(response)}" if response is not None else ""
    return f"[TRACE {trace.id}] {trace.attrs.get('route', '-')} | " + " · ".join(parts) + tail


def _seconds(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)

    return {"n": len(ordered), "p50": pick(0.5), "p95": pick(0.95), "max": round(ordered[-1], 1)}


def _trace_row(row: tuple) -> dict:
    keys = ["id", "started", "total_ms", "response_ms", "route", "status", "input", "reply"]
    trace = dict(zip(keys, row[:8]))
    trace["attrs"] = json.loads(row[8] or "{}")
    return trace
