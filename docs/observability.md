# Observability and traces

Every conversation turn is traced on the device: which path the request took, how long
each step ran, how many records and memories were found, how many tokens the model read
and wrote, and whether a guard or a fallback fired. Traces answer "why was that reply
slow?" and "why did it say that?" without attaching a debugger to the Pi.

Nothing leaves the device: traces go to a local SQLite file, and the dashboard is served
on `127.0.0.1` with no external assets.

## What a trace contains

One trace per turn, made of spans:

| Span | Attributes |
|---|---|
| `listen` | `audio_seconds` (includes the wait for speech to start) |
| `stt` | Whisper model size |
| `knowledge` | `records` found, `sources` as category/owner (`financial/John`), never the record contents |
| `memory` | `items` found, their `kinds` (note/turn) and whether each was an `exact` keyword match |
| `schedule` | `day` asked about, `items` listed |
| `online_lookup` | `kind` (weather/market/news); failures marked as errors |
| `llm_generation` | `prompt_tokens`, `completion_tokens`, `tokens_per_s` (overall, including reading the prompt), `finish` reason |
| `speak` | `chars` spoken |

And per turn:

| Field | Meaning |
|---|---|
| `route` | How it was answered: `llm`, `note`, `note_hedged`, `not_in_records`, `schedule`, `offer_note`, `memory_remember`, `memory_forget_all`, `pending_confirm`, `weather`, `market`, `market_fallback`, `news`, `reminder`, `ignored`, `exit` |
| `response_ms` | **Reply time**: from the end of the user's speech (or typed request) to the reply starting to play |
| `status` | `ok` or `error` (a step raised, or the turn failed) |
| `input`, `reply` | The question and reply text, unless `TRACE_CONTENT=0` |
| attributes | `follow_up`, `remembered` (stored as a past exchange), `uncertain`, `guard` (`example_figure` / `false_promise`, with the model's rejected answer), `online_failed`, `ignored` (`unclear audio` / `too short`) |

Not stored: audio, prompts, record contents, and the text of ignored speech. Silent
listening timeouts are not traced at all. "Forget everything" deletes all traces, and traces
older than `TRACE_RETENTION_DAYS` are deleted at startup.

## Viewing traces

**On the console**, one line after each turn:

```
[TRACE b30fd672] llm | knowledge 2ms · memory 16ms · llm 8.6s (351 prompt -> 18 reply tokens, 2.1 tok/s overall) · speak 0ms | reply after 8.6s
```

**From the command line:**

```bash
.venv/bin/python scripts/traces.py list -n 20     # recent turns: route, reply time, question
.venv/bin/python scripts/traces.py show last      # one turn as a waterfall with every span's attributes
.venv/bin/python scripts/traces.py show b30fd672
.venv/bin/python scripts/traces.py stats --hours 24   # routes, median/p95 per step, LLM tokens
```

**In a browser** (turn list, waterfall per turn, route mix, step latency percentiles;
refreshes every 5 s):

```bash
.venv/bin/python scripts/traces.py serve          # http://127.0.0.1:8765
```

On the Pi it listens on localhost only. To open it from a laptop, tunnel over SSH:

```bash
ssh -L 8765:127.0.0.1:8765 mr4vibes@<pi-address> \
  'cd ~/makeathon21_new && .venv/bin/python scripts/traces.py serve'
# then open http://127.0.0.1:8765 on the laptop
```

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TRACING` | `1` | `0` turns tracing off |
| `TRACE_FILE` | `data/traces.sqlite3` | Where traces are stored (with `MEMORY=0`, kept in RAM only) |
| `TRACE_CONTENT` | `1` | `0` stores timings and routes without the question and reply text |
| `TRACE_CONSOLE` | `1` | `0` hides the `[TRACE]` line |
| `TRACE_RETENTION_DAYS` | `30` | Older traces are deleted at startup |

## Reading the numbers (Pi 4, Qwen3-0.6B)

- A first measured session: a records question took 8.6 s, almost all of it the model
  **reading the 351-token prompt** (only 18 tokens were generated). Shorter prompts
  (fewer records, a shorter system prompt) cut reply time more than a faster sampler would.
- Answers from notes, schedules and "not in your records" take milliseconds: no model call.
- An online lookup adds ~3 s on the Pi's network; the model's advice sentence after it
  costs another ~10 s.

## For developers

`companion/tracing.py` has no dependencies. `span(name, **attrs)` and `annotate(**attrs)`
can be called from any module; outside a turn they do nothing. `telemetry.timed_event`
opens a span too, so anything already timed (Whisper, the model, online lookups) appears in
traces automatically. `Assistant.run` opens one trace per turn with `Tracer.turn()`.

```python
from companion.tracing import annotate, span

with span("rerank", candidates=len(items)):
    best = rerank(items)
    annotate(kept=len(best))
```
