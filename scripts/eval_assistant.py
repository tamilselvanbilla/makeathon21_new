"""End-to-end test of the assistant's conversations: notes, reminders, schedule,
recall, follow-ups, forgetting and restarts, through the real components.

Usage (from the repository root; run it on the Raspberry Pi for real latencies):
    .venv/bin/python scripts/eval_assistant.py [--only note_recall reminder_due] [--save results.json]

Uses the configured local model (LLM_MODEL_PATH etc.), knowledge_base/personal_data.json
and the memory embedding model. Memory lives in a temporary folder, never data/.
Online lookups are off. The clock is simulated so "tomorrow" and reminders are
repeatable. Each reply is scored against the patterns in
tests/data/assistant_scenarios.json; latency and route come from the tracer.
"""

import argparse
import contextlib
import io
import json
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from companion.app import load_embedder  # noqa: E402
from companion.brain.knowledge import KnowledgeBase, load_knowledge  # noqa: E402
from companion.brain.llm import LocalLLM  # noqa: E402
from companion.brain.memory import ConversationMemory  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.device.indicator import Indicator  # noqa: E402
from companion.device.tts import PrintSpeaker  # noqa: E402
from companion.online_gateway import OnlineGateway  # noqa: E402
from companion.pipeline import Assistant  # noqa: E402
from companion.tracing import Tracer, set_turn  # noqa: E402

SCENARIOS = ROOT / "tests" / "data" / "assistant_scenarios.json"
START = datetime(2026, 10, 9, 10, 0)  # a Friday morning


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class QuietIndicator(Indicator):
    def show(self, state) -> None:
        pass


def check(reply: str, step: dict) -> list[str]:
    """Problems with a reply; empty when it passes."""
    problems = [f"missing /{p}/" for p in step.get("all", []) if not re.search(p, reply, re.I)]
    if step.get("any") and not any(re.search(p, reply, re.I) for p in step["any"]):
        problems.append(f"none of {step['any']}")
    problems += [f"forbidden /{p}/" for p in step.get("forbid", []) if re.search(p, reply, re.I)]
    return problems


class Harness:
    def __init__(self, config: AppConfig):
        print(f"Loading '{config.llm.name}' ({config.llm.threads} threads)...")
        self.config = config
        self.llm = LocalLLM(config.llm)
        records = load_knowledge(config.knowledge.path)
        self.knowledge = KnowledgeBase(records, primary_user=config.knowledge.primary_user,
                                       top_k=config.knowledge.top_k, currency=config.knowledge.currency)
        self.embedder = load_embedder(config)
        self.gateway = OnlineGateway(enabled=False)
        self.tracer = Tracer()
        self.warmed = False

    def assistant(self, path: Path, clock: Clock) -> Assistant:
        memory = ConversationMemory(path, clock=clock, embedder=self.embedder,
                                    min_similarity=self.config.memory.min_similarity,
                                    top_k=self.config.memory.top_k)
        assistant = Assistant(llm=self.llm, knowledge=self.knowledge, gateway=self.gateway,
                              indicator=QuietIndicator(), speaker=PrintSpeaker(), memory=memory,
                              name=self.config.assistant_name, tracer=self.tracer)
        if not self.warmed:
            self.llm.warm_up(assistant.system_prompt)
            self.warmed = True
        return assistant

    def run(self, scenario: dict) -> list[dict]:
        results = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite3"
            clock = Clock(START)
            assistant = self.assistant(path, clock)
            for step in scenario["steps"]:
                if "advance" in step:
                    clock.now += timedelta(minutes=step["advance"])
                    continue
                if step.get("restart"):
                    assistant = self.assistant(path, clock)
                    continue
                if step.get("announce") or step.get("briefing"):
                    said = assistant.due_announcement() if step.get("announce") else assistant.briefing()
                    expect = step.get("expect")
                    ok = (said == "") if expect is None else bool(re.search(expect, said, re.I))
                    results.append({"say": "(reminder check)" if step.get("announce") else "(startup briefing)",
                                    "reply": said, "ok": ok, "problems": [] if ok else [f"expected {expect!r}"],
                                    "ms": 0.0, "route": "reminder" if step.get("announce") else "briefing",
                                    "llm": False})
                    continue
                started = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()), self.tracer.turn():
                    set_turn(input=step["say"])
                    reply = assistant.respond(step["say"])
                    set_turn(reply=reply)
                ms = (time.perf_counter() - started) * 1000
                trace = self.tracer.get(self.tracer.recent(1)[0]["id"])
                llm = [s for s in trace["spans"] if s["name"] == "llm_generation"]
                problems = check(reply, step)
                results.append({
                    "say": step["say"], "reply": reply, "ok": not problems, "problems": problems,
                    "known_miss": step.get("known_miss"),
                    "ms": round(ms), "route": trace["route"], "llm": bool(llm),
                    "prompt_tokens": llm[-1]["attrs"].get("prompt_tokens") if llm else None,
                    "completion_tokens": llm[-1]["attrs"].get("completion_tokens") if llm else None,
                })
        return results


def seconds(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="scenario ids")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    data = json.loads(SCENARIOS.read_text())
    scenarios = [s for s in data["scenarios"] if not args.only or s["id"] in args.only]
    harness = Harness(AppConfig.from_env())

    all_results = {}
    for scenario in scenarios:
        print(f"\n== {scenario['id']} ({scenario['category']})")
        results = harness.run(scenario)
        all_results[scenario["id"]] = {"category": scenario["category"], "steps": results}
        for r in results:
            mark = "PASS" if r["ok"] else ("KNOWN" if r.get("known_miss") else "FAIL")
            print(f"  {mark} {seconds(r['ms']):>8}  {r['route']:16} You: {r['say']}")
            print(f"  {'':4} {'':8}  {'':16} Sam: {r['reply']}")
            if not r["ok"]:
                print(f"  {'':4} {'':8}  {'':16} -> {'; '.join(r['problems'])}")

    every = [r for s in all_results.values() for r in s["steps"]]
    known = [r for r in every if r.get("known_miss")]
    steps = [r for r in every if not r.get("known_miss")]
    print(f"\n{'=' * 70}\nOverall: {sum(r['ok'] for r in steps)}/{len(steps)} steps passed"
          f" (+ {len(known)} known limitation(s), {sum(r['ok'] for r in known)} passed)")
    by_category: dict[str, list[bool]] = {}
    for s in all_results.values():
        by_category.setdefault(s["category"], []).extend(r["ok"] for r in s["steps"] if not r.get("known_miss"))
    for category, oks in by_category.items():
        print(f"  {category:10} {sum(oks)}/{len(oks)}")
    scenario_ok = sum(all(r["ok"] or r.get("known_miss") for r in s["steps"]) for s in all_results.values())
    print(f"  scenarios fully passed: {scenario_ok}/{len(all_results)}")

    turns = [r for r in every if r["say"][0] != "("]
    for label, group in (("with the model", [r for r in turns if r["llm"]]), ("without the model", [r for r in turns if not r["llm"]])):
        if group:
            ms = [r["ms"] for r in group]
            print(f"Latency {label:18} n={len(ms):3}  median {seconds(statistics.median(ms))}  "
                  f"p95 {seconds(percentile(ms, 0.95))}  worst {seconds(max(ms))}")
    tokens = [r for r in turns if r["llm"]]
    if tokens:
        print(f"Model calls: {len(tokens)}/{len(turns)} turns; prompt tokens median "
              f"{statistics.median(r['prompt_tokens'] for r in tokens):.0f}, reply tokens median "
              f"{statistics.median(r['completion_tokens'] for r in tokens):.0f}")
    routes: dict[str, int] = {}
    for r in turns:
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    print("Routes:", ", ".join(f"{k} {v}" for k, v in sorted(routes.items(), key=lambda kv: -kv[1])))

    if args.save:
        args.save.write_text(json.dumps({"model": harness.config.llm.name, "results": all_results}, indent=1))
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
