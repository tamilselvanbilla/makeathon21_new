"""Benchmark candidate language models on the assistant's real job.

Usage (from the repository root):
    .venv/bin/python scripts/eval_llm.py                       # every downloaded model
    .venv/bin/python scripts/eval_llm.py --models qwen3-0.6b qwen3-1.7b --threads 4

Each case in tests/data/llm_eval.json is rendered with the app's own prompts
(system prompt, record retrieval, memory, previous question, online facts) and
scored automatically. Every model runs in a fresh process so its RAM is
measured cleanly. Decoding is greedy (temperature 0) so results are repeatable.
Run it on the Raspberry Pi for real Pi timings.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CASES = ROOT / "tests" / "data" / "llm_eval.json"
MODELS_DIR = ROOT / "models" / "llm"


@dataclass(frozen=True)
class Candidate:
    file: str
    chat_format: str
    licence: str
    no_think: bool = False  # Qwen3: skip the reasoning trace
    system_in_user: bool = False  # Gemma has no system role: merge it into the first user turn
    max_tokens: int = 128


CANDIDATES = {
    "qwen3-0.6b": Candidate("Qwen3-0.6B-Q4_K_M.gguf", "chatml", "Apache-2.0", no_think=True),
    "qwen2.5-0.5b": Candidate("qwen2.5-0.5b-instruct-q4_k_m.gguf", "chatml", "Apache-2.0"),
    "llama3.2-1b": Candidate("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama-3", "Llama 3.2 Community"),
    "gemma3-1b": Candidate("gemma-3-1b-it-Q4_K_M.gguf", "gemma", "Gemma Terms", system_in_user=True),
    "qwen2.5-1.5b": Candidate("qwen2.5-1.5b-instruct-q4_k_m.gguf", "chatml", "Apache-2.0"),
    "smollm2-1.7b": Candidate("smollm2-1.7b-instruct-q4_k_m.gguf", "chatml", "Apache-2.0"),
    "qwen3-1.7b": Candidate("Qwen3-1.7B-Q4_K_M.gguf", "chatml", "Apache-2.0", no_think=True),
    # Same models with their reasoning trace: better at multi-step questions, far slower.
    "qwen3-0.6b-think": Candidate("Qwen3-0.6B-Q4_K_M.gguf", "chatml", "Apache-2.0", max_tokens=1024),
    "qwen3-1.7b-think": Candidate("Qwen3-1.7B-Q4_K_M.gguf", "chatml", "Apache-2.0", max_tokens=1024),
}


def rss_mb() -> float:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024


def render(case: dict, knowledge, records: dict) -> tuple[str, str]:
    """The system and user messages the app would send for this case."""
    from companion.brain.knowledge import _format_record
    from companion.brain.prompts import build_advice_prompt, build_system_prompt, build_user_prompt

    system = build_system_prompt(knowledge.primary_user, knowledge.currency, "Sam", knowledge.family)
    if case.get("online_facts"):
        return system, build_advice_prompt(case["q"], case["online_facts"], "")
    if case.get("context") == "app":
        context = knowledge.context_for(case["q"])
    elif case.get("records"):
        lines = []
        for category, record_type, owner in case["records"]:
            entry = next(e for e in records[category] if e["record_type"] == record_type and e["owner"] == owner)
            lines.append(f"{category.title()} record of {owner}: {_format_record(entry)}")
        context = "\n".join(lines)
    else:
        context = ""
    return system, build_user_prompt(case["q"], context, case.get("memory", ""), case.get("earlier", ""))


def score(case: dict, answer: str) -> dict:
    found = lambda pattern: re.search(pattern, answer, re.IGNORECASE)  # noqa: E731
    correct = (
        all(found(p) for p in case.get("all", []))
        and (not case.get("any") or any(found(p) for p in case["any"]))
        and not any(found(p) for p in case.get("forbid", []))
    )
    style = {
        "concise": len(re.findall(r"[.!?](?:\s|$)", answer)) <= 3,
        "no_leak": not re.search(r"record type|record of|knowledge:|[{}]|<think>", answer, re.IGNORECASE),
        "currency": not re.search(r"[$£€]|\busd\b|dollars|pounds", answer, re.IGNORECASE),
        "addresses_you": not re.match(r"\s*(my|i)\b", answer, re.IGNORECASE),
    }
    return {"correct": bool(correct), "style": style}


def worker(name: str, threads: int) -> dict:
    """Run every case for one model; called in a fresh process."""
    from llama_cpp import Llama

    from companion.brain.knowledge import KnowledgeBase, load_knowledge
    from companion.brain.llm import clean_model_response

    spec = CANDIDATES[name]
    base = rss_mb()
    start = time.perf_counter()
    llm = Llama(
        model_path=str(MODELS_DIR / spec.file),
        chat_format=spec.chat_format,
        n_ctx=2048,
        n_threads=threads,
        n_batch=256,
        seed=42,
        verbose=False,
    )
    load_s = time.perf_counter() - start
    records = load_knowledge()
    knowledge = KnowledgeBase(records)

    results = []
    for case in json.loads(CASES.read_text())["cases"]:
        system, user = render(case, knowledge, records)
        if spec.no_think:
            user += "\n/no_think"
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}] if spec.system_in_user else [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        start = time.perf_counter()
        first = None
        raw = ""
        for chunk in llm.create_chat_completion(
            messages=messages, max_tokens=spec.max_tokens, temperature=0.0, repeat_penalty=1.1, stream=True
        ):
            delta = chunk["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - start
            raw += delta
        total = time.perf_counter() - start
        answer = clean_model_response(raw)
        tokens = len(llm.tokenize(raw.encode("utf-8"), add_bos=False)) if raw else 0
        results.append({
            "id": case["id"], "category": case["category"], "answer": answer,
            "seconds": round(total, 2), "first_token_s": round(first or total, 2), "tokens": tokens,
            **score(case, answer),
        })
    return {
        "model": name, "file_mb": round((MODELS_DIR / spec.file).stat().st_size / 1e6), "licence": spec.licence,
        "load_s": round(load_s, 2), "ram_mb": round(rss_mb() - base), "results": results,
    }


def summarise(run: dict) -> dict:
    results = run["results"]
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["correct"])
    styles = [all(r["style"].values()) for r in results]
    gen_rates = [r["tokens"] / max(r["seconds"] - r["first_token_s"], 1e-6) for r in results if r["tokens"] > 1]
    return {
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "categories": {c: f"{sum(v)}/{len(v)}" for c, v in by_category.items()},
        "style": sum(styles) / len(styles),
        "median_s": sorted(r["seconds"] for r in results)[len(results) // 2],
        "first_token_s": sorted(r["first_token_s"] for r in results)[len(results) // 2],
        "tokens_per_s": sorted(gen_rates)[len(gen_rates) // 2] if gen_rates else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(CANDIDATES))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--save", type=Path, help="write raw results (all answers) to this JSON file")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(worker(args.worker, args.threads)))
        return

    names = args.models or [n for n, c in CANDIDATES.items() if (MODELS_DIR / c.file).is_file()]
    runs = []
    for name in names:
        print(f"running {name} ...", file=sys.stderr, flush=True)
        out = subprocess.run(
            [sys.executable, __file__, "--worker", name, "--threads", str(args.threads)],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout
        runs.append(json.loads(out.strip().splitlines()[-1]))
    if args.save:
        args.save.write_text(json.dumps(runs, indent=2))

    categories = list(dict.fromkeys(r["category"] for r in runs[0]["results"]))
    print(f"\n{len(runs[0]['results'])} cases, {args.threads} threads, greedy decoding\n")
    print("| Model | Licence | File | RAM | Accuracy | " + " | ".join(categories) + " | Style | Median reply | First token | Tokens/s |")
    print("|---" * (10 + len(categories)) + "|")
    for run in runs:
        s = summarise(run)
        cells = " | ".join(s["categories"][c] for c in categories)
        print(
            f"| {run['model']} | {run['licence']} | {run['file_mb']} MB | {run['ram_mb']} MB | {s['accuracy']:.0%} | {cells} | "
            f"{s['style']:.0%} | {s['median_s']:.1f} s | {s['first_token_s']:.1f} s | {s['tokens_per_s']:.0f} |"
        )


if __name__ == "__main__":
    main()
