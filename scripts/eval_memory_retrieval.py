"""Measure memory retrieval: keyword (FTS5), embeddings, and hybrid.

Usage (from the repository root):
    .venv/bin/python scripts/eval_memory_retrieval.py [--models minilm-int8 bge-small] [--threads 2]

Reports, per method: accuracy on questions that should find a note (by kind)
and on questions that must find nothing; plus model load time, RAM, file size,
and per-question embedding time. Run it on the Pi for real Pi numbers.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from companion.brain.embeddings import MODELS, Embedder  # noqa: E402
from companion.brain.memory import ConversationMemory  # noqa: E402

RRF_K = 60
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.80, 0.05)]


def rss_mb() -> float:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024


def ram_in_fresh_process(name: str, threads: int) -> int:
    """RAM added by loading the model and embedding once, measured in a clean process
    (onnxruntime's own start-up is counted, as it would be on the device)."""
    code = (
        "import os, subprocess, sys; sys.path.insert(0, 'src');"
        "rss = lambda: int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])) / 1024;"
        "import numpy; base = rss();"
        "from pathlib import Path; from companion.brain.embeddings import Embedder;"
        f"e = Embedder(Path('models/embedding'), '{name}', threads={threads}); e.embed(['warm up'], query=True);"
        "print(round(rss() - base))"
    )
    return int(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, stderr=subprocess.DEVNULL))


def score(predictions: dict[str, str | None], queries: list[dict]) -> dict:
    by_kind: dict[str, list[bool]] = {}
    for item in queries:
        by_kind.setdefault(item["kind"], []).append(predictions[item["q"]] == item["expect"])
    positives = [p for item in queries if item["expect"] for p in [predictions[item["q"]] == item["expect"]]]
    negatives = [p for item in queries if not item["expect"] for p in [predictions[item["q"]] is None]]
    return {
        "find": sum(positives) / len(positives),
        "reject": sum(negatives) / len(negatives),
        "balanced": (sum(positives) / len(positives) + sum(negatives) / len(negatives)) / 2,
        "kinds": {kind: f"{sum(v)}/{len(v)}" for kind, v in by_kind.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["minilm-int8"], choices=sorted(MODELS))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    data = json.loads((ROOT / "tests" / "data" / "memory_retrieval_eval.json").read_text())
    note_ids = list(data["notes"])
    note_texts = [data["notes"][i] for i in note_ids]
    queries = data["queries"]

    memory = ConversationMemory(top_k=len(note_ids))
    # Notes are stored with relative dates resolved, so map them back by the stored text.
    text_to_id = {memory.add_note(text)[0]: note_id for text, note_id in zip(note_texts, note_ids)}
    lexical = {item["q"]: [text_to_id[m.question] for m in memory.search(item["q"])] for item in queries}
    # Per question and note: does the note contain all / some / none of the question's words?
    coverage = {item["q"]: {text_to_id[t]: c for t, c in memory.coverage(item["q"]).items()} for item in queries}

    results = {"keyword (FTS5)": score({q: (ids[0] if ids else None) for q, ids in lexical.items()}, queries)}
    costs = {}

    for name in args.models:
        start = time.perf_counter()
        embedder = Embedder(ROOT / "models" / "embedding", name, threads=args.threads)
        load_s = time.perf_counter() - start
        note_vecs = embedder.embed(note_texts)

        timings = []
        query_vecs = []
        for item in queries:
            start = time.perf_counter()
            query_vecs.append(embedder.embed([item["q"]], query=True)[0])
            timings.append((time.perf_counter() - start) * 1000)
        sims = np.array(query_vecs) @ note_vecs.T  # (queries, notes)
        spec = MODELS[name]
        costs[name] = {
            "file MB": round((ROOT / "models" / "embedding" / spec.directory / spec.onnx_file).stat().st_size / 1e6),
            "load s": round(load_s, 2),
            "RAM MB": ram_in_fresh_process(name, args.threads),
            "query ms (median)": round(float(np.median(timings)), 1),
            "query ms (p95)": round(float(np.percentile(timings, 95)), 1),
        }

        best: dict[str, tuple[str, dict]] = {}
        for tau in THRESHOLDS:
            vector, hybrid_or, hybrid_veto, hybrid_cover = {}, {}, {}, {}
            for qi, item in enumerate(queries):
                q, row = item["q"], sims[qi]
                order = list(np.argsort(-row))
                top = order[0]
                vector[q] = note_ids[top] if row[top] >= tau else None

                lex_rank = {nid: r for r, nid in enumerate(lexical[q])}
                fused = {
                    nid: (1 / (RRF_K + lex_rank[nid]) if nid in lex_rank else 0)
                    + 1 / (RRF_K + order.index(note_ids.index(nid)))
                    for nid in note_ids
                }
                cos = dict(zip(note_ids, row))
                # Hybrid A: keyword match OR strong semantic match, ranked by fusion.
                cands = [n for n in note_ids if n in lex_rank or cos[n] >= tau]
                hybrid_or[q] = max(cands, key=fused.get) if cands else None
                # Hybrid B: as A, but a keyword match must also be somewhat similar in meaning.
                cands = [n for n in note_ids if (n in lex_rank and cos[n] >= tau - 0.15) or cos[n] >= tau]
                hybrid_veto[q] = max(cands, key=fused.get) if cands else None
                # Hybrid C: accept a note that contains every word of the question, or one that
                # shares no words but is close in meaning; a partial overlap is a near-miss.
                cover = coverage[q]
                cands = [n for n in note_ids if cover[n] == "all" or (cover[n] == "none" and cos[n] >= tau)]
                hybrid_cover[q] = max(cands, key=lambda n: (cover[n] == "all", cos[n])) if cands else None

            for label, preds in (
                ("embeddings", vector),
                ("hybrid A", hybrid_or),
                ("hybrid B", hybrid_veto),
                ("hybrid C", hybrid_cover),
            ):
                result = score(preds, queries)
                key = f"{label} [{name}]"
                if key not in best or result["balanced"] > best[key][1]["balanced"]:
                    best[key] = (f"threshold {tau}", result)
        for key, (setting, result) in best.items():
            results[f"{key} ({setting})"] = result

        # The search the assistant actually uses, with its default threshold.
        shipped = ConversationMemory(top_k=1, embedder=embedder)
        for text in note_texts:
            shipped.add_note(text)
        found = {item["q"]: shipped.search(item["q"]) for item in queries}
        results[f"shipped hybrid [{name}] (threshold {shipped.min_similarity})"] = score(
            {q: (text_to_id[hits[0].question] if hits else None) for q, hits in found.items()}, queries
        )
        wrong = [item for item in queries if (found[item["q"]] and text_to_id[found[item["q"]][0].question] != item["expect"])]
        hedged = sum(1 for item in wrong if not found[item["q"]][0].exact)
        costs[name]["wrong answers hedged"] = f"{hedged}/{len(wrong)}"

    positives = sum(1 for q in queries if q["expect"])
    print(f"\n{len(note_ids)} notes, {positives} questions with an answer, {len(queries) - positives} without\n")
    print(f"{'method':58} {'find':>6} {'reject':>7} {'balanced':>9}")
    for label, r in results.items():
        print(f"{label:58} {r['find']:6.0%} {r['reject']:7.0%} {r['balanced']:9.0%}")
    print("\nper kind (correct/total):")
    for label, r in results.items():
        print(f"  {label:56} {r['kinds']}")
    print(f"\ncosts ({args.threads} threads):")
    for name, c in costs.items():
        print(f"  {name:12} {c}")


if __name__ == "__main__":
    main()
