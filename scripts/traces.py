"""View the assistant's conversation traces (data/traces.sqlite3).

Usage (from the repository root):
    .venv/bin/python scripts/traces.py list [-n 20]       recent turns
    .venv/bin/python scripts/traces.py show [ID|last]     one turn as a waterfall
    .venv/bin/python scripts/traces.py stats [--hours 24] routes, latency percentiles, LLM speed
    .venv/bin/python scripts/traces.py serve [--port 8765] local web dashboard

The dashboard listens on 127.0.0.1 only. To view the Pi's traces from a laptop:
    ssh -L 8765:127.0.0.1:8765 <user>@<pi> 'cd ~/makeathon21_new && .venv/bin/python scripts/traces.py serve'
then open http://127.0.0.1:8765. Nothing is sent anywhere else; the page has no external assets.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from companion.tracing import Tracer  # noqa: E402

DEFAULT_PATH = Path(os.getenv("TRACE_FILE", str(ROOT / "data" / "traces.sqlite3")))
DASHBOARD = Path(__file__).with_name("trace_dashboard.html")
BAR_WIDTH = 40


def open_traces(path: Path) -> Tracer:
    if not path.is_file():
        sys.exit(f"No traces yet at {path}. Run the assistant first (TRACING=1, the default).")
    return Tracer(path, retention_days=36500)  # reading only: never prune here


def seconds(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def print_list(tracer: Tracer, limit: int) -> None:
    rows = tracer.recent(limit)
    print(f"{'time':19}  {'id':8}  {'route':16} {'reply after':>11} {'total':>7}  {'status':6}  question")
    for row in reversed(rows):
        question = (row["input"] or "").replace("\n", " ")
        print(
            f"{row['started'].replace('T', ' '):19}  {row['id']:8}  {row['route'] or '-':16} "
            f"{seconds(row['response_ms']):>11} {seconds(row['total_ms']):>7}  {row['status']:6}  "
            f"{question[:60]}"
        )


def print_trace(trace: dict) -> None:
    print(f"Trace {trace['id']}  {trace['started'].replace('T', ' ')}  route={trace['route'] or '-'}  "
          f"status={trace['status']}  reply after {seconds(trace['response_ms'])}  total {seconds(trace['total_ms'])}")
    if trace["input"]:
        print(f"  You:       {trace['input']}")
    if trace["reply"]:
        print(f"  Assistant: {trace['reply']}")
    if trace["attrs"]:
        print(f"  {json.dumps(trace['attrs'])}")
    total = max(trace["total_ms"] or 1, 1)
    depth: dict[int, int] = {}
    print()
    for s in trace["spans"]:
        depth[s["seq"]] = depth.get(s["parent"], -1) + 1 if s["parent"] is not None else 0
        offset = int(s["start_ms"] / total * BAR_WIDTH)
        width = max(1, int(s["duration_ms"] / total * BAR_WIDTH))
        bar = " " * offset + "█" * min(width, BAR_WIDTH - offset)
        name = "  " * depth[s["seq"]] + s["name"] + (" ✗" if s["status"] != "ok" else "")
        attrs = ", ".join(f"{k}={v}" for k, v in s["attrs"].items() if k not in ("status",))
        print(f"  {name:22} {bar:{BAR_WIDTH}} {seconds(s['duration_ms']):>7}  {attrs}")


def print_stats(stats: dict) -> None:
    print(f"Turns: {stats['turns']}   status: {stats['statuses']}")
    r = stats["response_ms"]
    if r.get("n"):
        print(f"Reply after speech: median {seconds(r['p50'])}, p95 {seconds(r['p95'])}, worst {seconds(r['max'])}")
    print("\nRoutes:")
    for route, count in stats["routes"].items():
        print(f"  {route:18} {count:4}  {'▇' * max(1, round(count / max(stats['turns'], 1) * 30))}")
    print(f"\n{'Step':18} {'n':>5} {'median':>8} {'p95':>8} {'worst':>8}")
    for name, p in stats["steps"].items():
        print(f"  {name:16} {p['n']:5} {seconds(p['p50']):>8} {seconds(p['p95']):>8} {seconds(p['max']):>8}")
    llm = stats["llm"]
    if llm["calls"]:
        pt, ct, sp = llm["prompt_tokens"], llm["completion_tokens"], llm["tokens_per_s"]
        print(f"\nLLM calls: {llm['calls']}   prompt tokens median {pt.get('p50')} (p95 {pt.get('p95')})   "
              f"reply tokens median {ct.get('p50')}   overall speed median {sp.get('p50')} tok/s (includes reading the prompt)")


def serve(path: Path, host: str, port: int) -> None:
    tracer = open_traces(path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(json.dumps(data).encode(), "application/json", status)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                self._send(DASHBOARD.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/traces":
                self._json(tracer.recent(int(query.get("limit", ["50"])[0])))
            elif url.path.startswith("/api/traces/"):
                trace = tracer.get(url.path.rsplit("/", 1)[-1])
                self._json(trace if trace else {"error": "not found"}, 200 if trace else 404)
            elif url.path == "/api/stats":
                hours = float(query.get("hours", ["0"])[0])
                self._json(tracer.stats(datetime.now() - timedelta(hours=hours) if hours else None))
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, *args) -> None:  # keep the console quiet
            pass

    server = HTTPServer((host, port), Handler)
    print(f"Trace dashboard on http://{host}:{port}  (Ctrl+C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH, help="traces database")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("-n", type=int, default=20)
    show = commands.add_parser("show")
    show.add_argument("id", nargs="?", default="last")
    stats = commands.add_parser("stats")
    stats.add_argument("--hours", type=float, default=0, help="only the last N hours (0: all)")
    serving = commands.add_parser("serve")
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.command == "serve":
        serve(args.file, args.host, args.port)
        return
    tracer = open_traces(args.file)
    if args.command == "list":
        print_list(tracer, args.n)
    elif args.command == "show":
        trace_id = args.id
        if trace_id == "last":
            recent = tracer.recent(1)
            if not recent:
                sys.exit("No traces yet.")
            trace_id = recent[0]["id"]
        trace = tracer.get(trace_id)
        if trace is None:
            sys.exit(f"No trace {trace_id}.")
        print_trace(trace)
    else:
        print_stats(tracer.stats(datetime.now() - timedelta(hours=args.hours) if args.hours else None))


if __name__ == "__main__":
    main()
