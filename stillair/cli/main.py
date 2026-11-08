"""Command line: `python -m stillair <demo|serve|eval|report>`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ..backends.registry import BACKENDS, make_backend
from ..evaluation.experiments import run_all
from ..evaluation.report import to_markdown
from .demo import cmd_demo


def _write_report(res: dict, out_dir: str) -> None:
    md = to_markdown(res)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def cmd_eval(a):
    seeds = list(range(a.seeds))
    t = time.time()
    res = run_all(seeds, quick=a.quick, out_dir=a.out)
    _write_report(res, a.out)
    print(f"\n[done in {time.time() - t:.0f}s] wrote {a.out}/results.json and {a.out}/results.md")


def cmd_report(a):
    with open(os.path.join(a.out, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    _write_report(res, a.out)


def cmd_chat(a):
    from ..assistant.server import chat

    chat(a.port, a.memory)


def cmd_real(a):
    from ..evaluation.real_run import run_real

    print(run_real(make_backend(a.backend), a.backend, range(a.seeds), a.steps, a.out))
    print(f"wrote {a.out}/results.json and {a.out}/results.md")


def _load_dotenv(path: str = ".env") -> None:
    """Read STILLAIR_* settings from a gitignored .env file; real environment variables win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and key.startswith("STILLAIR_"):
                os.environ.setdefault(key, value.strip().strip("\"'"))


def cmd_serve(a):
    from ..live.server import serve

    serve(a.port, a.seed, make_backend(a.backend), a.token_budget)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # reports contain characters a Windows code page cannot encode
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv()
    p = argparse.ArgumentParser(prog="stillair", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="narrated terminal demo")
    d.add_argument("--seed", type=int, default=0, help="0 = success run, 3 = a run where a confabulated link slips through")
    d.add_argument("--steps", type=int, default=30)
    d.add_argument("--fast", action="store_true", help="no pauses")
    d.add_argument("--backend", choices=BACKENDS, default="sim", help="sim = offline simulated associator (default); openai = any OpenAI-compatible API")
    d.add_argument("--no-selector", action="store_true", help="show what happens without the selector (condition B)")
    d.add_argument("--hallucination", type=float, default=0.3)
    d.add_argument("--verifier-miss", type=float, default=0.15, help="how often the grounding judge is fooled")
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve", help="live web dashboard")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--backend", choices=BACKENDS, default="sim", help="sim = offline simulator (default); openai = any OpenAI-compatible API")
    s.add_argument("--token-budget", type=int, default=50_000, help="real models only: the idle loop stops after this many tokens")
    s.set_defaults(fn=cmd_serve)
    e = sub.add_parser("eval", help="run the three-condition experiment and write results/")
    e.add_argument("--seeds", type=int, default=20)
    e.add_argument("--quick", action="store_true")
    e.add_argument("--out", default="results")
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("report", help="rebuild results/results.md from results/results.json")
    r.add_argument("--out", default="results")
    r.set_defaults(fn=cmd_report)
    t = sub.add_parser("chat", help="chat assistant: main model on the API, small local model thinking between messages")
    t.add_argument("--port", type=int, default=8770)
    t.add_argument("--memory", default="stillair_memory.json", help="where the conversation and memories are saved")
    t.set_defaults(fn=cmd_chat)
    m = sub.add_parser("real", help="small A/C experiment sized for free API quotas")
    m.add_argument("--backend", choices=BACKENDS, default="openai")
    m.add_argument("--seeds", type=int, default=3)
    m.add_argument("--steps", type=int, default=10, help="idle steps per session")
    m.add_argument("--out", default=os.path.join("results", "real"))
    m.set_defaults(fn=cmd_real)
    args = p.parse_args(argv)
    args.fn(args)
