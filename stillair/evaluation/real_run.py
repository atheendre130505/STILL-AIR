"""A small experiment sized for free API quotas: A vs C vs C with all upgrades, on a few small worlds.

Works with any backend, so `--backend sim` gives a directly comparable simulator baseline.
Results are saved after each condition, so running out of quota keeps what finished.
"""

from __future__ import annotations

import json
import os
import statistics
import time

from .config import ALL_UPGRADES, HEADLINE_GENEROUS, HEADLINE_STRICT
from .episode import run_episode
from .stats import fmt, mean_ci

SMALL_WORLD = dict(n_chains=6, n_sessions=3, n_dangling_rel=3, n_dangling_prop=3, n_controls=2)
CONDITIONS = (("A  no idle loop", "A", {}), ("C  full Still Air", "C", {}), ("C + all upgrades", "C", ALL_UPGRADES))
_KEYS = {"strict": f"H{HEADLINE_STRICT[0]}k{HEADLINE_STRICT[1]}", "generous": f"H{HEADLINE_GENEROUS[0]}k{HEADLINE_GENEROUS[1]}"}


def _summary(rs: list) -> dict:
    out = {k: mean_ci([r.get(k) for r in rs]) for k in ("precision", "coverage", "connections", "idle_steps")}
    for name, key in _KEYS.items():
        for metric in ("acc", "wrong", "tokens", "control_acc"):
            out[f"{name}.{metric}"] = mean_ci([r[key][metric] for r in rs])
    return out


def to_markdown(res: dict) -> str:
    L = [f"# Still Air: small real-model run\n",
         f"Backend `{res['backend']}`, model `{res['model']}`, {len(res['seeds'])} small worlds, "
         f"{res['steps_per_session']} idle steps per session.\n",
         "| Condition | precision | coverage | acc (1 round) | acc (2 rounds) | wrong (2 rounds) | control | API calls | tokens | seconds |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["conditions"]:
        s = r["summary"]
        L.append(f"| {r['label']} | {fmt(s['precision'], True)} | {fmt(s['coverage'], True)} | {fmt(s['strict.acc'], True)} | "
                 f"{fmt(s['generous.acc'], True)} | {fmt(s['generous.wrong'], True)} | {fmt(s['generous.control_acc'], True)} | "
                 f"{r['calls']} | {r['tokens']} | {r['seconds']:.0f} |")
    return "\n".join(L) + "\n"


def _save(res: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    md = to_markdown(res)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def run_real(llm, backend: str, seeds, steps_per_session: int, out_dir: str) -> str:
    res = {"backend": backend, "model": getattr(llm, "model", "simulator"), "seeds": list(seeds),
           "steps_per_session": steps_per_session, "conditions": []}
    md = ""
    for label, mode, flags in CONDITIONS:
        calls, tokens, start = getattr(llm, "calls", 0), getattr(llm, "tokens_used", 0), time.monotonic()
        rs = [run_episode(s, mode, llm=llm, world=SMALL_WORLD, idle_steps_per_session=steps_per_session,
                          test_configs=[HEADLINE_STRICT, HEADLINE_GENEROUS], **flags) for s in seeds]
        res["conditions"].append({
            "label": label, "summary": _summary(rs),
            "calls": getattr(llm, "calls", 0) - calls, "tokens": getattr(llm, "tokens_used", 0) - tokens,
            "seconds": time.monotonic() - start,
        })
        md = _save(res, out_dir)
    return md
