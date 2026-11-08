"""Evaluation harness for the concept document's section 4 (hypothesis and evaluation).

Conditions (compute is matched in *tokens*, not steps; see `compute_matched`):
    A  no idle loop
    B  idle loop driven by random noise only, no selector
    C  full Still Air (perturbation + selector + grounding + memory writes)

Ablations: remove the selector, remove memory writes, remove the internal-state drive,
and vary the perturbation amplitude.

Each episode: the world's facts arrive over S sessions; between sessions the idle loop (if
any) gets a token budget; at the end the agent answers two-hop questions (chains) and
one-hop controls under several test-time budgets (hops x memories-read).

Ground truth lives in the world, never in the agent: stored connections are scored
afterwards against `world.valid_derived`.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

from .config import HEADLINE_GENEROUS, HEADLINE_STRICT, TEST_CONFIGS, DEFAULTS, UPGRADE_ROWS
from .episode import run_episode, run_focus_episode
from .stats import mean_ci


def _focus_row(rs, **fields):
    row = dict(fields)
    for k in ("asked_before", "asked_after", "unasked_before", "unasked_after"):
        row[k] = mean_ci([r[k] for r in rs])
    row["asked_gain"] = mean_ci([r["asked_after"] - r["asked_before"] for r in rs])
    row["unasked_gain"] = mean_ci([r["unasked_after"] - r["unasked_before"] for r in rs])
    return row


def focus_experiment(seeds, amps=(0.0, 0.25, 0.5, 0.75, 1.0), **over):
    rows = []
    cfgs = [(f"drive a={a}", a, True) for a in amps] + [("blind noise (no state)", 1.0, False)]
    for label, a, st in cfgs:
        rs = [run_focus_episode(s, a, st, **over) for s in seeds]
        rows.append(_focus_row(rs, label=label, amplitude=a, use_state=st))
    return rows


def upgrades_experiment(seeds, **over):
    """Condition C with each upgrade on its own, then all together; plus adaptive noise on targeted recall."""
    rows = []
    for label, flags in UPGRADE_ROWS:
        rows.append({"label": label, "summary": summarize(run_many(seeds, ["full"], **{**over, **flags}))["full"]})
    focus = [
        _focus_row([run_focus_episode(s, 0.5, True, adaptive_noise=adaptive, **over) for s in seeds], label=label)
        for label, adaptive in (("fixed amplitude 0.5", False), ("adaptive noise", True))
    ]
    return {"rows": rows, "focus": focus}


def _job(args):
    seed, mode, over = args
    return run_episode(seed, mode, **over)


def run_many(seeds, modes, workers=None, **over):
    jobs = [(s, m, over) for m in modes for s in seeds]
    workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if workers <= 1:
        res = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_job, jobs, chunksize=1))
    by: dict = {m: [] for m in modes}
    for r in res:
        by[r["mode"]].append(r)
    return by


def summarize(by: dict) -> dict:
    """Aggregate per-mode lists of episode dicts into mean +/- CI."""
    out = {}
    for mode, rs in by.items():
        row = {"n": len(rs)}
        for key in ("idle_tokens", "idle_steps", "connections", "valid_unique", "invalid", "redundant",
                    "precision", "coverage", "memory_size", "repetition_rate", "incoherence_rate",
                    "drift_failure_rate", "accepted_per_1k_tokens", "pruned"):
            vals = [r.get(key) for r in rs]
            if any(v is not None for v in vals):
                row[key] = mean_ci(vals)
        for H, k in TEST_CONFIGS:
            key = f"H{H}k{k}"
            for metric in ("acc", "wrong", "tokens", "control_acc"):
                row[f"{key}.{metric}"] = mean_ci([r[key][metric] for r in rs])
        out[mode] = row
    return out


# ---------------------------------------------------------------- experiments
def headline(seeds, **over):
    modes = ["A", "B", "C"]
    by = run_many(seeds, modes, **over)
    return by, summarize(by)


def ablations(seeds, **over):
    modes = ["A", "C", "noise_only", "no_selector", "no_writes", "noise_selector"]
    by = run_many(seeds, modes, **over)
    return by, summarize(by)


def budget_sweep(seeds, budgets=(0, 300, 600, 1200, 2400, 4800), **over):
    rows = []
    for b in budgets:
        for mode in ("noise_only", "full"):
            by = run_many(seeds, [mode], idle_tokens_per_session=b, **over)
            s = summarize(by)[mode]
            rows.append({"budget": b, "mode": mode, "summary": s})
    return rows


def amplitude_sweep(seeds, amps=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0), **over):
    rows = []
    for a in amps:
        by = run_many(seeds, ["full"], amplitude=a, **over)
        rows.append({"amplitude": a, "summary": summarize(by)["full"]})
    return rows


def robustness_grid(seeds, hs=(0.0, 0.15, 0.3, 0.5), ms=(0.0, 0.15, 0.4), **over):
    rows = []
    for h in hs:
        for m in ms:
            by = run_many(seeds, ["noise_only", "full"], hallucination_rate=h, verifier_miss=m, **over)
            s = summarize(by)
            rows.append({"h": h, "m": m, "B": s["noise_only"], "C": s["full"]})
    return rows


def run_all(seeds, quick=False, out_dir="results", **over):
    os.makedirs(out_dir, exist_ok=True)
    res = {"seeds": list(seeds), "defaults": {**DEFAULTS, **over}}
    by, res["headline"] = headline(seeds, **over)
    by2, res["ablations"] = ablations(seeds, **over)
    res["budget_sweep"] = budget_sweep(seeds, budgets=(0, 600, 1200, 2400) if quick else (0, 300, 600, 1200, 2400, 4800), **over)
    res["amplitude_sweep"] = amplitude_sweep(seeds, amps=(0.0, 0.25, 0.5, 1.0) if quick else (0.0, 0.1, 0.25, 0.5, 0.75, 1.0), **over)
    res["robustness"] = robustness_grid(seeds, hs=(0.0, 0.3, 0.5) if quick else (0.0, 0.15, 0.3, 0.5), ms=(0.0, 0.4) if quick else (0.0, 0.15, 0.4), **over)
    res["focus"] = focus_experiment(seeds, amps=(0.0, 0.5, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0), **over)
    res["upgrades"] = upgrades_experiment(seeds, **over)
    # paired differences C - A and C - B on the headline configs
    res["paired"] = {}
    for label, cfg in (("strict", HEADLINE_STRICT), ("generous", HEADLINE_GENEROUS)):
        key = f"H{cfg[0]}k{cfg[1]}"
        for other, name in (("A", "A"), ("B", "B")):
            d = [c[key]["acc"] - o[key]["acc"] for c, o in zip(by["C"], by[other])]
            res["paired"][f"C-{name}@{label}"] = mean_ci(d)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    return res
