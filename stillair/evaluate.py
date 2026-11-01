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
import math
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from . import world as W
from .agent import Agent
from .llm import SimLLM
from .loop import build_loop
from .memory import Memory

# test-time budgets: (max retrieval hops, memories read per hop)
TEST_CONFIGS = [(1, 2), (1, 3), (1, 5), (2, 3), (2, 5)]
HEADLINE_STRICT = (1, 3)  # one retrieval round
HEADLINE_GENEROUS = (2, 3)  # two retrieval rounds

DEFAULTS = dict(
    n_chains=24,
    n_sessions=5,
    idle_tokens_per_session=2400,
    amplitude=0.5,
    hallucination_rate=0.3,
    verifier_miss=0.15,
    retrieval_noise=0.04,
)

_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 14: 2.14, 19: 2.09, 29: 2.05}


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return (m, 0.0)
    df = len(xs) - 1
    t = _T95.get(df) or _T95[max(k for k in _T95 if k <= df)] if df < 30 else 1.96
    return (m, t * statistics.stdev(xs) / math.sqrt(len(xs)))


def run_episode(seed: int, mode: str, **over) -> dict:
    p = {**DEFAULTS, **over}
    world = W.generate(seed, p["n_chains"], p["n_sessions"])
    mem = Memory()
    llm_ing = SimLLM(seed)
    loop = None
    if mode != "A":
        loop = build_loop(
            mem, seed, {"B": "noise_only", "C": "full"}.get(mode, mode),
            p["amplitude"], p["hallucination_rate"], p["verifier_miss"],
        )
    idle_tokens = 0
    for s, facts in enumerate(world.sessions):
        for f in facts:
            claim = llm_ing.extract(f.text)
            mem.add("fact", f.text, claim, "user", session=s, born=loop.step_no if loop else 0)
        if loop is not None and p["idle_tokens_per_session"] > 0:
            idle_tokens += loop.run(token_budget=p["idle_tokens_per_session"])

    out = {"seed": seed, "mode": mode, "idle_tokens": idle_tokens, "idle_steps": loop.step_no if loop else 0}

    # ---- stored-connection quality (against ground truth) ----
    conns = [n for n in mem.nodes.values() if n.kind == "connection"]
    seen, valid_unique, invalid, redundant = set(), 0, 0, 0
    for n in conns:
        c = n.claim
        if c is None:
            invalid += 1
        elif len(c[1]) == 2:
            if (c[0], c[1], c[2]) in world.valid_derived:
                if (c[0], c[1], c[2]) in seen:
                    redundant += 1
                else:
                    seen.add((c[0], c[1], c[2]))
                    valid_unique += 1
            else:
                invalid += 1
        else:
            redundant += 1  # restated base fact
    out.update(
        connections=len(conns),
        valid_unique=valid_unique,
        invalid=invalid,
        redundant=redundant,
        precision=(valid_unique / len(conns)) if conns else None,
        coverage=valid_unique / max(1, world.n_chains),
        memory_size=len(mem.nodes),
    )

    # ---- idle-loop drift statistics (from the thought log) ----
    if loop and loop.events:
        ev = loop.events
        n = len(ev)
        rep = sum(1 for e in ev if e["kind"] == "restate" or e.get("dup"))
        bad = sum(
            1
            for e in ev
            if e["kind"] == "analogy"
            or (e["kind"] == "derive" and tuple([e["claim"][0], tuple(e["claim"][1]), e["claim"][2]]) not in world.valid_derived)
        )
        out.update(
            repetition_rate=rep / n,
            incoherence_rate=bad / n,
            drift_failure_rate=sum(
                1
                for e in ev
                if e["kind"] in ("restate", "analogy", "none")
                or e.get("dup")
                or (e["kind"] == "derive" and tuple([e["claim"][0], tuple(e["claim"][1]), e["claim"][2]]) not in world.valid_derived)
            )
            / n,
            accepted_per_1k_tokens=1000 * sum(1 for e in ev if e["wrote"]) / max(1, loop.tokens),
        )

    # ---- test time ----
    answerer = SimLLM(seed + 999)
    for H, k in TEST_CONFIGS:
        rng = random.Random(seed * 1000 + H * 10 + k)
        ag = Agent(mem, answerer, k=k, max_hops=H, retrieval_noise=p["retrieval_noise"], rng=rng)
        c_ok = c_wrong = c_tok = 0
        ctl_ok = 0
        for q in world.chain_questions():
            tr = ag.ask(q.text)
            c_ok += tr.answer == q.answer
            c_wrong += tr.answer is not None and tr.answer != q.answer
            c_tok += tr.tokens
        for q in world.control_questions():
            ctl_ok += ag.ask(q.text).answer == q.answer
        nc, nk = len(world.chain_questions()), max(1, len(world.control_questions()))
        key = f"H{H}k{k}"
        out[key] = dict(acc=c_ok / nc, wrong=c_wrong / nc, tokens=c_tok / nc, control_acc=ctl_ok / nk)
    return out


def run_focus_episode(seed: int, amplitude: float, use_state: bool = True, extra_tokens: int = 1200, base_tokens: int = 300, **over) -> dict:
    """Does the internal drive (open questions) beat blind noise at *targeted* consolidation?

    Half of the chain questions are asked at test time with a one-hop budget; the ones that
    fail are stored as open questions. The loop then gets `extra_tokens` more idle budget.
    We measure recovery on the asked half and collateral gain on the never-asked half.
    """
    p = {**DEFAULTS, **over}
    world = W.generate(seed, p["n_chains"], p["n_sessions"])
    mem = Memory()
    ing = SimLLM(seed)
    mode = "full" if use_state else "noise_selector"
    loop = build_loop(mem, seed, mode, amplitude, p["hallucination_rate"], p["verifier_miss"])
    for s, facts in enumerate(world.sessions):
        for f in facts:
            mem.add("fact", f.text, ing.extract(f.text), "user", session=s)
        loop.run(token_budget=base_tokens)
    chains = world.chain_questions()
    rng = random.Random(seed + 5)
    asked = set(rng.sample(range(len(chains)), len(chains) // 2))
    ans = SimLLM(seed + 999)

    def agent(readonly):
        return Agent(mem, ans, k=3, max_hops=1, retrieval_noise=p["retrieval_noise"], rng=random.Random(seed), readonly=readonly)

    before = {i: (agent(i not in asked).ask(q.text).answer == q.answer) for i, q in enumerate(chains)}
    loop.run(token_budget=extra_tokens)
    after = {i: (agent(True).ask(q.text).answer == q.answer) for i, q in enumerate(chains)}
    a, u = sorted(asked), sorted(set(range(len(chains))) - asked)
    return dict(
        seed=seed,
        asked_before=sum(before[i] for i in a) / len(a), asked_after=sum(after[i] for i in a) / len(a),
        unasked_before=sum(before[i] for i in u) / len(u), unasked_after=sum(after[i] for i in u) / len(u),
    )


def focus_experiment(seeds, amps=(0.0, 0.25, 0.5, 0.75, 1.0), **over):
    rows = []
    cfgs = [(f"drive a={a}", a, True) for a in amps] + [("blind noise (no state)", 1.0, False)]
    for label, a, st in cfgs:
        rs = [run_focus_episode(s, a, st, **over) for s in seeds]
        row = {"label": label, "amplitude": a, "use_state": st}
        for k in ("asked_before", "asked_after", "unasked_before", "unasked_after"):
            row[k] = mean_ci([r[k] for r in rs])
        row["asked_gain"] = mean_ci([r["asked_after"] - r["asked_before"] for r in rs])
        row["unasked_gain"] = mean_ci([r["unasked_after"] - r["unasked_before"] for r in rs])
        rows.append(row)
    return rows


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
                    "drift_failure_rate", "accepted_per_1k_tokens"):
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
    # paired differences C - A and C - B on the headline configs
    res["paired"] = {}
    for label, cfg in (("strict", HEADLINE_STRICT), ("generous", HEADLINE_GENEROUS)):
        key = f"H{cfg[0]}k{cfg[1]}"
        for other, name in (("A", "A"), ("B", "B")):
            d = [c[key]["acc"] - o[key]["acc"] for c, o in zip(by["C"], by[other])]
            res["paired"][f"C-{name}@{label}"] = mean_ci(d)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    return res


def fmt(x, pct=False, digits=2):
    if x is None or x[0] is None:
        return "n/a"
    m, ci = x
    if pct:
        return f"{100*m:.0f}% ±{100*ci:.0f}"
    return f"{m:.{digits}f} ±{ci:.{digits}f}"
