"""One episode of the experiment: teach a world, idle, score memory, answer questions.

Ground truth lives in the world, never in the agent: stored connections are scored
afterwards against `world.valid_derived`.
"""

from __future__ import annotations

import random

from ..backends.sim import SimLLM
from ..core.agent import Agent
from ..core.factory import build_loop
from ..core.memory import Memory, tokens
from ..world import generator as W
from .config import DEFAULTS, TEST_CONFIGS, UPGRADES_OFF

CONDITION_MODES = {"B": "noise_only", "C": "full"}


def answer_matches(given, truth: str) -> bool:
    """Exact for the simulator; a real model may reword, so most of the truth's content words must appear."""
    if given is None:
        return False
    if given == truth:
        return True
    want = set(tokens(truth))
    return bool(want) and len(want & set(tokens(given))) / len(want) >= 0.75


def _params(over: dict) -> dict:
    return {**DEFAULTS, **UPGRADES_OFF, **over}


def _world(seed: int, p: dict):
    # facts are parsed from the templates even with a real model, so its wording cannot break ground-truth scoring
    return W.generate(seed, **p["world"]) if p.get("world") else W.generate(seed, p["n_chains"], p["n_sessions"])


def _loop_flags(p: dict) -> dict:
    return dict(skip_unlinked=p["skip_unlinked"], trial_expiry=p["trial_expiry"], adaptive_noise=p["adaptive_noise"], llm=p.get("llm"))


def _agent_flags(p: dict) -> dict:
    return dict(trial_links=p["trial_links"], verify_on_use=p["verify_on_use"])


def _connection_quality(mem: Memory, world) -> dict:
    """Score stored connections against ground truth."""
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
    return dict(
        connections=len(conns),
        valid_unique=valid_unique,
        invalid=invalid,
        redundant=redundant,
        precision=(valid_unique / len(conns)) if conns else None,
        coverage=valid_unique / max(1, world.n_chains),
        memory_size=len(mem.nodes),
    )


def _drift_stats(loop, world) -> dict:
    """Repetition and incoherence rates from the thought log."""
    ev = loop.events
    n = len(ev)

    def false_derivation(e):
        if e["kind"] != "derive":
            return False
        return e["claim"] is None or (e["claim"][0], tuple(e["claim"][1]), e["claim"][2]) not in world.valid_derived

    return dict(
        repetition_rate=sum(1 for e in ev if e["kind"] == "restate" or e.get("dup")) / n,
        incoherence_rate=sum(1 for e in ev if e["kind"] == "analogy" or false_derivation(e)) / n,
        drift_failure_rate=sum(1 for e in ev if e["kind"] in ("restate", "analogy", "none") or e.get("dup") or false_derivation(e)) / n,
        accepted_per_1k_tokens=1000 * sum(1 for e in ev if e["wrote"]) / max(1, loop.tokens),
    )


def _test_time_scores(mem: Memory, world, seed: int, p: dict) -> dict:
    """Answer every question under each test-time budget."""
    out = {}
    answerer = p.get("llm") or SimLLM(seed + 999, p["hallucination_rate"], p["verifier_miss"])
    for H, k in p.get("test_configs") or TEST_CONFIGS:
        rng = random.Random(seed * 1000 + H * 10 + k)
        ag = Agent(mem, answerer, k=k, max_hops=H, retrieval_noise=p["retrieval_noise"], rng=rng, **_agent_flags(p))
        c_ok = c_wrong = c_tok = ctl_ok = 0
        for q in world.chain_questions():
            tr = ag.ask(q.text)
            ok = answer_matches(tr.answer, q.answer)
            c_ok += ok
            c_wrong += tr.answer is not None and not ok
            c_tok += tr.tokens
        for q in world.control_questions():
            ctl_ok += answer_matches(ag.ask(q.text).answer, q.answer)
        nc, nk = len(world.chain_questions()), max(1, len(world.control_questions()))
        out[f"H{H}k{k}"] = dict(acc=c_ok / nc, wrong=c_wrong / nc, tokens=c_tok / nc, control_acc=ctl_ok / nk)
    return out


def run_episode(seed: int, mode: str, **over) -> dict:
    p = _params(over)
    world = _world(seed, p)
    mem = Memory()
    llm_ing = SimLLM(seed)
    loop = None
    if mode != "A":
        loop = build_loop(mem, seed, CONDITION_MODES.get(mode, mode), p["amplitude"], p["hallucination_rate"], p["verifier_miss"], **_loop_flags(p))
    idle_tokens = 0
    for s, facts in enumerate(world.sessions):
        for f in facts:
            mem.add("fact", f.text, llm_ing.extract(f.text), "user", session=s, born=loop.step_no if loop else 0)
        if loop is not None and p.get("idle_steps_per_session"):
            idle_tokens += loop.run(steps=p["idle_steps_per_session"])
        elif loop is not None and p["idle_tokens_per_session"] > 0:
            idle_tokens += loop.run(token_budget=p["idle_tokens_per_session"])

    out = {"seed": seed, "mode": mode, "idle_tokens": idle_tokens, "idle_steps": loop.step_no if loop else 0}
    out.update(_connection_quality(mem, world))
    if loop and loop.events:
        out.update(_drift_stats(loop, world))
    out["pruned"] = mem.self_model["pruned"]
    out.update(_test_time_scores(mem, world, seed, p))
    return out


def run_focus_episode(seed: int, amplitude: float, use_state: bool = True, extra_tokens: int = 1200, base_tokens: int = 300, **over) -> dict:
    """Does the internal drive (open questions) beat blind noise at *targeted* consolidation?

    Half of the chain questions are asked at test time with a one-hop budget; the ones that
    fail are stored as open questions. The loop then gets `extra_tokens` more idle budget.
    We measure recovery on the asked half and collateral gain on the never-asked half.
    """
    p = _params(over)
    world = _world(seed, p)
    mem = Memory()
    ing = SimLLM(seed)
    mode = "full" if use_state else "noise_selector"
    loop = build_loop(mem, seed, mode, amplitude, p["hallucination_rate"], p["verifier_miss"], **_loop_flags(p))
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
