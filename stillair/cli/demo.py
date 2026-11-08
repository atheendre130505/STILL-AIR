"""The narrated terminal demo (`python -m stillair demo`)."""

from __future__ import annotations

import sys
import time

from ..backends.registry import make_backend
from ..backends.sim import SimLLM
from ..core.agent import Agent
from ..core.factory import build_loop
from ..core.memory import Memory
from ..world import generator as W

_TTY = sys.stdout.isatty()


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


GREEN, RED, YEL, DIM, BOLD, CYAN = (lambda s: c(32, s)), (lambda s: c(31, s)), (lambda s: c(33, s)), (lambda s: c(2, s)), (lambda s: c(1, s)), (lambda s: c(36, s))
BADGE = {"keep": GREEN("KEEP     "), "reinforce": YEL("REINFORCE"), "discard": RED("DISCARD  ")}


def _pause(a, s=0.0):
    if not a.fast:
        time.sleep(s)


def cmd_demo(a):
    seed = a.seed
    world = W.generate(seed, n_chains=6, n_sessions=3, n_dangling_rel=3, n_dangling_prop=3, n_controls=2)
    mem = Memory()
    llm = make_backend(a.backend)
    real = llm is not None
    ing = SimLLM(seed)  # facts are parsed from the templates either way, so entity names stay exact
    ans = llm or SimLLM(seed + 999)
    if real:
        print(DIM(f"backend: {llm.name} ({llm.model}); the scoreboard against ground truth is skipped"))
    loop = build_loop(mem, seed, "noise_only" if a.no_selector else "full", amplitude=0.5,
                      hallucination_rate=a.hallucination, verifier_miss=a.verifier_miss, llm=llm)
    ask = lambda q, rw=False: Agent(mem, ans, k=3, max_hops=1, readonly=not rw).ask(q)

    print(BOLD("\nSTILL AIR  -  an idle loop for self-initiated AI thought"))
    print(DIM("simulated associator, thought-only, every thought logged\n"))

    # pick a chain whose two facts arrive in different sessions
    target = world.chain_questions()[0]
    print(BOLD("1. Learning (two facts, two different sessions)"))
    for s, facts in enumerate(world.sessions):
        for f in facts:
            mem.add("fact", f.text, ing.extract(f.text), "user", session=s)
            if f.chain == target.chain and f.role.startswith("chain"):
                print(f"   session {s + 1}: {CYAN(f.text)}")
        _pause(a, 0.3)
    print(DIM(f"   (memory now holds {mem.counts()['fact']} facts; nobody has ever stated the combined fact)\n"))

    print(BOLD("2. A question arrives, with a strict budget (one retrieval round)"))
    print(f"   Q: {target.text}")
    tr = ask(target.text, rw=True)
    print(f"   -> {RED('I do not know') if tr.answer is None else tr.answer}   ({tr.hops} hop, {tr.tokens} tokens)")
    print(DIM("   The question is stored as an OPEN QUESTION: a drive for the idle loop.\n"))
    _pause(a, 0.8)

    print(BOLD(f"3. Idle time: no prompt. The loop perturbs, associates, evaluates, consolidates ({a.steps} steps)"))
    shown = 0
    for _ in range(a.steps):
        ev = loop.step()
        shown += 1
        src = {"open_question": "open question", "drift": "own drift", "noise": "random noise"}[ev["source"]]
        txt = (ev["text"] or "(nothing)")[:92]
        print(f"   {ev['step']:>3} {DIM(src.ljust(13))} {BADGE[ev['action']]} v={ev['validity']:.2f} n={ev['novelty']:.2f}  {txt}")
        _pause(a, 0.12)
    kept = [n for n in mem.nodes.values() if n.kind == "connection"]
    print(DIM(f"\n   {len(kept)} connections in memory, {mem.self_model['discarded']} thoughts discarded, "
              f"{mem.self_model['reinforced']} reinforced, {loop.tokens} idle tokens spent\n"))
    _pause(a, 0.8)

    print(BOLD("4. The same question again, same strict budget"))
    print(f"   Q: {target.text}")
    tr = ask(target.text)
    good = tr.answer is not None and (tr.answer.lower() == target.answer.lower() or target.answer.lower() in tr.answer.lower())
    print(f"   -> {GREEN(tr.answer) if good else RED(str(tr.answer))}   ({tr.hops} hop, {tr.tokens} tokens)  "
          f"{'[correct]' if good else '[still wrong or unknown]'}")
    if tr.via:
        n = mem.nodes[tr.via[0]]
        print(DIM(f"   answered from an {n.source} {n.kind}: \"{n.text}\""))

    if real:
        print()
        return
    # whole-world scoreboard against hidden ground truth
    ok = sum(ask(q.text).answer == q.answer for q in world.chain_questions())
    wrong = [n for n in kept if not n.claim or len(n.claim[1]) != 2 or (n.claim[0], n.claim[1], n.claim[2]) not in world.valid_derived]
    print(BOLD("\n5. Scoreboard against hidden ground truth"))
    print(f"   two-hop questions answered in one hop: {ok}/{len(world.chain_questions())}")
    print(f"   stored connections that are NOT true:  {len(wrong)}/{len(kept)}"
          + (DIM("   (selector off: nothing filters confabulation)") if a.no_selector else ""))
    if wrong:
        print(DIM(f'   e.g. "{wrong[0].text[:100]}"'))
    print()
