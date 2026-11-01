"""Command line: `python3 -m stillair <demo|serve|eval|ask>`."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from . import evaluate as E
from . import world as W
from .agent import Agent
from .llm import SimLLM
from .loop import build_loop
from .memory import Memory
from .report import to_markdown

_TTY = sys.stdout.isatty()


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


GREEN, RED, YEL, DIM, BOLD, CYAN = (lambda s: c(32, s)), (lambda s: c(31, s)), (lambda s: c(33, s)), (lambda s: c(2, s)), (lambda s: c(1, s)), (lambda s: c(36, s))
BADGE = {"keep": GREEN("KEEP     "), "reinforce": YEL("REINFORCE"), "discard": RED("DISCARD  ")}


def cmd_eval(a):
    seeds = list(range(a.seeds))
    t = time.time()
    res = E.run_all(seeds, quick=a.quick, out_dir=a.out)
    md = to_markdown(res)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"\n[done in {time.time() - t:.0f}s] wrote {a.out}/results.json and {a.out}/results.md")


def cmd_report(a):
    with open(os.path.join(a.out, "results.json")) as f:
        res = json.load(f)
    md = to_markdown(res)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(md)
    print(md)


def _pause(a, s=0.0):
    if not a.fast:
        time.sleep(s)


def cmd_demo(a):
    seed = a.seed
    world = W.generate(seed, n_chains=6, n_sessions=3, n_dangling_rel=3, n_dangling_prop=3, n_controls=2)
    mem = Memory()
    real = a.backend == "claude"
    if real:
        from .claude_backend import ClaudeLLM

        llm = ing = ans = ClaudeLLM()
        print(DIM(f"backend: Claude ({llm.model}); the scoreboard against ground truth is skipped"))
    else:
        llm, ing, ans = None, SimLLM(seed), SimLLM(seed + 999)
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


def cmd_serve(a):
    from .live import serve

    serve(a.port, a.seed)


def main(argv=None):
    p = argparse.ArgumentParser(prog="stillair", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="narrated terminal demo")
    d.add_argument("--seed", type=int, default=0, help="0 = success run, 3 = a run where a confabulated link slips through")
    d.add_argument("--steps", type=int, default=30)
    d.add_argument("--fast", action="store_true", help="no pauses")
    d.add_argument("--backend", choices=("sim", "claude"), default="sim", help="sim = offline simulated associator (default)")
    d.add_argument("--no-selector", action="store_true", help="show what happens without the selector (condition B)")
    d.add_argument("--hallucination", type=float, default=0.3)
    d.add_argument("--verifier-miss", type=float, default=0.15, help="how often the grounding judge is fooled")
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve", help="live web dashboard")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_serve)
    e = sub.add_parser("eval", help="run the three-condition experiment and write results/")
    e.add_argument("--seeds", type=int, default=20)
    e.add_argument("--quick", action="store_true")
    e.add_argument("--out", default="results")
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("report", help="rebuild results/results.md from results/results.json")
    r.add_argument("--out", default="results")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
