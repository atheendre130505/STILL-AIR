"""Live demo state: the idle loop running in a background thread.

Task mode vs idle mode: `ask()` takes the session lock, so a user prompt pauses idle thinking
for exactly as long as it takes to answer; afterwards the loop resumes, seeded with the
question (its drive is the open question if the answer was not known).
"""

from __future__ import annotations

import threading
import time

from ..backends.sim import SimLLM
from ..core.agent import Agent
from ..core.factory import build_loop
from ..core.memory import Memory
from ..evaluation.episode import answer_matches
from ..world import generator as W
from ..world.schemas import parse_question


class BadRequest(ValueError):
    """Client input the session refuses; the server reports it as HTTP 400."""


def _number(cfg: dict, key: str, low: float, high: float) -> float:
    value = cfg[key]
    # the comparison also rejects NaN, which json.loads accepts
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise BadRequest(f"{key} must be a number from {low} to {high}")
    return float(value)


def _flag(cfg: dict, key: str) -> bool:
    if not isinstance(cfg[key], bool):
        raise BadRequest(f"{key} must be true or false")
    return cfg[key]


class _ModelOutsideLock:
    """Wraps a real model so the session lock is released during its slow calls, keeping the dashboard live."""

    def __init__(self, llm, lock):
        self._llm, self._lock = llm, lock

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def _unlocked(self, call, *args):
        self._lock.release()
        try:
            return call(*args)
        finally:
            self._lock.acquire()

    def associate(self, seeds, nudge=None):
        return self._unlocked(self._llm.associate, seeds, nudge)

    def judge(self, thought, parents):
        return self._unlocked(self._llm.judge, thought, parents)

    def answer(self, question, context, scratch):
        return self._unlocked(self._llm.answer, question, context, scratch)


class LiveSession:
    def __init__(self, seed: int = 0, llm=None, token_budget: int = 50_000):
        self.lock = threading.RLock()  # shared with every loop, so swapping state on reset is atomic
        self.real = llm  # None = the simulator
        self.model = _ModelOutsideLock(llm, self.lock) if llm is not None else None
        self.token_budget = token_budget  # stops the loop before it can drain a free daily quota
        self.error = None
        self.reset(seed)

    def reset(self, seed=None):
        with self.lock:
            if hasattr(self, "_stop"):
                self._stop.set()
            self.seed = self.seed if seed is None else seed
            self.world = W.generate(self.seed, n_chains=10, n_sessions=4, n_dangling_rel=5, n_dangling_prop=5, n_controls=4)
            self.mem = Memory()
            self.ing = SimLLM(self.seed)
            # with a real model, spend calls only on seeds that could connect
            self.loop = build_loop(self.mem, self.seed, "full", amplitude=0.5, llm=self.model, skip_unlinked=self.real is not None)
            self.loop.lock = self.lock
            self.answerer = self.model or SimLLM(self.seed + 999)
            self.fed = 0
            self.running = False
            self.rate = 4.0
            self.mode = "idle"
            self.last_answer = None
            self.error = None
            self._in_task = False
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._stop, self.loop, self.mem), daemon=True)
            self._thread.start()

    # ---- background idle thread -----------------------------------------
    def _budget_spent(self) -> bool:
        return self.real is not None and self.real.tokens_used >= self.token_budget

    def _stop_with(self, message: str) -> None:
        self.running = False
        self.error = message

    def _run(self, stop, loop, mem):
        while not stop.is_set():
            if not (self.running and not self._in_task and mem.readable()):
                time.sleep(0.1)
                continue
            if self._budget_spent():
                self._stop_with(f"Idle loop stopped: the {self.token_budget:,} token budget for this run is used up.")
                continue
            try:
                loop.step()
            except Exception as e:
                self._stop_with(f"Idle loop stopped, the model call failed: {e}")
                continue
            time.sleep(1.0 / max(0.5, self.rate))

    # ---- actions ----------------------------------------------------------
    def feed(self):
        with self.lock:
            if self.fed >= len(self.world.sessions):
                return {"fed": self.fed, "added": []}
            added = []
            for f in self.world.sessions[self.fed]:
                self.mem.add("fact", f.text, self.ing.extract(f.text), "user", session=self.fed, born=self.loop.step_no)
                added.append(f.text)
            self.fed += 1
            return {"fed": self.fed, "added": added}

    def ask(self, question):
        if not isinstance(question, str) or not question.strip():
            raise BadRequest("question must be a non-empty string")
        question = question.strip()
        truth = next((q.answer for q in self.world.questions if q.text == question), None)
        # The simulator only reads the world's templated questions. Anything else could never
        # be answered, so it must not become an open question the loop chases forever.
        understood = self.real is not None or parse_question(question) is not None

        def result(trace):
            correct = None if truth is None else answer_matches(trace.answer, truth)
            return {"answer": trace.answer, "tokens": trace.tokens, "hops": trace.hops, "used": trace.via, "correct": correct}

        with self.lock:  # idle thinking pauses while the task runs
            self.mode = "task"
            self._in_task = True
            try:
                base = Agent(self.mem, self.answerer, k=3, max_hops=1, kinds=("fact",)).ask(question)
                tr = Agent(self.mem, self.answerer, k=3, max_hops=1, readonly=not understood).ask(question)
                if understood:
                    self.loop.perturb.seed(self.mem.vec(question))  # resume idle thinking seeded by the task
                self.last_answer = {
                    "question": question,
                    "truth": truth,
                    "understood": understood,
                    "baseline": result(base),
                    "still_air": result(tr),
                }
            except Exception as e:
                self.error = f"The model call failed: {e}"
                raise BadRequest(self.error) from None
            finally:
                self.mode = "idle"
                self._in_task = False
        return self.last_answer

    def config(self, c: dict):
        # validate everything first, so a bad field cannot leave a half-applied update
        updates = {}
        for key, low, high in (("amplitude", 0.0, 1.0), ("hallucination", 0.0, 1.0), ("verifier_miss", 0.0, 1.0), ("rate", 0.5, 100.0)):
            if key in c:
                updates[key] = _number(c, key, low, high)
        for key in ("selector", "writes", "running"):
            if key in c:
                updates[key] = _flag(c, key)
        with self.lock:
            if "amplitude" in updates:
                self.loop.perturb.amplitude = updates["amplitude"]
            if "selector" in updates:
                self.loop.selector.enabled = updates["selector"]
            if "writes" in updates:
                self.loop.cfg.write_memory = updates["writes"]
            if "hallucination" in updates and self.real is None:
                self.loop.llm.h = updates["hallucination"]
            if "verifier_miss" in updates and self.real is None:
                self.loop.llm.m = updates["verifier_miss"]
            if "rate" in updates:
                self.rate = updates["rate"]
            if "running" in updates:
                self.running = updates["running"]
                if self.running:
                    self.error = None
        return {"ok": True}

    # ---- state for the dashboard -----------------------------------------
    def snapshot(self):
        with self.lock:
            nodes = []
            valid_unique, seen, n_conn, false_conn = 0, set(), 0, 0
            for n in self.mem.nodes.values():
                v = None
                if n.kind == "connection":
                    n_conn += 1
                    c = n.claim
                    ok = bool(c and len(c[1]) == 2 and (c[0], tuple(c[1]), c[2]) in self.world.valid_derived)
                    v = ok
                    if ok and (c[0], tuple(c[1]), c[2]) not in seen:
                        seen.add((c[0], tuple(c[1]), c[2]))
                        valid_unique += 1
                    if not ok and not (c and len(c[1]) == 1):
                        false_conn += 1
                nodes.append({"id": n.id, "kind": n.kind, "text": n.text, "session": n.session, "source": n.source,
                              "strength": round(n.strength, 2), "parents": n.parents, "valid": v,
                              "status": n.meta.get("status")})
            sm = self.mem.self_model
            total_chains = self.world.n_chains
            return {
                "running": self.running, "mode": self.mode, "rate": self.rate,
                "fed": self.fed, "sessions": len(self.world.sessions),
                "step": self.loop.step_no, "tokens": self.loop.tokens,
                "nodes": nodes,
                "events": self.loop.events[-60:],
                "stats": {"kept": sm["kept"], "reinforced": sm["reinforced"], "discarded": sm["discarded"], "pruned": sm["pruned"],
                          "curiosity": sm["curiosity"], "uncertainty": sm["uncertainty"],
                          "connections": n_conn, "valid_unique": valid_unique, "false": false_conn,
                          "precision": (valid_unique / n_conn) if n_conn else None,
                          "coverage": valid_unique / total_chains},
                "config": {"amplitude": self.loop.perturb.amplitude, "selector": self.loop.selector.enabled,
                           "writes": self.loop.cfg.write_memory, "hallucination": getattr(self.loop.llm, "h", None),
                           "verifier_miss": getattr(self.loop.llm, "m", None)},
                "backend": self.real.name if self.real is not None else "sim",
                "model": getattr(self.real, "model", None),
                "tokens_used": getattr(self.real, "tokens_used", 0),
                "token_budget": self.token_budget,
                "error": self.error,
                "suggested": [{"text": q.text, "kind": q.kind} for q in self.world.chain_questions()],
                "last_answer": self.last_answer,
                "seed": self.seed,
            }
