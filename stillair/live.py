"""Live demo: the idle loop running in a background thread, plus a tiny HTTP server for the dashboard.

Task mode vs idle mode: `ask()` takes the loop's lock, so a user prompt pauses idle thinking
for exactly as long as it takes to answer; afterwards the loop resumes, seeded with the
question (its drive is the open question if the answer was not known).
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import world as W
from .agent import Agent
from .llm import SimLLM
from .loop import build_loop
from .memory import KINDS_READABLE, Memory

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class LiveSession:
    def __init__(self, seed: int = 0):
        self.reset(seed)

    def reset(self, seed=None):
        if hasattr(self, "_stop"):
            self._stop.set()
        self.seed = self.seed if seed is None else seed
        self.world = W.generate(self.seed, n_chains=10, n_sessions=4, n_dangling_rel=5, n_dangling_prop=5, n_controls=4)
        self.mem = Memory()
        self.ing = SimLLM(self.seed)
        self.loop = build_loop(self.mem, self.seed, "full", amplitude=0.5)
        self.answerer = SimLLM(self.seed + 999)
        self.fed = 0
        self.running = False
        self.rate = 4.0
        self.mode = "idle"
        self.last_answer = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True)
        self._thread.start()

    # ---- background idle thread -----------------------------------------
    def _run(self, stop):
        while not stop.is_set():
            if self.running and self.mem.readable():
                self.loop.step()
                time.sleep(1.0 / max(0.5, self.rate))
            else:
                time.sleep(0.1)

    # ---- actions ----------------------------------------------------------
    def feed(self):
        with self.loop.lock:
            if self.fed >= len(self.world.sessions):
                return {"fed": self.fed, "added": []}
            added = []
            for f in self.world.sessions[self.fed]:
                self.mem.add("fact", f.text, self.ing.extract(f.text), "user", session=self.fed, born=self.loop.step_no)
                added.append(f.text)
            self.fed += 1
            return {"fed": self.fed, "added": added}

    def ask(self, question: str):
        truth = next((q.answer for q in self.world.questions if q.text == question), None)
        with self.loop.lock:  # idle thinking pauses while the task runs
            self.mode = "task"
            try:
                base = Agent(self.mem, self.answerer, k=3, max_hops=1, kinds=("fact",)).ask(question)
                tr = Agent(self.mem, self.answerer, k=3, max_hops=1, readonly=False).ask(question)
                if tr.answer is None:
                    self.loop.perturb.seed(self.mem.vec(question))  # resume idle thinking seeded by the task
                self.last_answer = {
                    "question": question,
                    "truth": truth,
                    "baseline": {"answer": base.answer, "tokens": base.tokens, "hops": base.hops, "used": base.via},
                    "still_air": {"answer": tr.answer, "tokens": tr.tokens, "hops": tr.hops, "used": tr.via},
                }
            finally:
                self.mode = "idle"
        return self.last_answer

    def config(self, c: dict):
        with self.loop.lock:
            if "amplitude" in c:
                self.loop.perturb.amplitude = float(c["amplitude"])
            if "selector" in c:
                self.loop.selector.enabled = bool(c["selector"])
            if "writes" in c:
                self.loop.cfg.write_memory = bool(c["writes"])
            if "hallucination" in c:
                self.loop.llm.h = float(c["hallucination"])
            if "verifier_miss" in c:
                self.loop.llm.m = float(c["verifier_miss"])
            if "rate" in c:
                self.rate = float(c["rate"])
            if "running" in c:
                self.running = bool(c["running"])
        return {"ok": True}

    # ---- state for the dashboard -----------------------------------------
    def snapshot(self):
        with self.loop.lock:
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
                           "writes": self.loop.cfg.write_memory, "hallucination": self.loop.llm.h,
                           "verifier_miss": self.loop.llm.m},
                "suggested": [{"text": q.text, "kind": q.kind} for q in self.world.chain_questions()],
                "last_answer": self.last_answer,
                "seed": self.seed,
            }


def serve(port: int = 8765, seed: int = 0):
    live = LiveSession(seed)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                return self._json(live.snapshot())
            if self.path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            route = self.path.rstrip("/").split("/")[-1]
            if route == "feed":
                return self._json(live.feed())
            if route == "ask":
                return self._json(live.ask(str(data.get("question", ""))))
            if route == "config":
                return self._json(live.config(data))
            if route == "reset":
                live.reset(int(data.get("seed", live.seed)))
                return self._json({"ok": True})
            self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"Still Air dashboard: http://127.0.0.1:{port}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
