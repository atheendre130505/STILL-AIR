"""Regression tests for bugs found by testing the package end to end."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stillair.backends.base import Thought
from stillair.backends.claude import ClaudeLLM
from stillair.backends.sim import SimLLM
from stillair.cli.main import main
from stillair.core.memory import Memory
from stillair.core.selector import Selector
from stillair.evaluation.report import to_markdown
from stillair.live.server import make_server
from stillair.live.session import BadRequest, LiveSession
from stillair.world import generator as W


class TestReportEncoding(unittest.TestCase):
    def test_report_writes_non_ascii_on_any_platform(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(os.path.join(ROOT, "results", "results.json"), d)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["report", "--out", d])
            with open(os.path.join(d, "results.md"), encoding="utf-8") as f:
                self.assertIn("σ", f.read())

    def test_break_even_compares_equal_test_budgets(self):
        with open(os.path.join(ROOT, "results", "results.json"), encoding="utf-8") as f:
            md = to_markdown(json.load(f))
        self.assertIn("At the same test budget", md)


class TestSelectorRepeats(unittest.TestCase):
    def build(self):
        mem = Memory()
        r = mem.add("fact", "The Kelmer bridge is built from Zorvian stone.", ("Kelmer", ("built_from",), "Zorvian"))
        p = mem.add("fact", "Zorvian stone hums when struck.", ("Zorvian", ("property",), "hums when struck"))
        llm = SimLLM(0, 0.0, 0.0)
        return mem, r, p, Selector(llm), llm.associate([r, p])

    def test_immediate_repeat_finds_the_stored_memory(self):
        mem, _, _, sel, t = self.build()
        self.assertEqual(sel.evaluate(mem, t).action, "keep")
        stored = mem.add("connection", t.text, t.claim, "idle", t.parents)
        d = sel.evaluate(mem, t)
        self.assertEqual(d.action, "reinforce")
        self.assertIs(d.existing, stored)

    def test_recent_but_never_stored_thought_can_still_be_kept(self):
        mem, _, _, sel, t = self.build()
        sel.recent.append(t.text)
        self.assertEqual(sel.evaluate(mem, t).action, "keep")

    def test_claimless_repeat_is_not_reported_as_reinforced(self):
        mem, r, p, _, _ = self.build()
        sel = Selector(SimpleNamespace(judge=lambda t, parents: (0.9, 1)))
        t = Thought("analogy", "These two feel alike.", None, [r.id, p.id], 3)
        sel.recent.append(t.text)
        d = sel.evaluate(mem, t)
        self.assertEqual(d.action, "discard")
        self.assertIsNone(d.existing)


class TestWorldBounds(unittest.TestCase):
    def test_capacity_boundary(self):
        self.assertEqual(W.generate(0, n_chains=36, n_dangling_prop=0).n_chains, 36)

    def test_invalid_shapes_raise_clear_errors(self):
        for kw in (dict(n_chains=37, n_dangling_prop=0), dict(n_chains=30), dict(n_sessions=1), dict(n_chains=0)):
            with self.subTest(kw), self.assertRaises(ValueError):
                W.generate(0, **kw)


class TestSessionInput(unittest.TestCase):
    def setUp(self):
        self.s = LiveSession(0)
        self.addCleanup(lambda: self.s._stop.set())
        for _ in range(4):
            self.s.feed()

    def test_bad_config_is_rejected_and_applies_nothing(self):
        before = self.s.snapshot()["config"]
        for cfg in ({"amplitude": "abc"}, {"amplitude": 5}, {"rate": 0}, {"running": "false"}, {"amplitude": float("nan")},
                    {"selector": False, "hallucination": 2}):
            with self.subTest(cfg), self.assertRaises(BadRequest):
                self.s.config(cfg)
        self.assertEqual(self.s.snapshot()["config"], before)

    def test_blank_question_is_rejected(self):
        for q in (None, "", "   ", 7):
            with self.subTest(q), self.assertRaises(BadRequest):
                self.s.ask(q)
        self.assertEqual(self.s.mem.open_questions(), [])

    def test_free_text_is_answered_but_not_stored(self):
        a = self.s.ask("What is the capital of France?")
        self.assertFalse(a["understood"])
        self.assertEqual(self.s.mem.open_questions(), [])

    def test_the_loop_is_seeded_with_the_question(self):
        q = self.s.world.chain_questions()[0].text
        self.s.loop.perturb.focus = {}
        self.assertTrue(self.s.ask(q)["understood"])
        self.assertTrue(self.s.loop.perturb.focus)

    def test_reset_shares_the_session_lock(self):
        self.s.reset(2)
        self.assertIs(self.s.loop.lock, self.s.lock)


class TestServer(unittest.TestCase):
    def setUp(self):
        self.srv, self.live = make_server(0, 0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.addCleanup(lambda: self.live._stop.set())
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def post(self, path, raw):
        req = urllib.request.Request(self.base + path, data=raw, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read()

    def test_bad_input_gets_a_400_not_a_dropped_connection(self):
        for path, raw in (("/api/config", b'{"amplitude":"abc"}'), ("/api/reset", b'{"seed":"x"}'), ("/api/ask", b"{}"),
                          ("/api/ask", b"[1,2]"), ("/api/ask", b"{nope"), ("/api/ask", b"\xff\xfe")):
            with self.subTest(raw):
                self.assertEqual(self.post(path, raw)[0], 400)

    def test_routes_match_exactly(self):
        self.assertEqual(self.post("/foo/bar/feed", b"{}")[0], 404)
        self.assertEqual(self.post("/api/feed", b"{}")[0], 200)


def client(behaviour):
    class Messages:
        def create(self, **kw):
            return behaviour()

    return SimpleNamespace(messages=Messages())


def reply(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))


class TestClaudeFailures(unittest.TestCase):
    def seed_node(self):
        return Memory().add("fact", "x", ("a", ("b",), "c"))

    def test_transient_errors_become_empty_thoughts(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("overloaded")
            return reply('{"kind":"none","text":"","parent_ids":[]}')

        llm = ClaudeLLM(client=client(flaky))
        self.assertEqual(llm.associate([self.seed_node()]).kind, "none")
        self.assertEqual(llm.associate([self.seed_node()]).kind, "none")
        self.assertEqual(llm.consecutive_errors, 0)

    def test_persistent_errors_are_raised(self):
        def down():
            raise RuntimeError("bad key")

        llm = ClaudeLLM(client=client(down), max_consecutive_errors=3)
        llm.associate([self.seed_node()])
        llm.associate([self.seed_node()])
        with self.assertRaises(RuntimeError):
            llm.associate([self.seed_node()])

    def test_unreadable_verdict_does_not_pass(self):
        for body in ('{"score":"high","reason":"r"}', '{"reason":"r"}'):
            with self.subTest(body):
                llm = ClaudeLLM(client=client(lambda: reply(body)))
                self.assertEqual(llm.judge(Thought("derive", "t", None, [], 1), [])[0], 0.0)

    def test_partial_thought_is_empty(self):
        llm = ClaudeLLM(client=client(lambda: reply('{"kind":"derive"}')))
        self.assertEqual(llm.associate([self.seed_node()]).kind, "none")


if __name__ == "__main__":
    unittest.main()
