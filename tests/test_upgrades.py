"""Tests for the memory and thinking upgrades and the OpenAI-compatible backend."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillair.backends.base import Answer
from stillair.backends.openai_compat import OpenAICompatLLM
from stillair.backends.prompted import JUDGE, THOUGHT
from stillair.backends.sim import SimLLM
from stillair.core.agent import Agent
from stillair.core.factory import build_loop
from stillair.core.loop import seeds_share_entity
from stillair.core.memory import Memory
from stillair.core.perturb import Perturbation
from stillair.evaluation import episode as E
from stillair.evaluation.real_run import run_real
from stillair.live.session import LiveSession
from stillair.world.schemas import SCHEMAS

import random

QUESTION = SCHEMAS["built_from"].q_chain.format(A="Kelmer")


def linked_memory():
    mem = Memory()
    r = mem.add("fact", "The Kelmer bridge is built from Zorvian stone.", ("Kelmer", ("built_from",), "Zorvian"))
    p = mem.add("fact", "Zorvian stone hums when struck.", ("Zorvian", ("property",), "hums when struck"))
    link = mem.add("connection", "The Kelmer bridge is built from Zorvian stone, and Zorvian stone hums when struck.",
                   ("Kelmer", ("built_from", "property"), "hums when struck"), "idle", [r.id, p.id], meta={"thought_kind": "derive"})
    return mem, link


class TestTrialLinks(unittest.TestCase):
    def test_unconfirmed_link_cannot_answer_until_rederived(self):
        mem, link = linked_memory()
        ag = Agent(mem, SimLLM(0, 0, 0), k=3, max_hops=1, trial_links=True)
        self.assertIsNone(ag.ask(QUESTION).answer)
        mem.reinforce(link)
        self.assertEqual(ag.ask(QUESTION).answer, "hums when struck")

    def test_expiry_removes_only_unconfirmed_links(self):
        mem, link = linked_memory()
        kept = mem.add("connection", "another link", ("A", ("r", "property"), "x"), "idle", born=0)
        mem.reinforce(kept)
        self.assertEqual(mem.expire_unconfirmed(59, 60), [])
        self.assertEqual(mem.expire_unconfirmed(60, 60), [link.id])
        self.assertIn(kept.id, mem.nodes)


class TestVerifyOnUse(unittest.TestCase):
    def test_false_link_falls_back_to_the_two_hop_path(self):
        mem, link = linked_memory()
        link.claim = ("Kelmer", ("built_from", "property"), "floats on water")  # no longer follows from its parents
        mem._claims.clear()
        ag = Agent(mem, SimLLM(0, 0, 0), k=3, max_hops=2, verify_on_use=True)
        self.assertEqual(ag.ask(QUESTION).answer, "hums when struck")

    def test_true_link_answers_in_one_hop(self):
        mem, _ = linked_memory()
        tr = Agent(mem, SimLLM(0, 0, 0), k=3, max_hops=2, verify_on_use=True).ask(QUESTION)
        self.assertEqual((tr.answer, tr.hops), ("hums when struck", 1))


class TestThinkingUpgrades(unittest.TestCase):
    def test_seeds_share_entity(self):
        mem, link = linked_memory()
        facts = [n for n in mem.nodes.values() if n.kind == "fact"]
        self.assertTrue(seeds_share_entity(facts))
        self.assertFalse(seeds_share_entity(facts[:1] + [link]))

    def test_skip_spends_no_model_call(self):
        class Counting(SimLLM):
            calls = 0

            def associate(self, seeds, nudge=None):
                Counting.calls += 1
                return super().associate(seeds, nudge)

        mem = Memory()
        mem.add("fact", "Zorvian stone hums when struck.", ("Zorvian", ("property",), "hums when struck"))
        loop = build_loop(mem, 0, llm=Counting(0), skip_unlinked=True)
        loop.run(steps=5)
        self.assertEqual(Counting.calls, 0)
        self.assertEqual(loop.tokens, 5)

    def test_adaptive_amplitude(self):
        mem, _ = linked_memory()
        mem.add("question", QUESTION, None, meta={"status": "open"})
        p = Perturbation(random.Random(1), 0.5, adaptive=True)
        p.focus = mem.vec("Zorvian stone")
        p.curiosity = 1.0
        draws = [p.draw(mem) for _ in range(40)]
        self.assertEqual({d.amplitude for d in draws if d.source == "open_question"}, {0.0})
        self.assertEqual({d.amplitude for d in draws if d.source == "drift"}, {0.75})

    def test_flags_off_matches_default_episode(self):
        self.assertEqual(E.run_episode(1, "full"), E.run_episode(1, "full", trial_links=False, skip_unlinked=False))

    def test_all_upgrades_run_end_to_end(self):
        r = E.run_episode(1, "full", trial_links=True, verify_on_use=True, trial_expiry=60, skip_unlinked=True, adaptive_noise=True)
        self.assertGreater(r["idle_steps"], 0)


class TestAnswerMatching(unittest.TestCase):
    def test_rewording_counts_but_wrong_answers_do_not(self):
        self.assertTrue(E.answer_matches("It hums when struck.", "hums when struck"))
        self.assertFalse(E.answer_matches("floats on water", "hums when struck"))
        self.assertFalse(E.answer_matches(None, "hums when struck"))


class FakeProvider:
    """A scripted OpenAI-compatible server recording what it receives."""

    def __init__(self, replies):
        provider = self
        self.replies, self.requests = list(replies), []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append((self.path, dict(self.headers), body))
                status, payload, headers = provider.replies.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def ok(content, tokens=30, finish="stop"):
    return 200, {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"total_tokens": tokens}}, {}


class TestOpenAICompat(unittest.TestCase):
    def provider(self, replies):
        p = FakeProvider(replies)
        self.addCleanup(p.close)
        return p

    def llm(self, p, **kw):
        self.slept = []
        return OpenAICompatLLM(model="m", base_url=p.url, api_key=kw.pop("api_key", "k"), rpm=0, sleep=self.slept.append, **kw)

    def test_request_shape_and_token_count(self):
        p = self.provider([ok('{"score": 0.9, "reason": "r"}', 42)])
        llm = self.llm(p)
        self.assertEqual(llm._call("hello", JUDGE)[0]["score"], 0.9)
        path, headers, body = p.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertIs(type(body["response_format"]["json_schema"]["schema"]), dict)
        self.assertEqual(llm.tokens_used, 42)

    def test_no_key_sends_no_auth_header(self):
        p = self.provider([ok("{}")])
        self.llm(p, api_key="")._call("x", JUDGE)
        self.assertNotIn("Authorization", p.requests[0][1])

    def test_missing_key_for_hosted_default_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            OpenAICompatLLM()

    def test_429_waits_for_retry_after_then_succeeds(self):
        p = self.provider([(429, {"error": "slow"}, {"Retry-After": "7"}), ok('{"score": 1, "reason": "r"}')])
        llm = self.llm(p)
        self.assertEqual(llm._call("x", JUDGE)[0]["score"], 1)
        self.assertEqual(self.slept, [7.0])

    def test_exhausted_daily_quota_is_not_waited_out(self):
        p = self.provider([(429, {"error": "daily"}, {"Retry-After": "40000"})])
        llm = self.llm(p, max_consecutive_errors=1)
        with self.assertRaisesRegex(RuntimeError, "HTTP 429.*daily"):
            llm._call("x", JUDGE)
        self.assertEqual(self.slept, [])

    def test_falls_back_when_json_schema_is_unsupported(self):
        p = self.provider([(400, {"error": {"message": "response_format json_schema not supported"}}, {}), ok('{"score": 1, "reason": "r"}')])
        llm = self.llm(p)
        self.assertEqual(llm._call("x", JUDGE)[0]["score"], 1)
        self.assertEqual(p.requests[1][2]["response_format"], {"type": "json_object"})
        self.assertFalse(llm.use_json_schema)

    def test_malformed_json_from_the_model_is_retried(self):
        p = self.provider([(400, {"error": {"code": "json_validate_failed", "message": "Failed to validate JSON"}}, {}), ok('{"score": 1, "reason": "r"}')])
        llm = self.llm(p)
        self.assertEqual(llm._call("x", JUDGE)[0]["score"], 1)
        self.assertTrue(llm.use_json_schema)

    def test_reported_tokens_replace_the_word_estimate(self):
        p = self.provider([ok('{"kind":"none","text":"","parent_ids":[]}', 50)])
        mem, _ = linked_memory()
        loop = build_loop(mem, 0, llm=self.llm(p))
        loop.step()
        self.assertEqual(loop.tokens, 50)

    def test_rejected_link_is_hidden_from_the_retry(self):
        p = self.provider([
            ok('{"answer":"wrong","lookup":null,"used_ids":[3]}'),
            ok('{"score":0.0,"reason":"no"}'),
            ok('{"answer":null,"lookup":null,"used_ids":[]}'),
        ])
        mem, _ = linked_memory()
        Agent(mem, self.llm(p), k=3, max_hops=1, verify_on_use=True).ask(QUESTION)
        self.assertNotIn("[3]", p.requests[2][2]["messages"][1]["content"])


class StubModel(SimLLM):
    name, model, usage_includes_input, tokens_used = "stub", "stub-1", True, 0


class TestDashboardWithRealModel(unittest.TestCase):
    def session(self, llm, **kw):
        s = LiveSession(0, llm, **kw)
        self.addCleanup(lambda: s._stop.set())
        for _ in range(4):
            s.feed()
        return s

    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not condition():
            time.sleep(0.01)
        self.assertTrue(condition())

    def test_dashboard_stays_live_during_a_slow_model_call(self):
        entered, release = threading.Event(), threading.Event()

        class Slow(StubModel):
            def associate(self, seeds, nudge=None):
                entered.set()
                release.wait(5)
                return super().associate(seeds, nudge)

        s = self.session(Slow(0))
        self.addCleanup(release.set)
        thread = threading.Thread(target=s.loop.step, daemon=True)
        s.loop.cfg.skip_unlinked = False
        thread.start()
        self.assertTrue(entered.wait(5))
        result = []
        poll = threading.Thread(target=lambda: result.append(s.snapshot()), daemon=True)
        poll.start()
        poll.join(2)
        self.assertTrue(result, "snapshot blocked while the model was thinking")
        release.set()
        thread.join(5)
        self.assertEqual(s.snapshot()["step"], 1)

    def test_snapshot_describes_the_real_backend(self):
        snap = self.session(StubModel(0)).snapshot()
        self.assertEqual((snap["backend"], snap["model"]), ("stub", "stub-1"))
        json.dumps(snap)

    def test_token_budget_stops_the_loop(self):
        model = StubModel(0)
        model.tokens_used = 100
        s = self.session(model, token_budget=100)
        s.config({"running": True})
        self.wait_for(lambda: not s.running)
        self.assertEqual(s.loop.step_no, 0)
        self.assertIn("budget", s.error)

    def test_model_failure_is_reported_not_silent(self):
        class Down(StubModel):
            def associate(self, seeds, nudge=None):
                raise RuntimeError("HTTP 401")

        s = self.session(Down(0))
        s.loop.cfg.skip_unlinked = False
        s.config({"running": True})
        self.wait_for(lambda: s.error is not None)
        self.assertIn("HTTP 401", s.error)
        self.assertFalse(s.running)

    def test_reworded_answer_is_marked_correct(self):
        class Wordy(StubModel):
            def answer(self, question, context, scratch):
                return Answer("It is said that " + truth, None, [], 5)

        s = self.session(Wordy(0))
        q = s.world.chain_questions()[0]
        truth = q.answer
        self.assertTrue(s.ask(q.text)["still_air"]["correct"])


class TestRealRun(unittest.TestCase):
    def test_simulator_baseline_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            md = run_real(None, "sim", range(2), 5, d)
            self.assertIn("C + all upgrades", md)
            with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)["conditions"]), 3)


if __name__ == "__main__":
    unittest.main()
