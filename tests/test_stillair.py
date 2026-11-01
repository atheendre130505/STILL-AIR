import json
import os
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillair import evaluate as E
from stillair import world as W
from stillair.agent import Agent
from stillair.llm import SimLLM, Thought
from stillair.loop import build_loop
from stillair.memory import Memory
from stillair.schemas import SCHEMAS, parse_fact, parse_question
from stillair.selector import Selector


def loaded(seed=1, **kw):
    w = W.generate(seed, **kw)
    mem, llm = Memory(), SimLLM(seed)
    for s, facts in enumerate(w.sessions):
        for f in facts:
            mem.add("fact", f.text, llm.extract(f.text), session=s)
    return w, mem, llm


class TestSchemasAndWorld(unittest.TestCase):
    def test_every_template_parses_back(self):
        for sc in SCHEMAS.values():
            self.assertEqual(parse_fact(sc.f1.format(A="Kelmer", M="Zorvian")), ("Kelmer", (sc.rel,), "Zorvian"))
            self.assertEqual(parse_fact(sc.f2.format(M="Zorvian", P=sc.props[0])), ("Zorvian", ("property",), sc.props[0]))
            self.assertEqual(parse_question(sc.q_chain.format(A="Kelmer")), ("chain", "Kelmer", sc.rel))
            self.assertEqual(parse_question(sc.q_direct.format(M="Zorvian")), ("direct", "Zorvian", sc.rel))

    def test_chain_facts_arrive_in_different_sessions(self):
        w = W.generate(5)
        where = {}
        for s, facts in enumerate(w.sessions):
            for f in facts:
                if f.chain is not None:
                    where.setdefault(f.chain, set()).add(s)
        self.assertEqual(len(where), 24)
        self.assertTrue(all(len(v) == 2 for v in where.values()))

    def test_deterministic_and_truth_consistent(self):
        a, b = W.generate(7), W.generate(7)
        self.assertEqual([[f.text for f in s] for s in a.sessions], [[f.text for f in s] for s in b.sessions])
        self.assertEqual(len(a.valid_derived), 24)
        self.assertEqual(len(a.chain_questions()), 24)
        self.assertEqual(len(a.control_questions()), 12)

    def test_names_unique(self):
        w = W.generate(3)
        claims = [f.claim for s in w.sessions for f in s]
        self.assertEqual(len(claims), len(set(claims)))


class TestMemory(unittest.TestCase):
    def test_retrieval_ranks_the_matching_fact_first(self):
        w, mem, _ = loaded()
        q = w.chain_questions()[0]
        top = mem.retrieve(q.text, 1)[0][1]
        self.assertIn(top.claim[0], q.text)  # the fact about the entity in the question ranks first

    def test_decay_and_prune_only_touch_connections(self):
        mem = Memory()
        f = mem.add("fact", "Zorvian stone hums.", ("Zorvian", ("property",), "hums"))
        c = mem.add("connection", "Kelmer bridge hums", ("Kelmer", ("built_from", "property"), "hums"), "idle")
        for _ in range(100):
            pruned = mem.decay_and_prune(0.05, 0.2)
        self.assertIn(f.id, mem.nodes)
        self.assertNotIn(c.id, mem.nodes)
        self.assertGreaterEqual(mem.self_model["pruned"], 1)

    def test_reinforce_slows_pruning(self):
        mem = Memory()
        a = mem.add("connection", "alpha beta", None, "idle")
        b = mem.add("connection", "gamma delta", None, "idle")
        for _ in range(10):
            mem.reinforce(b, 0.5)
        for _ in range(30):
            mem.decay_and_prune(0.03, 0.2)
        self.assertNotIn(a.id, mem.nodes)
        self.assertIn(b.id, mem.nodes)

    def test_save_load_roundtrip(self):
        w, mem, _ = loaded()
        mem.add("connection", "x y", ("a", ("r", "property"), "z"), "idle", [1, 2])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            mem.save(p)
            m2 = Memory.load(p)
        self.assertEqual(len(m2.nodes), len(mem.nodes))
        self.assertIsNotNone(m2.find_claim(("a", ("r", "property"), "z")))

    def test_find_claim_dedupes(self):
        mem = Memory()
        mem.add("fact", "A b.", ("A", ("r",), "B"))
        self.assertIsNotNone(mem.find_claim(("A", ["r"], "B")))
        self.assertIsNone(mem.find_claim(("A", ("r",), "C")))


class TestSimLLMAndSelector(unittest.TestCase):
    def pair(self, mem, llm):
        w, mem, llm = loaded(2)
        chain = w.chain_questions()[0]
        a = next(n for n in mem.nodes.values() if n.claim and n.claim[0] in chain.text and n.claim[1][0] != "property")
        p = next(n for n in mem.nodes.values() if n.claim and n.claim[0] == a.claim[2] and n.claim[1] == ("property",))
        return w, mem, a, p

    def test_perfect_judge_separates_valid_from_confabulated(self):
        w, mem, llm = loaded(2)
        j = SimLLM(0, hallucination_rate=0.0, verifier_miss=0.0)
        _, mem, a, p = self.pair(mem, llm)
        good = j.associate([a, p])
        self.assertEqual(good.kind, "derive")
        self.assertGreater(j.judge(good, [a, p])[0], 0.5)
        other = next(n for n in mem.nodes.values() if n.claim and n.claim[1] == ("property",) and n.claim[0] != a.claim[2])
        bad = Thought("derive", "x", (a.claim[0], (a.claim[1][0], "property"), other.claim[2]), [a.id, other.id], 5)
        self.assertLess(j.judge(bad, [a, other])[0], 0.5)

    def test_selector_keep_reinforce_discard(self):
        w, mem, llm = loaded(2)
        j = SimLLM(0, 0.0, 0.0)
        _, mem, a, p = self.pair(mem, llm)
        sel = Selector(j)
        t = j.associate([a, p])
        d1 = sel.evaluate(mem, t)
        self.assertEqual(d1.action, "keep")
        mem.add("connection", t.text, t.claim, "idle", t.parents)
        t2 = j.associate([a, p])
        t2.text += " again"
        d2 = sel.evaluate(mem, t2)
        self.assertEqual(d2.action, "reinforce")
        other = next(n for n in mem.nodes.values() if n.claim and n.claim[1] == ("property",) and n.claim[0] != a.claim[2])
        bad = Thought("derive", "bad", (a.claim[0], (a.claim[1][0], "property"), other.claim[2]), [a.id, other.id], 5)
        self.assertEqual(sel.evaluate(mem, bad).action, "discard")

    def test_disabled_selector_accepts_everything(self):
        w, mem, llm = loaded(2)
        sel = Selector(SimLLM(0), enabled=False)
        self.assertEqual(sel.evaluate(mem, Thought("analogy", "junk", None, [], 3)).action, "keep")

    def test_answer_prefers_stronger_of_conflicting_connections(self):
        llm = SimLLM(0)
        weak = Memory().add("connection", "w", ("Kelmer", ("built_from", "property"), "wrong"), "idle")
        strong = Memory().add("connection", "s", ("Kelmer", ("built_from", "property"), "right"), "idle")
        strong.strength = 1.5
        q = SCHEMAS["built_from"].q_chain.format(A="Kelmer")
        self.assertEqual(llm.answer(q, [weak, strong], {}).text, "right")


class TestAgent(unittest.TestCase):
    def test_two_hops_needed_without_idle_thinking(self):
        w, mem, llm = loaded(1)
        q = w.chain_questions()[0]
        one = Agent(mem, llm, k=3, max_hops=1).ask(q.text)
        two = Agent(mem, llm, k=3, max_hops=2).ask(q.text)
        self.assertIsNone(one.answer)
        self.assertEqual(two.answer, q.answer)
        self.assertGreater(two.tokens, one.tokens)

    def test_failed_question_becomes_open_question(self):
        w, mem, llm = loaded(1)
        q = w.chain_questions()[0]
        Agent(mem, llm, k=3, max_hops=1, readonly=False).ask(q.text)
        Agent(mem, llm, k=3, max_hops=1, readonly=False).ask(q.text)
        self.assertEqual(len(mem.open_questions()), 1)

    def test_readonly_agent_leaves_memory_untouched(self):
        w, mem, llm = loaded(1)
        before = (len(mem.nodes), sum(n.hits for n in mem.nodes.values()))
        Agent(mem, llm, k=3, max_hops=2).ask(w.chain_questions()[0].text)
        self.assertEqual(before, (len(mem.nodes), sum(n.hits for n in mem.nodes.values())))


class TestLoop(unittest.TestCase):
    def run_loop(self, mode="full", seed=1, steps=40, **kw):
        w, mem, llm = loaded(seed)
        loop = build_loop(mem, seed, mode, **kw)
        loop.run(steps=steps)
        return w, mem, loop

    def test_deterministic(self):
        _, _, a = self.run_loop()
        _, _, b = self.run_loop()
        self.assertEqual([e["text"] for e in a.events], [e["text"] for e in b.events])

    def test_token_budget_stops_the_loop(self):
        w, mem, llm = loaded(1)
        loop = build_loop(mem, 1)
        used = loop.run(token_budget=500)
        self.assertGreaterEqual(used, 500)
        self.assertLess(used, 500 + 200)

    def test_no_writes_ablation_leaves_memory_unchanged(self):
        w, mem, llm = loaded(1)
        n0 = len(mem.nodes)
        loop = build_loop(mem, 1, "no_writes")
        loop.run(steps=40)
        self.assertEqual(len(mem.nodes), n0)

    def test_every_thought_is_logged(self):
        w, mem, llm = loaded(1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.jsonl")
            loop = build_loop(mem, 1, log_path=p)
            loop.run(steps=15)
            with open(p) as fh:
                lines = [json.loads(x) for x in fh]
        self.assertEqual(len(lines), 15)
        self.assertTrue(all("action" in x and "text" in x for x in lines))

    def test_selector_keeps_memory_cleaner_than_no_selector(self):
        prec = {}
        for mode in ("full", "noise_only"):
            vals = []
            for seed in range(6):
                r = E.run_episode(seed, mode)
                if r["precision"] is not None:
                    vals.append(r["precision"])
            prec[mode] = sum(vals) / len(vals)
        self.assertGreater(prec["full"], prec["noise_only"] + 0.3)

    def test_open_question_drives_idle_thought(self):
        w, mem, llm = loaded(1)
        q = w.chain_questions()[0]
        Agent(mem, llm, k=3, max_hops=1, readonly=False).ask(q.text)
        loop = build_loop(mem, 1, amplitude=0.0)
        loop.run(steps=40)
        self.assertIn("open_question", {e["source"] for e in loop.events})


class TestEvaluate(unittest.TestCase):
    def test_condition_A_needs_two_hops(self):
        r = E.run_episode(0, "A", retrieval_noise=0.0)
        self.assertEqual(r["H1k3"]["acc"], 0.0)
        self.assertEqual(r["H2k3"]["acc"], 1.0)
        self.assertEqual(r["H1k3"]["control_acc"], 1.0)

    def test_idle_compute_matched_in_tokens(self):
        b = E.run_episode(0, "noise_only")["idle_tokens"]
        c = E.run_episode(0, "full")["idle_tokens"]
        self.assertLess(abs(b - c) / c, 0.05)

    def test_summarize_shape(self):
        by = {"A": [E.run_episode(0, "A"), E.run_episode(1, "A")]}
        s = E.summarize(by)
        self.assertIn("H1k3.acc", s["A"])


class TestClaudeBackendWithFakeClient(unittest.TestCase):
    def fake(self, payloads):
        calls = []

        class Msgs:
            def create(self, **kw):
                calls.append(kw)
                body = payloads.pop(0)
                return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(body))],
                                       usage=SimpleNamespace(input_tokens=10, output_tokens=5))

        return SimpleNamespace(messages=Msgs()), calls

    def test_extract_associate_judge_answer(self):
        from stillair.claude_backend import ClaudeLLM

        client, calls = self.fake([
            {"has_triple": True, "subject": "Kelmer", "relation": "built_from", "object": "Zorvian"},
            {"kind": "derive", "text": "Kelmer is built from a stone that glows.", "parent_ids": [2, 1]},
            {"score": 0.9, "reason": "entailed"},
            {"answer": "glows", "lookup": None, "used_ids": [3]},
        ])
        llm = ClaudeLLM(client=client)
        self.assertEqual(llm.extract("The Kelmer bridge is built from Zorvian stone."), ("Kelmer", ("built_from",), "Zorvian"))
        mem = Memory()
        f1 = mem.add("fact", "rel", ("Kelmer", ("built_from",), "Zorvian"))
        f2 = mem.add("fact", "prop", ("Zorvian", ("property",), "glows"))
        t = llm.associate([f1, f2])
        self.assertEqual(t.kind, "derive")
        self.assertEqual(t.claim, ("Kelmer", ("built_from", "property"), "glows"))  # order fixed, claim computed from parents
        self.assertEqual(llm.judge(t, [f1, f2])[0], 0.9)
        self.assertEqual(llm.answer("q", [f1], {}).text, "glows")
        self.assertEqual(calls[0]["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(calls[0]["model"], "claude-opus-5-5")

    def test_refusal_becomes_empty_thought(self):
        from stillair.claude_backend import ClaudeLLM

        class Msgs:
            def create(self, **kw):
                return SimpleNamespace(stop_reason="refusal", content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=0))

        llm = ClaudeLLM(client=SimpleNamespace(messages=Msgs()))
        mem = Memory()
        n = mem.add("fact", "x", ("a", ("b",), "c"))
        self.assertEqual(llm.associate([n]).kind, "none")


class TestLive(unittest.TestCase):
    def test_feed_ask_config_snapshot(self):
        from stillair.live import LiveSession

        s = LiveSession(0)
        for _ in range(4):
            s.feed()
        q = s.world.chain_questions()[0]
        a = s.ask(q.text)
        self.assertIsNone(a["baseline"]["answer"])
        self.assertEqual(len(s.mem.open_questions()), 1 if a["still_air"]["answer"] is None else 0)
        s.config({"amplitude": 0.9, "selector": False, "writes": False})
        self.assertFalse(s.loop.selector.enabled)
        s.loop.run(steps=5)
        snap = s.snapshot()
        self.assertEqual(snap["step"], 5)
        self.assertEqual(snap["fed"], 4)
        json.dumps(snap)  # serialisable
        s._stop.set()


if __name__ == "__main__":
    unittest.main()
