"""The maths and memory rules need no model. The live test runs on the real local models, when asked for."""

import json
import os
import random
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillair.conscious import NOTED, unanswered_messages
from stillair.links import MEANING_FLOOR, Embedder, Links
from stillair.model import Model
from stillair.session import ChatSession
from stillair.state import SharedState
from stillair.subconscious import SYSTEM, Subconscious


def setup():
    state = SharedState()
    links = Links(state)
    return state, links, Subconscious(SimpleNamespace(), state, links, random.Random(1))


def note(state, text, **meta):
    return state.mem.add("fact", text, "user", meta=meta)


def link(state, text="a connection", parents=(), **meta):
    return state.mem.add("connection", text, "subconscious", parents, meta)


def problem(state, text, **meta):
    return state.mem.add("problem", text, "digest", meta={"status": "open", "attempts": 0, "tried": [], **meta})


class TestRelatedness(unittest.TestCase):
    def test_shared_rare_word(self):
        state, links, _ = setup()
        a, b = note(state, "My dentist appointment is Friday"), note(state, "Friday is the wedding rehearsal")
        self.assertEqual(links.relation(a, b)[0], "words")

    def test_similar_meaning_with_no_shared_words(self):
        state, links, _ = setup()
        a = note(state, "My tooth hurts badly", vec=[1.0, 0.0])
        b = note(state, "Booked a dentist visit", vec=[0.8, 0.6])
        self.assertAlmostEqual(links.weight(a, b), (0.8 - MEANING_FLOOR) / (1 - MEANING_FLOOR))

    def test_activation_reaches_two_steps_away(self):
        state, links, _ = setup()
        a = note(state, "apples oranges")
        note(state, "oranges bananas")
        far = note(state, "bananas cherries")
        self.assertIn(far.id, links.spread(a))

    def test_search_stays_fast(self):
        state, links, _ = setup()
        rng = random.Random(1)
        words = [f"word{i}" for i in range(80)]
        for _ in range(130):
            note(state, " ".join(rng.sample(words, 6)), at=time.time())
        links.search("word1 word2")
        started = time.monotonic()
        links.search("word3 word4")
        self.assertLess(time.monotonic() - started, 2.0)


class TestWhatItThinksAbout(unittest.TestCase):
    def test_unsolved_problems_get_most_of_the_attention(self):
        state, _, sub = setup()
        note(state, "alpha one", at=time.time())
        p = problem(state, "a hard sum")
        picks = []
        for i in range(6):
            sub.chains_started = i
            picks.append(sub._pick_focus().id)
        self.assertEqual(picks.count(p.id), 4)

    def test_least_tried_problem_comes_first(self):
        state, _, sub = setup()
        problem(state, "tried a lot", attempts=4)
        fresh = problem(state, "not tried")
        self.assertEqual(sub._pick_focus().id, fresh.id)

    def test_recent_and_unvisited_facts_come_first(self):
        state, _, sub = setup()
        t = time.time()
        stale, fresh, visited = note(state, "b two", at=t - 86400), note(state, "a one", at=t), note(state, "c three", at=t, visits=3)
        self.assertGreater(sub.salience(fresh), sub.salience(stale))
        self.assertGreater(sub.salience(fresh), sub.salience(visited))

    def test_a_distraction_can_come_to_mind(self):
        state, _, sub = setup()
        focus = note(state, "apples oranges")
        far = note(state, "completely different subject")
        sub.rng = SimpleNamespace(random=lambda: 0.0, choice=lambda items: items[0])
        self.assertIn(far, sub._comes_to_mind({"steps": []}, focus))

    def test_nothing_to_think_about_with_one_memory(self):
        state, _, sub = setup()
        sub.links.embedder = None
        note(state, "only one")
        self.assertFalse(sub.think_once())


class TestChains(unittest.TestCase):
    def chain_on(self, state, sub, focus, reply, steps=()):
        sub.chain = {"focus": focus.id, "goal": "goal", "steps": list(steps)}
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: (reply, 5))
        sub._step()

    def test_dead_end_is_remembered_so_the_next_attempt_differs(self):
        state, _, sub = setup()
        p = problem(state, "a hard sum")
        note(state, "a hard sum needs care")
        self.chain_on(state, sub, p, {"thought": "tried factoring", "status": "dead_end", "conclusion": "", "used_ids": [], "important": False}, steps=["a", "b"])
        self.assertEqual(p.meta["tried"], ["tried factoring"])
        self.assertEqual(p.meta["attempts"], 1)
        self.assertIsNone(sub.chain)

    def test_a_garbled_reply_is_retried_not_counted_as_a_dead_end(self):
        state, _, sub = setup()
        p = problem(state, "a hard sum")
        note(state, "a hard sum needs care")
        self.chain_on(state, sub, p, None)
        self.assertIsNotNone(sub.chain)
        self.assertEqual(p.meta["attempts"], 0)

    def test_a_chain_does_not_give_up_on_its_first_step(self):
        state, _, sub = setup()
        p = problem(state, "a hard sum")
        note(state, "a hard sum needs care")
        self.chain_on(state, sub, p, {"thought": "stuck", "status": "dead_end", "conclusion": "", "used_ids": [], "important": False})
        self.assertEqual(sub.chain["steps"], ["stuck"])

    def test_a_chain_continues_until_it_has_run_its_course(self):
        state, _, sub = setup()
        f = note(state, "topic alpha")
        note(state, "topic beta")
        self.chain_on(state, sub, f, {"thought": "one", "status": "continue", "conclusion": "", "used_ids": [], "important": False})
        self.assertEqual(sub.chain["steps"], ["one"])
        for _ in range(4):
            sub._step()
        self.assertIsNone(sub.chain)

    def test_conclusion_must_cite_a_memory_it_was_shown(self):
        state, _, sub = setup()
        f, _ = note(state, "topic alpha"), note(state, "topic beta")
        self.chain_on(state, sub, f, {"thought": "t", "status": "concluded", "conclusion": "Made up", "used_ids": [999], "important": False})
        self.assertEqual(state.mem.connections(), [])

    def test_solving_a_problem_asks_the_large_model_to_check(self):
        state, _, sub = setup()
        p = problem(state, "topic alpha sum")
        note(state, "topic alpha hint")
        self.chain_on(state, sub, p, {"thought": "t", "status": "concluded", "conclusion": "The answer is 4", "used_ids": [], "important": False})
        self.assertEqual(p.meta["status"], "proposed")
        self.assertEqual(state.escalations, [state.mem.connections()[0].id])

    def test_ordinary_conclusion_does_not_bother_the_large_model(self):
        state, _, sub = setup()
        f, other = note(state, "topic alpha"), note(state, "topic alpha beta")
        self.chain_on(state, sub, f, {"thought": "t", "status": "concluded", "conclusion": "They relate in a specific way", "used_ids": [other.id], "important": False})
        self.assertEqual(len(state.mem.connections()), 1)
        self.assertEqual(state.escalations, [])


class TestIncubation(unittest.TestCase):
    def test_after_repeated_failure_it_rests_and_forgets_the_attempts(self):
        state, _, sub = setup()
        p = problem(state, "a hard sum", tried=["one", "two"])
        sub._end_chain({"goal": "g", "steps": ["three"]}, p, "dead end", "", 0)
        self.assertEqual(p.meta["tried"], [])
        self.assertEqual(p.meta["archived"], ["one", "two", "three"])
        self.assertGreater(p.meta["rest_until"], time.time())

    def test_a_resting_problem_is_not_picked(self):
        state, _, sub = setup()
        problem(state, "a hard sum", rest_until=time.time() + 100)
        note(state, "alpha one", at=time.time())
        self.assertEqual(sub._pick_focus().kind, "fact")

    def test_new_related_information_wakes_a_sleeping_problem_first(self):
        state, _, sub = setup()
        p = problem(state, "plants dying", vec=[1.0, 0.0], rest_until=time.time() + 100)
        other = problem(state, "a sum", vec=[0.0, 1.0])
        new = note(state, "the heating is on", vec=[1.0, 0.1])
        sub._wake_problems(new)
        self.assertEqual(sub._pick_focus().id, p.id)
        self.assertNotIn("woken_by", other.meta)

    def test_what_woke_it_comes_to_mind(self):
        state, _, sub = setup()
        p = problem(state, "aaa bbb")
        spark = note(state, "ccc ddd")
        self.assertIn(spark, sub._comes_to_mind({"steps": [], "woken_by": spark.id}, p))

    def test_targeted_reading_sends_keywords_only_and_wakes_the_problem(self):
        state, _, sub = setup()
        p = problem(state, "Why do my tomato plants keep wilting after watering")
        sent = []
        sub.outside = SimpleNamespace(about=lambda k: sent.extend(k) or {"title": "Juglone", "text": "A compound from walnut trees."}, random=lambda: None)
        sub.rng = SimpleNamespace(choice=lambda items: items[0], random=lambda: 1.0)
        sub.read_something()
        self.assertLessEqual(len(sent), 3)
        self.assertTrue(p.meta["woken_by"])
        self.assertEqual([n.kind for n in state.mem.readable()].count("reading"), 1)

    def test_replies_never_see_outside_reading(self):
        state, links, _ = setup()
        state.mem.add("reading", "walnut trees release juglone", "wikipedia", meta={})
        self.assertEqual(links.search("walnut juglone"), [])
        self.assertEqual(len(links.search("walnut juglone", include_outside=True)), 1)

    def test_unused_reading_expires_but_cited_reading_stays(self):
        state, _, sub = setup()
        kept = state.mem.add("reading", "one", "wikipedia", meta={"read_at_chain": 0, "cited": True})
        gone = state.mem.add("reading", "two", "wikipedia", meta={"read_at_chain": 0})
        sub.chains_started = 20
        sub._expire_readings()
        self.assertIn(kept.id, state.mem.nodes)
        self.assertNotIn(gone.id, state.mem.nodes)


class TestMemoryKeepsOnlyWhatMatters(unittest.TestCase):
    def test_unused_connections_weaken_and_are_dropped(self):
        state, _, _ = setup()
        c, fact = link(state), note(state, "alpha one")
        for _ in range(20):
            state.mem.fade(0.05, 0.2)
        self.assertNotIn(c.id, state.mem.nodes)
        self.assertIn(fact.id, state.mem.nodes)

    def test_using_a_conclusion_makes_it_and_its_sources_stronger(self):
        state, links, _ = setup()
        a = note(state, "alpha one")
        c = link(state, "a conclusion", [a.id])
        links.learn_from_use([c.id])
        self.assertGreater(c.strength, 1.0)
        self.assertGreater(a.strength, 1.0)

    def test_weakest_go_first_past_the_limit(self):
        state, _, _ = setup()
        weak, strong = link(state, "weak"), link(state, "strong")
        weak.strength, strong.strength = 0.5, 1.5
        state.mem.keep_strongest(1)
        self.assertEqual([n.id for n in state.mem.connections()], [strong.id])

    def test_most_relevant_favours_own_words(self):
        state = SharedState()
        found = [(0.9, link(state, f"h {i}")) for i in range(5)] + [(0.9, note(state, f"f {i}")) for i in range(8)] + [(0.01, note(state, "weak"))]
        self.assertEqual([n.kind for n in Links.most_relevant(found)], ["fact"] * 6 + ["connection"] * 3)


class TestDigesting(unittest.TestCase):
    def test_notes_and_problems_are_kept_and_solved_ones_close(self):
        state, _, sub = setup()
        old = problem(state, "an old puzzle")
        state.undigested.append({"user": "u", "assistant": "a"})
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: ({"notes": ["Ram loves Van Gogh."], "problems": ["Prove the identity holds."], "solved_ids": [old.id]}, 5))
        sub.digest()
        self.assertEqual(state.undigested, [])
        self.assertEqual(old.meta["status"], "solved")
        self.assertEqual([p.text for p in state.open_problems()], ["Prove the identity holds."])
        self.assertTrue(any(n.text == "Ram loves Van Gogh." for n in state.mem.readable()))

    def test_blank_items_are_ignored(self):
        state, _, sub = setup()
        state.undigested.append({"user": "u", "assistant": "a"})
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: ({"notes": [" "], "problems": [""], "solved_ids": []}, 5))
        sub.digest()
        self.assertEqual(len(state.mem.nodes), 0)

    def test_failed_digest_is_retried_once_then_given_up(self):
        state, _, sub = setup()
        state.undigested.append({"user": "u", "assistant": "a"})
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: (None, 0))
        sub.digest()
        self.assertEqual(len(state.undigested), 1)
        sub.digest()
        self.assertEqual(state.undigested, [])


class TestWithLargeModelOff(unittest.TestCase):
    def session(self):
        def refuse(*a, **k):
            raise AssertionError("the large model must not be called")
        large = SimpleNamespace(name="large", tokens_used=0, calls=0, ask_json=refuse)
        small = SimpleNamespace(name="small", ask_json=lambda *a, **k: (None, 0))
        s = ChatSession(large, small, autostart=False)
        s.conscious_on = False
        self.addCleanup(s.close)
        return s

    def test_message_goes_to_the_subconscious_only(self):
        s = self.session()
        self.assertEqual(s.say("I am stuck on a sum"), NOTED)
        self.assertTrue(s.state.undigested[0]["direct"])

    def test_switching_on_checks_everything_that_happened(self):
        s = self.session()
        p = problem(s.state, "a sum", status="solved")
        good, bad = link(s.state, "right", about=p.id), link(s.state, "wrong", about=p.id)
        s.conscious.llm = SimpleNamespace(ask_json=lambda *a, **k: ({"wrong_ids": [bad.id], "message": "Here is what held up."}, 5))
        s.conscious.catch_up(["what is 2 plus 2"])
        self.assertTrue(good.meta["confirmed"])
        self.assertNotIn(bad.id, s.state.mem.nodes)
        self.assertEqual(p.meta["status"], "open")
        self.assertTrue(s.state.turns[-1]["unprompted"])

    def test_conclusion_is_posted_marked_as_unchecked(self):
        s = self.session()
        p = problem(s.state, "a sum")
        c = link(s.state, "The answer is 4", about=p.id)
        s.state.escalations.append(c.id)
        s.post_unchecked()
        self.assertIn("not checked", s.state.turns[-1]["text"])
        self.assertTrue(s.state.turns[-1]["unprompted"])
        self.assertEqual(p.meta["status"], "solved")


class TestChatsAndClearing(unittest.TestCase):
    def test_counts_unanswered_messages(self):
        state = SharedState()
        state.add_turn("user", "hello")
        state.add_turn("assistant", "one", unprompted=True)
        self.assertEqual(unanswered_messages(state), 1)

    def test_new_and_cleared_chats_keep_memory(self):
        state = SharedState()
        state.add_turn("user", "first")
        note(state, "alpha one")
        state.new_chat()
        self.assertEqual(state.turns, [])
        state.open_chat(1)
        state.clear_chat()
        self.assertEqual(state.turns, [])
        self.assertEqual(len(state.mem.nodes), 1)

    def test_clear_memory_keeps_chats(self):
        state, _, _ = setup()
        problem(state, "a sum")
        state.undigested.append({"user": "u", "assistant": "a"})
        state.add_turn("user", "keep this")
        state.clear_memory()
        self.assertEqual((len(state.mem.nodes), state.undigested, state.open_problems()), (0, [], []))
        self.assertEqual(state.turns[0]["text"], "keep this")

    def test_survives_a_restart_and_older_files_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            state = SharedState(path)
            a = problem(state, "a sum")
            state.undigested.append({"user": "u", "assistant": "a"})
            state.save()
            again = SharedState(path)
            self.assertEqual(again.open_problems()[0].id, a.id)
            self.assertEqual(again.undigested, state.undigested)
            old = {"memory": {"next": 2, "nodes": [{"id": 1, "kind": "fact", "text": "older", "claim": None, "source": "user", "parents": [], "strength": 1.0, "hits": 0, "meta": {}}]},
                   "turns": [{"role": "user", "text": "older words"}], "thoughts": [], "tried": []}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(old, f)
            self.assertEqual(SharedState(path).turns[0]["text"], "older words")


@unittest.skipUnless(os.environ.get("STILLAIR_LIVE_TESTS"), "set STILLAIR_LIVE_TESTS=1 to run on the real local models")
class TestLiveLocalModels(unittest.TestCase):
    def test_small_model_digests_and_thinks_for_real(self):
        url = "http://127.0.0.1:11434/v1"
        state = SharedState()
        links = Links(state, Embedder("all-minilm", url))
        sub = Subconscious(Model("phi4-mini", url, SYSTEM, timeout=600.0), state, links)
        state.undigested.append({"user": "My dentist appointment is on Friday at 3pm.", "assistant": "Noted."})
        note(state, "My sister's wedding rehearsal is on Friday afternoon.", at=time.time())
        self.assertTrue(sub.think_once())
        self.assertEqual(state.undigested, [])
        self.assertTrue(sub.think_once())
        self.assertTrue(state.thoughts)


if __name__ == "__main__":
    unittest.main()
