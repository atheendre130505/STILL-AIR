"""The maths and memory rules need no model. The live test runs on the real local models, when asked for."""

import json
import os
import random
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillair.links import MEANING_FLOOR, Embedder, Links
from stillair.model import Model
from stillair.reminders import due_reminders, schedule_reminders, with_connections
from stillair.session import ChatSession
from stillair.state import SharedState
from stillair.subconscious import MAX_READINGS, SYSTEM, Subconscious, from_the_chat, says_nothing, uses_both


def setup():
    state = SharedState()
    links = Links(state)
    return state, links, Subconscious(SimpleNamespace(), state, links)


def note(state, text, **meta):
    return state.mem.add("fact", text, "user", meta=meta)


def link(state, text="a connection", parents=(), **meta):
    return state.mem.add("connection", text, "subconscious", parents, meta)


def page(state, text, **meta):
    return state.mem.add("reading", text, "wikipedia", meta={"topic": "t", **meta})


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


class TestConnecting(unittest.TestCase):
    def answers(self, sub, reply):
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: (reply, 5))

    def test_the_closest_pair_is_compared_first(self):
        state, _, sub = setup()
        a, b = note(state, "x", vec=[1.0, 0.0]), note(state, "y", vec=[0.6, 0.8])
        c = note(state, "z", vec=[0.9, 0.43589])
        self.assertEqual({n.id for n in sub.next_pair()}, {a.id, c.id})

    def test_unrelated_memories_are_never_compared(self):
        state, _, sub = setup()
        note(state, "apples oranges", vec=[1.0, 0.0])
        note(state, "trains planes", vec=[0.0, 1.0])
        self.assertIsNone(sub.next_pair())

    def test_two_wikipedia_pages_are_never_compared(self):
        state, _, sub = setup()
        page(state, "alpha beta", vec=[1.0, 0.0])
        page(state, "alpha beta gamma", vec=[1.0, 0.0])
        self.assertIsNone(sub.next_pair())

    def test_a_pair_is_only_compared_once(self):
        state, _, sub = setup()
        a, b = note(state, "dentist friday"), note(state, "wedding friday")
        self.answers(sub, {"connected": False, "connection": ""})
        sub.compare(a, b)
        self.assertIsNone(sub.next_pair())
        self.assertFalse(sub.think_once())

    def test_a_connection_is_stored_with_both_parents(self):
        state, _, sub = setup()
        a, b = note(state, "dentist friday"), note(state, "wedding friday")
        self.answers(sub, {"connected": True, "connection": "The dentist and the wedding clash."})
        sub.compare(a, b)
        c = state.mem.connections()[0]
        self.assertEqual(c.parents, [a.id, b.id])
        self.assertEqual(state.thoughts[-1]["outcome"], "connected")

    def test_a_connection_must_use_both_sides(self):
        state, _, sub = setup()
        a, b = note(state, "dentist friday"), note(state, "wedding friday")
        self.assertFalse(uses_both("The dentist is expensive", a, b))
        self.answers(sub, {"connected": True, "connection": "The dentist is expensive"})
        sub.compare(a, b)
        self.assertEqual(state.mem.connections(), [])

    def test_saying_nothing_is_not_a_connection(self):
        for empty in ("No further thoughts or connections identified.", "Nothing new", "", "None", "Unable to determine"):
            self.assertTrue(says_nothing(empty), empty)
        self.assertFalse(says_nothing("Both are on Friday"))

    def test_a_garbled_reply_stores_nothing(self):
        state, _, sub = setup()
        a, b = note(state, "dentist friday"), note(state, "wedding friday")
        self.answers(sub, None)
        sub.compare(a, b)
        self.assertEqual(state.mem.connections(), [])

    def test_a_failure_of_the_small_model_does_not_use_up_the_pair(self):
        state, _, sub = setup()
        a, b = note(state, "dentist friday"), note(state, "wedding friday")
        def fail(*a, **k):
            raise OSError("down")
        sub.llm = SimpleNamespace(ask_json=fail)
        with self.assertRaises(OSError):
            sub.compare(a, b)
        self.assertEqual(state.compared, set())

    def test_nothing_to_do_with_one_memory(self):
        state, _, sub = setup()
        note(state, "only one")
        self.assertFalse(sub.think_once())


class TestWikipedia(unittest.TestCase):
    def stub(self, sub, log):
        def search(q, skip):
            log.append(q)
            return [{"title": "Page", "key": "Page", "excerpt": "x"}] if "Page" not in skip else []
        sub.outside = SimpleNamespace(search=search, page=lambda k: {"title": "Page", "text": "A page."})

    def digest(self, sub, state, said, topics):
        state.undigested.append({"user": said, "assistant": "ok"})
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: ({"notes": [], "topics": topics}, 5))
        sub.digest()

    def test_a_topic_must_be_words_from_the_chat(self):
        self.assertTrue(from_the_chat("black walnut", "I planted a black walnut tree"))
        self.assertFalse(from_the_chat("deductive reasoning", "I planted a black walnut tree"))

    def test_only_topics_from_the_chat_are_queued(self):
        state, _, sub = setup()
        self.stub(sub, [])
        self.digest(sub, state, "I planted a black walnut tree", ["black walnut", "quantum physics"])
        self.assertEqual(state.topics, ["black walnut"])

    def test_a_topic_is_not_looked_up_twice(self):
        state, _, sub = setup()
        self.stub(sub, [])
        self.digest(sub, state, "I planted a black walnut tree", ["black walnut"])
        sub.read_next()
        self.digest(sub, state, "the black walnut is tall", ["black walnut"])
        self.assertEqual(state.topics, [])

    def test_reading_stores_the_page_and_its_topic(self):
        state, _, sub = setup()
        log = []
        self.stub(sub, log)
        state.topics.append("black walnut")
        sub.read_next()
        self.assertEqual(log, ["black walnut"])
        reading = state.mem.readable()[0]
        self.assertEqual((reading.kind, reading.meta["topic"]), ("reading", "black walnut"))

    def test_the_result_closest_to_the_topic_is_chosen(self):
        state, links, sub = setup()
        chosen = []
        links.embedder = SimpleNamespace(embed=lambda texts: [[0.0, 1.0], [1.0, 0.0]] if len(texts) == 2 else [[1.0, 0.0]])
        sub.outside = SimpleNamespace(search=lambda q, skip: [{"title": "Far", "key": "far", "excerpt": ""}, {"title": "Near", "key": "near", "excerpt": ""}], page=lambda k: chosen.append(k) or {"title": k, "text": "t"})
        state.topics.append("walnut")
        sub.read_next()
        self.assertEqual(chosen, ["near"])

    def test_a_result_too_far_from_the_topic_is_skipped(self):
        state, links, sub = setup()
        links.embedder = SimpleNamespace(embed=lambda texts: [[0.0, 1.0]] if texts != ["walnut"] else [[1.0, 0.0]])
        sub.outside = SimpleNamespace(search=lambda q, skip: [{"title": "Far", "key": "far", "excerpt": ""}], page=lambda k: {"title": k, "text": "t"})
        state.topics.append("walnut")
        sub.read_next()
        self.assertEqual(state.mem.readable(), [])
        self.assertEqual(state.thoughts[-1]["outcome"], "skipped")

    def test_only_the_newest_uncited_readings_are_kept(self):
        state, _, sub = setup()
        kept = page(state, "cited", cited=True)
        for i in range(MAX_READINGS + 5):
            page(state, f"page {i}")
        sub._keep_newest_readings()
        readings = [n for n in state.mem.readable() if n.kind == "reading"]
        self.assertEqual(len(readings), MAX_READINGS + 1)
        self.assertIn(kept, readings)

    def test_replies_never_see_raw_wikipedia(self):
        state, links, _ = setup()
        page(state, "walnut trees release juglone")
        self.assertEqual(links.search("walnut juglone"), [])


class TestReminders(unittest.TestCase):
    def later(self, minutes):
        return (datetime.now() + timedelta(minutes=minutes)).isoformat(timespec="seconds")

    def test_future_reminders_are_stored_and_bad_ones_skipped(self):
        state = SharedState()
        added = schedule_reminders(state, [{"text": "Doctor at 3", "remind_at": self.later(10)}, {"text": "past", "remind_at": self.later(-10)},
                                           {"text": "garbled", "remind_at": "soon"}, {"remind_at": self.later(10)}])
        self.assertEqual(added, 1)
        self.assertEqual(state.reminders[0]["text"], "Doctor at 3")

    def test_the_same_reminder_is_not_set_twice(self):
        state = SharedState()
        schedule_reminders(state, [{"text": "Doctor at 3", "remind_at": self.later(10)}])
        self.assertEqual(schedule_reminders(state, [{"text": "doctor at 3", "remind_at": self.later(20)}]), 0)

    def test_a_trailing_z_is_read_as_local_time(self):
        state = SharedState()
        self.assertEqual(schedule_reminders(state, [{"text": "x", "remind_at": self.later(10) + "Z"}]), 1)

    def test_a_reminder_is_delivered_once_when_due(self):
        state = SharedState()
        state.reminders.append({"text": "Doctor", "at": time.time() - 1, "fired": False})
        state.reminders.append({"text": "Later", "at": time.time() + 1000, "fired": False})
        self.assertEqual([r["text"] for r in due_reminders(state)], ["Doctor"])
        self.assertEqual(due_reminders(state), [])

    def test_a_reminder_appears_unprompted_without_any_model(self):
        def refuse(*a, **k):
            raise AssertionError("no model is needed to remind")
        model = SimpleNamespace(name="m", tokens_used=0, calls=0, ask_json=refuse)
        s = ChatSession(model, model, autostart=False)
        self.addCleanup(s.close)
        s.state.reminders.append({"text": "Doctor at 3pm", "at": time.time() - 1, "fired": False})
        s.deliver_reminders()
        self.assertEqual(s.state.turns[-1]["text"], "Doctor at 3pm")
        self.assertTrue(s.state.turns[-1]["unprompted"])

    def test_a_reminder_brings_its_connections(self):
        state, links, _ = setup()
        doctor = note(state, "doctor appointment friday")
        report = note(state, "blood report collected")
        link(state, "Take the blood report to the doctor appointment.", [doctor.id, report.id])
        self.assertIn("Also: Take the blood report", with_connections(links, "doctor appointment friday"))

    def test_the_reply_can_set_a_reminder(self):
        reply = {"reply": "Noted.", "used_ids": [], "wrong_ids": [], "reminders": [{"text": "Doctor", "remind_at": self.later(30)}]}
        model = SimpleNamespace(name="m", tokens_used=0, calls=0, ask_json=lambda *a, **k: (reply, 5))
        s = ChatSession(model, model, autostart=False)
        self.addCleanup(s.close)
        s.say("I have a doctor appointment soon")
        self.assertEqual(len(s.state.reminders), 1)
        self.assertEqual(len(s.state.undigested), 1)


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
    def test_notes_are_kept(self):
        state, _, sub = setup()
        state.undigested.append({"user": "Ram loves Van Gogh", "assistant": "a"})
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: ({"notes": ["Ram loves Van Gogh.", " "], "topics": []}, 5))
        sub.digest()
        self.assertEqual(state.undigested, [])
        self.assertEqual([n.text for n in state.mem.readable()], ["Ram loves Van Gogh."])

    def test_the_assistants_advice_is_not_remembered_as_the_persons(self):
        state, _, sub = setup()
        state.undigested.append({"user": "What should I take to the doctor?", "assistant": "Bring your insurance card."})
        sub.llm = SimpleNamespace(ask_json=lambda *a, **k: ({"notes": ["bring your health-insurance card and photo ID"], "topics": []}, 5))
        sub.digest()
        self.assertEqual(state.mem.readable(), [])

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

    def test_messages_are_collected_for_the_small_model(self):
        s = self.session()
        s.say("I like black walnut furniture")
        self.assertEqual(s.state.undigested[0]["user"], "I like black walnut furniture")

    def test_switching_on_checks_everything_that_happened(self):
        s = self.session()
        good, bad = link(s.state, "right"), link(s.state, "wrong")
        answer = {"wrong_ids": [bad.id], "message": "Here is what held up.", "reminders": []}
        s.conscious.llm = SimpleNamespace(ask_json=lambda *a, **k: (answer, 5))
        s.conscious.catch_up(["what is 2 plus 2"])
        self.assertTrue(good.meta["confirmed"])
        self.assertNotIn(bad.id, s.state.mem.nodes)
        self.assertTrue(s.state.turns[-1]["unprompted"])


class TestChatsAndClearing(unittest.TestCase):
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
        note(state, "alpha one")
        state.undigested.append({"user": "u", "assistant": "a"})
        state.reminders.append({"text": "x", "at": 1.0, "fired": False})
        state.add_turn("user", "keep this")
        state.clear_memory()
        self.assertEqual((len(state.mem.nodes), state.undigested, state.reminders), (0, [], []))
        self.assertEqual(state.turns[0]["text"], "keep this")

    def test_survives_a_restart_and_older_files_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            state = SharedState(path)
            a = note(state, "alpha one")
            state.undigested.append({"user": "u", "assistant": "a"})
            state.compared.add((1, 2))
            state.reminders.append({"text": "x", "at": 1.0, "fired": False})
            state.save()
            again = SharedState(path)
            self.assertIn(a.id, again.mem.nodes)
            self.assertEqual((again.undigested, again.compared, again.reminders), (state.undigested, state.compared, state.reminders))
            old = {"memory": {"next": 2, "nodes": [{"id": 1, "kind": "fact", "text": "older", "claim": None, "source": "user", "parents": [], "strength": 1.0, "hits": 0, "meta": {}}]},
                   "turns": [{"role": "user", "text": "older words"}], "thoughts": [], "tried": []}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(old, f)
            self.assertEqual(SharedState(path).turns[0]["text"], "older words")

    def test_older_files_lose_problems_and_random_material_but_keep_notes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            state = SharedState(path)
            kept = note(state, "my own words")
            state.mem.add("problem", "a sum", "user")
            state.mem.add("reading", "random page", "wikipedia")
            link(state, "from a chain", [kept.id], about=1)
            state.save()
            self.assertEqual(list(SharedState(path).mem.nodes), [kept.id])


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
