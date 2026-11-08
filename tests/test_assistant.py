"""The idle algorithm's maths needs no model. The live test runs the real local models, when asked for."""

import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillair.assistant.associations import MAX_LEARNED, MEANING_FLOOR, TIME_WEIGHT, Associations, OllamaEmbedder
from stillair.assistant.proactive import Voice, fire_due_reminders, schedule_reminders
from stillair.assistant.state import SharedState
from stillair.assistant.subconscious import Subconscious
from stillair.backends.openai_compat import OpenAICompatLLM
from stillair.core.activation import spread


def setup():
    state = SharedState()
    links = Associations(state)
    return state, links, Subconscious(SimpleNamespace(), state, links)


def note(state, text, **meta):
    return state.mem.add("fact", text, None, "user", meta=meta)


class TestRelatedness(unittest.TestCase):
    def test_shared_rare_word_links_memories(self):
        state, links, _ = setup()
        a, b = note(state, "My dentist appointment is Friday"), note(state, "Friday is the wedding rehearsal")
        self.assertGreater(links.weight(a, b), 0)

    def test_similar_meaning_links_memories_with_no_shared_words(self):
        state, links, _ = setup()
        a = note(state, "My tooth hurts badly", vec=[1.0, 0.0])
        b = note(state, "Booked a dentist visit", vec=[0.8, 0.6])
        self.assertAlmostEqual(links.weight(a, b), (0.8 - MEANING_FLOOR) / (1 - MEANING_FLOOR))

    def test_unrelated_meaning_scores_zero(self):
        state, links, _ = setup()
        a, b = note(state, "alpha beta", vec=[1.0, 0.0]), note(state, "gamma delta", vec=[0.1, 0.99])
        self.assertEqual(links.weight(a, b), 0)

    def test_time_link_fades_with_distance(self):
        state, links, _ = setup()
        t = time.time()
        a, b, c = note(state, "alpha one", at=t), note(state, "gamma two", at=t), note(state, "delta three", at=t - 7200)
        self.assertAlmostEqual(links.weight(a, b), TIME_WEIGHT)
        self.assertLess(links.weight(a, c), TIME_WEIGHT / 10)

    def test_used_paths_get_stronger_up_to_a_cap(self):
        state, links, _ = setup()
        a = note(state, "shared topic here")
        b = note(state, "shared topic there")
        before = links.weight(a, b)
        links.learn_from_use([a.id, b.id])
        self.assertGreater(links.weight(a, b), before)
        for _ in range(10):
            links.learn_from_use([a.id, b.id])
        self.assertEqual(state.learned[frozenset((a.id, b.id))], MAX_LEARNED)

    def test_using_a_hunch_strengthens_the_memories_behind_it(self):
        state, links, _ = setup()
        a, b = note(state, "alpha one"), note(state, "gamma two")
        hunch = state.mem.add("connection", "They connect", None, "subconscious", [a.id, b.id])
        links.learn_from_use([hunch.id, 999])
        self.assertGreater(a.strength, 1.0)
        self.assertGreater(state.learned[frozenset((hunch.id, b.id))], 0)


class TestIdleAlgorithm(unittest.TestCase):
    def test_activation_reaches_a_memory_two_steps_away(self):
        state, links, _ = setup()
        a = note(state, "apples oranges")
        b = note(state, "oranges bananas")
        c = note(state, "bananas cherries")
        self.assertIn(c.id, spread(state.mem, a, links.weight))

    def test_unresolved_memory_is_picked_first(self):
        state, _, sub = setup()
        t = time.time()
        fresh, stale_open = note(state, "alpha one", at=t), note(state, "gamma two", at=t - 86400, open=True)
        self.assertGreater(sub.salience(stale_open), sub.salience(fresh))

    def test_revisiting_lowers_priority(self):
        state, _, sub = setup()
        t = time.time()
        a, b = note(state, "alpha one", at=t), note(state, "gamma two", at=t, visits=3)
        self.assertGreater(sub.salience(a), sub.salience(b))

    def test_tried_pairs_are_never_repeated_and_it_comes_to_rest(self):
        state, _, sub = setup()
        note(state, "shared topic here", at=time.time())
        note(state, "shared topic there", at=time.time())
        seen = set()
        while (pair := sub.next_pair()) is not None:
            key = frozenset((pair[0].id, pair[1].id))
            self.assertNotIn(key, seen)
            seen.add(key)
            state.tried.add(key)
        self.assertEqual(len(seen), 1)


class TestContinuousThinking(unittest.TestCase):
    def test_second_round_retries_pairs_and_reaches_further(self):
        state, _, sub = setup()
        a = note(state, "apples oranges", at=time.time())
        b = note(state, "oranges bananas", at=time.time())
        state.tried.add(frozenset((a.id, b.id)))
        self.assertIsNone(sub.next_pair())
        state.tried.clear()
        sub.rounds = 1
        self.assertIsNotNone(sub.next_pair())


class TestSpeakingFirst(unittest.TestCase):
    def test_reminder_is_posted_once_when_due(self):
        state = SharedState()
        schedule_reminders(state, [{"text": "Dentist at 3", "remind_at": "2030-01-05T12:00:00"}], now_epoch=0)
        at = state.reminders[0]["at"]
        self.assertEqual(fire_due_reminders(state, now_epoch=at - 1), 0)
        self.assertEqual(fire_due_reminders(state, now_epoch=at), 1)
        self.assertEqual(fire_due_reminders(state, now_epoch=at + 1), 0)
        self.assertEqual(state.turns[-1], {"role": "assistant", "text": "Dentist at 3", "unprompted": True})

    def test_bad_past_and_duplicate_reminders_are_skipped(self):
        state = SharedState()
        schedule_reminders(state, [{"text": "x", "remind_at": "not a date"}, {"text": "old", "remind_at": "2001-01-01T00:00:00"},
                                   {"text": "same", "remind_at": "2030-01-05T12:00:00"}, {"text": "SAME", "remind_at": "2030-01-06T12:00:00"}],
                           now_epoch=1e9 + 1e7)
        self.assertEqual([r["text"] for r in state.reminders], ["same"])

    def test_stops_speaking_after_two_unanswered_messages(self):
        state = SharedState()
        state.mem.add("connection", "a hunch", None, "subconscious", [])
        voice = Voice(SimpleNamespace(), state)
        state.add_turn("assistant", "one", unprompted=True)
        self.assertTrue(voice.should_review())
        state.add_turn("assistant", "two", unprompted=True)
        self.assertFalse(voice.should_review())
        state.add_turn("user", "hello")
        self.assertTrue(voice.should_review())


class TestBackendView(unittest.TestCase):
    def test_link_is_labelled_with_its_strongest_reason(self):
        state, links, _ = setup()
        t = time.time()
        a = note(state, "alpha one", at=t, vec=[1.0, 0.0])
        b = note(state, "gamma two", at=t, vec=[0.9, 0.436])
        c = note(state, "delta three", at=t, vec=[0.0, 1.0])
        self.assertEqual(links.relation(a, b)[0], "meaning")
        self.assertEqual(links.relation(a, c)[0], "time")

    def test_links_update_when_memory_changes(self):
        state, links, _ = setup()
        note(state, "shared topic here")
        self.assertEqual(links.strongest_links(), [])
        note(state, "shared topic there")
        drawn = links.strongest_links()
        self.assertEqual([d["kind"] for d in drawn], ["words"])
        self.assertIs(links.strongest_links(), drawn)


class TestChats(unittest.TestCase):
    def test_new_chat_is_blank_but_memory_is_shared(self):
        state = SharedState()
        state.add_turn("user", "hello there")
        note(state, "alpha one")
        state.new_chat()
        self.assertEqual(state.turns, [])
        self.assertEqual(state.mem.counts()["fact"], 1)

    def test_blank_chats_do_not_pile_up(self):
        state = SharedState()
        state.new_chat()
        state.new_chat()
        self.assertEqual(len(state.chats), 1)

    def test_old_chat_can_be_reopened_with_its_title(self):
        state = SharedState()
        state.add_turn("user", "first chat message")
        state.new_chat()
        state.add_turn("user", "second chat message")
        self.assertTrue(state.open_chat(1))
        self.assertEqual(state.turns[0]["text"], "first chat message")
        self.assertEqual(state.chats[0]["title"], "first chat message")
        self.assertFalse(state.open_chat(99))

    def test_file_saved_before_chats_existed_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            old = {"memory": SharedState().mem.to_json(), "turns": [{"role": "user", "text": "older words"}], "thoughts": [], "tried": []}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(old, f)
            self.assertEqual(SharedState(path).turns[0]["text"], "older words")


class TestPersistence(unittest.TestCase):
    def test_everything_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            state = SharedState(path)
            a, b = note(state, "alpha one", vec=[1.0], at=1.0), note(state, "gamma two")
            state.tried.add(frozenset((a.id, b.id)))
            state.learned[frozenset((a.id, b.id))] = 0.5
            state.add_turn("user", "hi")
            state.save()
            again = SharedState(path)
            self.assertEqual(again.mem.nodes[a.id].meta["vec"], [1.0])
            self.assertEqual(again.tried, state.tried)
            self.assertEqual(again.learned, state.learned)
            self.assertEqual(again.turns, state.turns)


@unittest.skipUnless(os.environ.get("STILLAIR_LIVE_TESTS"), "set STILLAIR_LIVE_TESTS=1 to run on the real local models")
class TestLiveLocalModels(unittest.TestCase):
    def test_small_model_and_embeddings_run_for_real(self):
        url = "http://127.0.0.1:11434/v1"
        state = SharedState()
        links = Associations(state, OllamaEmbedder("all-minilm", url))
        llm = OpenAICompatLLM(model="phi4-mini", base_url=url, api_key="", timeout=600.0)
        sub = Subconscious(llm, state, links)
        note(state, "My dentist appointment is on Friday at 3pm.", at=time.time())
        note(state, "My sister's wedding rehearsal is on Friday afternoon.", at=time.time())
        self.assertTrue(sub.think_once())
        self.assertTrue(links.meaning_working)
        self.assertEqual(len(state.thoughts), 1)
        print("\nlive thought:", json.dumps(state.thoughts[0], indent=1))


if __name__ == "__main__":
    unittest.main()
