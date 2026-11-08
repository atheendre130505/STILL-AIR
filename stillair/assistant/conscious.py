"""The conscious mind: the larger main model that answers when you speak to it.

It reads what you said, related memories, and hunches the subconscious found. Hunches come
from a small model, so they are labelled unverified, and the main model acts as their judge:
what it actually uses gets stronger and teaches the memory which paths are useful.
"""

from __future__ import annotations

import re

from ..core.memory import tokens
from .associations import Associations, now
from .proactive import current_time, schedule_reminders
from .state import SharedState

REPLY = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "used_ids": {"type": "array", "items": {"type": "integer"}},
        "unresolved": {"type": "boolean"},
        "reminders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "remind_at": {"type": "string"}},
                "required": ["text", "remind_at"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "used_ids", "unresolved", "reminders"],
    "additionalProperties": False,
}
RECENT_TURNS = 8
MAX_FRESH_HUNCHES = 5
DECAY_PER_REPLY, PRUNE_FLOOR = 0.02, 0.2  # measured in conversation, so long idle runs do not erase hunches
MEMORIES_PER_REPLY = 8
MIN_NOTE_WORDS = 3  # shorter than this is chat, like "ok thanks", not something to remember
DUPLICATE_SIMILARITY = 0.95


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(tokens(s)) >= MIN_NOTE_WORDS]


class Conscious:
    system = (
        "You are a helpful assistant with a long-term memory of this person. Be direct and natural. "
        "Reply with JSON only."
    )

    def __init__(self, llm, state: SharedState, associations: Associations):
        self.llm, self.state, self.associations = llm, state, associations
        llm.system = self.system
        self.mem = state.mem

    def _remember(self, text: str, unresolved: bool) -> None:
        for sentence in _sentences(text):
            if self.mem.max_similarity(sentence) < DUPLICATE_SIMILARITY:
                self.mem.add("fact", sentence, None, "user", meta={"at": now(), "open": unresolved})

    def _prompt(self, text: str, recalled: list) -> tuple:
        relevant = {n.id for n in recalled}
        unseen = [n for n in self.mem.nodes.values() if n.kind == "connection" and not n.meta.get("seen")]
        fresh = sorted(unseen, key=lambda n: (n.id not in relevant, -n.id))[:MAX_FRESH_HUNCHES]
        hunches = {n.id: n for n in [*fresh, *(n for n in recalled if n.kind == "connection")]}
        notes = [n for n in recalled if n.kind == "fact"]
        parts = []
        if notes:
            parts.append("What this person told you before:\n" + "\n".join(f"[{n.id}] {n.text}" for n in notes))
        if hunches:
            parts.append("Hunches your subconscious formed while idle. They may be wrong; use one only if it truly "
                         "helps, and bring up a useful new one naturally:\n" + "\n".join(f"[{n.id}] {n.text}" for n in hunches.values()))
        recent = "\n".join(f"{t['role']}: {t['text']}" for t in self.state.turns[-RECENT_TURNS:])
        if recent:
            parts.append("Recent conversation:\n" + recent)
        parts.append(f"It is now {current_time()}.\n\nuser: {text}\n\nWrite your reply. List in used_ids the ids of any memories "
                     "or hunches you relied on. Set unresolved to true if something the person raised is still open and more "
                     "thought could help. In reminders, add one entry for each upcoming event or deadline this message "
                     "mentions that the person would want a heads-up about; otherwise leave it empty. Each entry has the "
                     "short message to send them, and remind_at, an ISO date and time in the future: a few hours before "
                     "for something today, otherwise the evening before.")
        return "\n\n".join(parts), fresh

    def respond(self, text: str) -> str:
        recalled = self.associations.recall(text, MEMORIES_PER_REPLY)
        with self.state.lock:
            prompt, fresh = self._prompt(text, recalled)
            self.state.main_activity = {"phase": "replying", "text": text, "ids": [n.id for n in recalled]}
        try:
            d, _ = self.llm.ask_json(prompt, REPLY, 2000)
        finally:
            self.state.main_activity = {**self.state.main_activity, "phase": "idle"}
        reply = d.get("reply") if d else None
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("The main model returned no reply.")
        with self.state.lock:
            self.state.add_turn("user", text)
            self.state.add_turn("assistant", reply)
            self._remember(text, d.get("unresolved") is True)  # the person's own words, never summarised or rewritten
            for n in fresh:
                n.meta["seen"] = True
            self.associations.learn_from_use(d.get("used_ids") or [])
            self.state.main_activity = {**self.state.main_activity, "used": d.get("used_ids") or []}
            schedule_reminders(self.state, d.get("reminders") or [])
            self.mem.decay_and_prune(DECAY_PER_REPLY, PRUNE_FLOOR)
            self.state.save()
        return reply
