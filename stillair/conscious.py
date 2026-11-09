"""The conscious mind: the large model, which does very little.

It answers when you speak to it, reading what the subconscious has already prepared, and in
that same call it notes anything worth a reminder. It is never called while you are away.
"""

from __future__ import annotations

from datetime import datetime

from .links import Links
from .model import Model
from .reminders import schedule_reminders
from .state import SharedState

REMINDERS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "remind_at": {"type": "string"}},
        "required": ["text", "remind_at"],
        "additionalProperties": False,
    },
}
REPLY = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "used_ids": {"type": "array", "items": {"type": "integer"}},
        "wrong_ids": {"type": "array", "items": {"type": "integer"}},
        "reminders": REMINDERS,
    },
    "required": ["reply", "used_ids", "wrong_ids", "reminders"],
    "additionalProperties": False,
}
CATCH_UP = {
    "type": "object",
    "properties": {"wrong_ids": {"type": "array", "items": {"type": "integer"}}, "message": {"type": "string"}, "reminders": REMINDERS},
    "required": ["wrong_ids", "message", "reminders"],
    "additionalProperties": False,
}
SYSTEM = "You are a helpful assistant with a long-term memory of this person. Be direct and natural. Reply with JSON only."
RECENT_TURNS = 8
MAX_NOTES_FOR_CATCH_UP = 30
FADE_PER_REPLY, DROP_BELOW = 0.05, 0.2  # a connection left unused for about 16 replies is dropped
REMINDER_RULES = ("In reminders, add one entry for each upcoming event, appointment or deadline the person mentions that they would want "
                  "a heads-up about, unless it is already set; otherwise leave it empty. Each entry has text, the short message to send them "
                  "including what and when, and remind_at, a local ISO date and time in the future with no timezone. If they asked for a "
                  "specific time, use exactly that. Otherwise use the evening before, or a few hours before for something today, or "
                  "halfway there for something under an hour away.")


def current_time() -> str:
    return datetime.now().strftime("%A %d %B %Y, %H:%M:%S")  # seconds, so "in 3 minutes" is not rounded early


class Conscious:
    def __init__(self, llm: Model, state: SharedState, links: Links):
        self.llm, self.state, self.links, self.mem = llm, state, links, state.mem

    def _already_set(self) -> str:
        waiting = [r["text"] for r in self.state.reminders if not r["fired"]]
        return "Reminders already set:\n" + "\n".join(f"- {t}" for t in waiting) + "\n\n" if waiting else ""

    def _prompt(self, text: str, recalled: list) -> str:
        notes = [n for n in recalled if n.kind == "fact"]
        connections = [n for n in recalled if n.kind == "connection"]
        parts = []
        if notes:
            parts.append("What you know about this person:\n" + "\n".join(f"[{n.id}] {n.text}" for n in notes))
        if connections:
            parts.append("Connections your subconscious made while idle. They may be wrong. Use one if it truly helps, "
                         "and list in wrong_ids any that are wrong:\n" + "\n".join(f"[{n.id}] {n.text}" for n in connections))
        recent = "\n".join(f"{t['role']}: {t['text']}" for t in self.state.turns[-RECENT_TURNS:])
        if recent:
            parts.append("Recent conversation:\n" + recent)
        parts.append(f"It is now {current_time()}.\n\n{self._already_set()}user: {text}\n\nWrite your reply. List in used_ids the ids of "
                     f"anything above you relied on. {REMINDER_RULES}")
        return "\n\n".join(parts)

    def respond(self, text: str) -> str:
        self.state.main_activity = {"phase": "searching", "text": text, "ids": []}
        found = self.links.search(text)
        recalled = self.links.most_relevant(found)
        with self.state.lock:
            prompt = self._prompt(text, recalled)
            self.state.main_activity = {"phase": "replying", "text": text, "ids": [n.id for n in recalled], "searched": len(found)}
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
            self.state.undigested.append({"user": text, "assistant": reply})
            schedule_reminders(self.state, d.get("reminders") or [])
            shown = {n.id for n in recalled}
            for wrong in d.get("wrong_ids") or []:
                if wrong in shown and self.mem.nodes[wrong].kind == "connection":
                    self.mem.remove(wrong)
            self.links.learn_from_use(d.get("used_ids") or [])
            self.state.main_activity = {**self.state.main_activity, "used": d.get("used_ids") or []}
            self.mem.fade(FADE_PER_REPLY, DROP_BELOW)
            self.state.save()
        return reply

    def note_only(self, text: str) -> str:
        """With the large model off, the message is simply collected for the small model to digest."""
        with self.state.lock:
            reply = "The large model is off, so I've just noted that."
            self.state.add_turn("user", text)
            self.state.add_turn("assistant", reply)
            self.state.undigested.append({"user": text, "assistant": ""})
            self.state.save()
        return reply

    def catch_up(self, missed: list) -> None:
        """Back from being switched off: check every unchecked connection and answer what was said meanwhile."""
        with self.state.lock:
            unchecked = [n for n in self.mem.connections() if not n.meta.get("confirmed")]
            if not unchecked and not missed:
                return
            notes = [n for n in self.mem.readable() if n.kind == "fact"][-MAX_NOTES_FOR_CATCH_UP:]
            prompt = (f"It is now {current_time()}. You were switched off for a while, and a small model worked alone.\n\n"
                      "What you know about the person:\n" + ("\n".join(f"- {n.text}" for n in notes) or "(nothing)") + "\n\n"
                      "What the person said while you were off:\n" + ("\n".join(f"- {t}" for t in missed) or "(nothing)") + "\n\n"
                      "Connections the small model made, none of them checked:\n" + ("\n".join(f"[{c.id}] {c.text}" for c in unchecked) or "(none)") + "\n\n"
                      f"{self._already_set()}List in wrong_ids every connection that is incorrect, vague, or adds something nobody said. Then write message: a short "
                      "note to the person in the first person, answering what they said while you were off. Never mention ids. "
                      f"{REMINDER_RULES}")
            self.state.main_activity = {"phase": "reviewing", "text": "Checking everything that happened while off", "ids": [c.id for c in unchecked]}
        try:
            d, _ = self.llm.ask_json(prompt, CATCH_UP, 1500)
        finally:
            self.state.main_activity = {**self.state.main_activity, "phase": "idle"}
        if not d:
            raise RuntimeError("The main model returned nothing while catching up.")
        with self.state.lock:
            wrong = set(d.get("wrong_ids") or [])
            for c in unchecked:
                if c.id in wrong:
                    self.mem.remove(c.id)
                else:
                    c.meta["confirmed"] = True
            schedule_reminders(self.state, d.get("reminders") or [])
            message = str(d.get("message") or "").strip()
            if message:
                self.state.add_turn("assistant", message, unprompted=True)
            self.state.save()
