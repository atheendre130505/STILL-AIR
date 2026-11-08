"""The conscious mind: the large model, which does almost no work.

It answers when you speak to it, reading what the subconscious has already prepared.
Between messages it is called only when the subconscious believes it has solved a problem
or noticed something urgent, to check that before telling you.
"""

from __future__ import annotations

from datetime import datetime

from .links import Links
from .model import Model
from .state import SharedState

REPLY = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "used_ids": {"type": "array", "items": {"type": "integer"}},
        "wrong_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["reply", "used_ids", "wrong_ids"],
    "additionalProperties": False,
}
VERIFY = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "message": {"type": "string"}},
    "required": ["correct", "message"],
    "additionalProperties": False,
}
CATCH_UP = {
    "type": "object",
    "properties": {"wrong_ids": {"type": "array", "items": {"type": "integer"}}, "message": {"type": "string"}},
    "required": ["wrong_ids", "message"],
    "additionalProperties": False,
}
SYSTEM = "You are a helpful assistant with a long-term memory of this person. Be direct and natural. Reply with JSON only."
RECENT_TURNS = 8
MAX_NOTES_FOR_CATCH_UP = 30
NOTED = "The large model is off. The small model will think about this."
FADE_PER_REPLY, DROP_BELOW = 0.05, 0.2  # a conclusion left unused for about 16 replies is dropped
MAX_CONNECTIONS_KEPT = 40  # memory does not keep everything: past this, the weakest go
MAX_UNANSWERED = 2  # after this many messages you have not answered, it stops speaking first


def current_time() -> str:
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


def unanswered_messages(state: SharedState) -> int:
    count = 0
    for turn in reversed(state.turns):
        if not turn.get("unprompted"):
            break
        count += 1
    return count


class Conscious:
    def __init__(self, llm: Model, state: SharedState, links: Links):
        self.llm, self.state, self.links, self.mem = llm, state, links, state.mem

    def _prompt(self, text: str, recalled: list) -> str:
        notes = [n for n in recalled if n.kind == "fact"]
        worked_out = [n for n in recalled if n.kind == "connection"]
        parts = []
        if notes:
            parts.append("What you know about this person:\n" + "\n".join(f"[{n.id}] {n.text}" for n in notes))
        if worked_out:
            parts.append("Things your subconscious worked out while idle. They may be wrong. Use one if it truly helps, "
                         "and list in wrong_ids any that are wrong:\n" + "\n".join(f"[{n.id}] {n.text}" for n in worked_out))
        recent = "\n".join(f"{t['role']}: {t['text']}" for t in self.state.turns[-RECENT_TURNS:])
        if recent:
            parts.append("Recent conversation:\n" + recent)
        parts.append(f"It is now {current_time()}.\n\nuser: {text}\n\nWrite your reply. List in used_ids the ids of anything above "
                     "you relied on.")
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
            shown = {n.id for n in recalled}
            for wrong in d.get("wrong_ids") or []:
                if wrong in shown and self.mem.nodes[wrong].kind == "connection":
                    self.mem.remove(wrong)
            self.links.learn_from_use(d.get("used_ids") or [])
            self.state.main_activity = {**self.state.main_activity, "used": d.get("used_ids") or []}
            self.mem.fade(FADE_PER_REPLY, DROP_BELOW)
            self.mem.keep_strongest(MAX_CONNECTIONS_KEPT)
            self.state.save()
        return reply

    def note_only(self, text: str) -> None:
        """With the large model switched off, a message goes straight to the subconscious."""
        with self.state.lock:
            self.state.add_turn("user", text)
            self.state.add_turn("assistant", NOTED)
            self.state.undigested.append({"user": text, "assistant": "(no reply)", "direct": True})
            self.state.save()

    def catch_up(self, missed: list) -> None:
        """Back from being switched off: check every unchecked conclusion and answer what was asked meanwhile."""
        with self.state.lock:
            unchecked = [n for n in self.mem.connections() if not n.meta.get("confirmed")]
            if not unchecked and not missed:
                return
            notes = [n for n in self.mem.readable() if n.kind == "fact"][-MAX_NOTES_FOR_CATCH_UP:]
            def line(c):
                about = self.mem.nodes.get(c.meta.get("about"))
                return f"[{c.id}] {c.text}" + (f" (working on: {about.text})" if about else "")
            prompt = (f"It is now {current_time()}. You were switched off for a while, and a small model worked alone.\n\n"
                      "What you know about the person:\n" + ("\n".join(f"- {n.text}" for n in notes) or "(nothing)") + "\n\n"
                      "What the person said while you were off:\n" + ("\n".join(f"- {t}" for t in missed) or "(nothing)") + "\n\n"
                      "Conclusions the small model reached, none of them checked:\n" + ("\n".join(line(c) for c in unchecked) or "(none)") + "\n\n"
                      "List in wrong_ids every conclusion that is incorrect, vague, or adds something nobody said. Then write message: a short "
                      "note to the person in the first person, answering what they asked while you were off and telling them what "
                      "held up and what did not. Never mention ids.")
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
                about = self.mem.nodes.get(c.meta.get("about"))
                is_problem = about is not None and about.kind == "problem"
                if c.id in wrong:
                    self.mem.remove(c.id)
                    if is_problem:
                        about.meta["status"] = "open"
                        about.meta["tried"] = (about.meta.get("tried", []) + [f"Concluded wrongly: {c.text}"])[-5:]
                else:
                    c.meta["confirmed"] = True
                    if is_problem:
                        about.meta["status"] = "solved"
            self.state.escalations = []
            message = str(d.get("message") or "").strip()
            if message:
                self.state.add_turn("assistant", message, unprompted=True)
            self.state.save()

    def verify_next(self) -> None:
        """Check one thing the subconscious wants to raise, and tell the person if it holds up."""
        with self.state.lock:
            insight = None
            while self.state.escalations and insight is None:
                insight = self.mem.nodes.get(self.state.escalations.pop(0))
            if insight is None:
                return
            about = self.mem.nodes.get(insight.meta.get("about"))
            sources = [self.mem.nodes[p] for p in insight.parents if p in self.mem.nodes]
            earlier = "".join(f"\n- {t}" for t in (about.meta.get("archived", []) + about.meta.get("tried", [])) ) if about else ""
            earlier = f"\n\nEarlier attempts that failed:{earlier}" if earlier else ""
            prompt = (f"It is now {current_time()}. The person is away. Your subconscious worked on this:\n"
                      f"{about.text if about else '(no specific problem)'}\n\nIts reasoning:\n" + "\n".join(f"- {s}" for s in insight.meta.get("chain", [])) +
                      "\n\nWhat it relied on:\n" + ("\n".join(f"- {s.text}" for s in sources) or "(nothing)") + earlier + f"\n\nIts conclusion: {insight.text}\n\n"
                      "Is the conclusion actually correct and supported? If so, set correct to true and write a short message to them in "
                      "the first person, as something that just occurred to you, for example 'I was thinking about your problem and...'. "
                      "Never mention a subconscious. If it is wrong, set correct to false and leave message empty.")
            self.state.main_activity = {"phase": "reviewing", "text": "Checking what the subconscious concluded", "ids": [insight.id]}
        try:
            d, _ = self.llm.ask_json(prompt, VERIFY, 1000)
        finally:
            self.state.main_activity = {**self.state.main_activity, "phase": "idle"}
        if not d:
            return
        with self.state.lock:
            if d.get("correct") is True:
                insight.meta["confirmed"] = True
                if about is not None and about.kind == "problem":
                    about.meta["status"] = "solved"
                message = str(d.get("message") or "").strip()
                if message and unanswered_messages(self.state) < MAX_UNANSWERED:
                    self.state.add_turn("assistant", message, unprompted=True)
            else:
                self.mem.remove(insight.id)
                if about is not None and about.kind == "problem":
                    about.meta["status"] = "open"
                    about.meta["tried"] = (about.meta.get("tried", []) + [f"Concluded wrongly: {insight.text}"])[-5:]
            self.state.save()
