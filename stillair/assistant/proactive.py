"""Speaking first: scheduled reminders, and bringing up something the subconscious noticed.

Reminders need no model when they fire. Anything from the subconscious is reviewed by the main
model against what the person actually said before it is mentioned, and hunches the review
finds unsupported are deleted. The review only runs when there is a new hunch, at most every
few minutes, and stops after two unanswered messages, so an idle night cannot drain the quota.
"""

from __future__ import annotations

from datetime import datetime

from .state import SharedState

SPEAK = {
    "type": "object",
    "properties": {
        "speak": {"type": "boolean"},
        "message": {"type": "string"},
        "wrong_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["speak", "message", "wrong_ids"],
    "additionalProperties": False,
}
MAX_HUNCHES_REVIEWED = 6
MAX_UNANSWERED = 2
RECENT_TURNS = 6


def current_time() -> str:
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


def schedule_reminders(state: SharedState, items: list, now_epoch: float = None) -> None:
    """Store reminders the main model proposed, skipping unreadable, past or duplicate ones."""
    now_epoch = datetime.now().timestamp() if now_epoch is None else now_epoch
    for item in items:
        try:
            at = datetime.fromisoformat(item["remind_at"]).timestamp()
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        duplicate = any(r["text"].lower() == text.lower() and not r["fired"] for r in state.reminders)
        if text and at > now_epoch and not duplicate:
            state.reminders.append({"text": text, "at": at, "fired": False})


def fire_due_reminders(state: SharedState, now_epoch: float = None) -> int:
    """Post every reminder whose time has come, including any missed while the app was closed."""
    now_epoch = datetime.now().timestamp() if now_epoch is None else now_epoch
    fired = 0
    with state.lock:
        for reminder in state.reminders:
            if not reminder["fired"] and reminder["at"] <= now_epoch:
                reminder["fired"] = True
                state.add_turn("assistant", reminder["text"], unprompted=True)
                fired += 1
        if fired:
            state.save()
    return fired


def unanswered_messages(state: SharedState) -> int:
    count = 0
    for turn in reversed(state.turns):
        if not turn.get("unprompted"):
            break
        count += 1
    return count


class Voice:
    """Decides, with the main model, whether something the subconscious found is worth saying."""

    def __init__(self, llm, state: SharedState):
        self.llm, self.state, self.mem = llm, state, state.mem

    def _unreviewed(self) -> list:
        unseen = [n for n in self.mem.nodes.values() if n.kind == "connection" and not n.meta.get("seen")]
        return sorted(unseen, key=lambda n: -n.id)[:MAX_HUNCHES_REVIEWED]

    def should_review(self) -> bool:
        with self.state.lock:
            return bool(self._unreviewed()) and unanswered_messages(self.state) < MAX_UNANSWERED

    def _prompt(self, hunches: list) -> str:
        sources = {p: self.mem.nodes[p] for h in hunches for p in h.parents if p in self.mem.nodes}
        recent = "\n".join(f"{t['role']}: {t['text']}" for t in self.state.turns[-RECENT_TURNS:])
        return (
            f"It is now {current_time()}. The person is away and has not asked you anything.\n\n"
            "What they told you:\n" + "\n".join(f"[{n.id}] {n.text}" for n in sources.values()) + "\n\n"
            "Hunches from your subconscious, which may be wrong:\n" + "\n".join(f"[{h.id}] {h.text}" for h in hunches) + "\n\n"
            f"Recent conversation:\n{recent}\n\n"
            "List in wrong_ids every hunch that adds a detail they did not actually say. Then decide whether anything "
            "supported is worth bringing up now without being asked, such as a clash, something time-sensitive or a "
            "genuinely useful connection. Do not repeat what the conversation already covered. If so, set speak to true "
            "and write a short, natural message to them. Otherwise set speak to false and leave message empty."
        )

    def review(self) -> None:
        with self.state.lock:
            hunches = self._unreviewed()
            if not hunches:
                return
            prompt = self._prompt(hunches)
            self.state.main_activity = {"phase": "reviewing", "text": "Checking new hunches against what you said", "ids": [h.id for h in hunches]}
        try:
            d, _ = self.llm.ask_json(prompt, SPEAK, 1000)
        finally:
            self.state.main_activity = {**self.state.main_activity, "phase": "idle"}
        if not d:
            return
        with self.state.lock:
            for h in hunches:
                h.meta["seen"] = True
            for wrong in d.get("wrong_ids") or []:
                if wrong in {h.id for h in hunches}:
                    self.mem.remove(wrong)
            message = str(d.get("message") or "").strip()
            if d.get("speak") is True and message:
                self.state.add_turn("assistant", message, unprompted=True)
            self.state.save()
