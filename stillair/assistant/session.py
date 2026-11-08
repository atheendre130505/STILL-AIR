"""Runs the two minds together: the conscious answers, the subconscious fills the silence, and either can speak first."""

from __future__ import annotations

import threading
import time

from .associations import Associations
from .conscious import Conscious
from .proactive import Voice, fire_due_reminders
from .state import SharedState
from .subconscious import Subconscious

SETTLE_SECONDS = 3.0  # give the person a moment after a reply before the subconscious starts
ERROR_PAUSE_SECONDS = 30.0
REVIEW_EVERY_SECONDS = 180.0  # how often the main model may review new hunches
REVIEW_QUIET_SECONDS = 30.0  # do not speak up right after a reply
VOICE_TICK_SECONDS = 5.0
LIMIT_CHOICES = (None, 5, 15, 60)  # minutes of thinking after each message; None = until the next message


class ChatSession:
    def __init__(self, main_llm, small_llm, embedder=None, path=None):
        self.state = SharedState(path)
        self.associations = Associations(self.state, embedder)
        self.conscious = Conscious(main_llm, self.state, self.associations)
        self.subconscious = Subconscious(small_llm, self.state, self.associations)
        self.voice = Voice(main_llm, self.state)
        self.main_llm, self.small_llm = main_llm, small_llm
        self.enabled = True
        self.limit_minutes = None
        self.responding = False
        self.status = "waiting"
        self.error = None
        self._quiet_since = time.monotonic()
        self._retry_at = 0.0
        self._last_review = 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        threading.Thread(target=self._voice_loop, daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def _quiet_for(self) -> float:
        return time.monotonic() - self._quiet_since

    def _time_up(self) -> bool:
        return self.limit_minutes is not None and self._quiet_for() >= self.limit_minutes * 60 + SETTLE_SECONDS

    def _wait_status(self) -> str:
        if not self.enabled:
            return "off"
        if self._time_up():
            return "time limit reached; your next message starts it again"
        return "waiting for you to finish"

    def _ready_to_think(self) -> bool:
        return (self.enabled and not self.responding and self._quiet_for() >= SETTLE_SECONDS
                and time.monotonic() >= self._retry_at and not self._time_up())

    def _idle_loop(self) -> None:
        while not self._stop.is_set():
            if not self._ready_to_think():
                self.status = self._wait_status()
                self._stop.wait(0.3)
                continue
            self.status = "thinking"
            try:
                found_work = self.subconscious.think_once()
            except Exception as e:
                self.error = f"The small model failed: {e}"
                self._retry_at = time.monotonic() + ERROR_PAUSE_SECONDS
                continue
            self.error = None
            if not found_work:
                self.status = "waiting for more memories"
                self._stop.wait(1.0)

    def _may_review(self) -> bool:
        return (not self.responding and self._quiet_for() >= REVIEW_QUIET_SECONDS
                and time.monotonic() - self._last_review >= REVIEW_EVERY_SECONDS and self.voice.should_review())

    def _voice_loop(self) -> None:
        while not self._stop.wait(VOICE_TICK_SECONDS):
            fire_due_reminders(self.state)
            if not self._may_review():
                continue
            self._last_review = time.monotonic()
            try:
                self.voice.review()
            except Exception as e:
                self.error = f"The main model failed while reviewing: {e}"

    def configure(self, on=None, limit_minutes=...) -> None:
        if on is not None:
            self.enabled = on
        if limit_minutes is not ...:
            self.limit_minutes = limit_minutes

    def new_chat(self) -> bool:
        """-> False while a reply is in flight, since it belongs to the chat it was asked in."""
        if self.responding:
            return False
        with self.state.lock:
            self.state.new_chat()
        return True

    def clear_chat(self) -> bool:
        if self.responding:
            return False
        with self.state.lock:
            self.state.clear_chat()
        return True

    def open_chat(self, chat_id: int) -> bool:
        if self.responding:
            return False
        with self.state.lock:
            return self.state.open_chat(chat_id)

    def say(self, text: str) -> str:
        self.responding = True
        try:
            reply = self.conscious.respond(text)
            self.error = None
            return reply
        except Exception as e:
            self.error = f"The main model failed: {e}"
            raise
        finally:
            self.responding = False
            self._quiet_since = time.monotonic()

    def backend_snapshot(self) -> dict:
        """Everything the backend view draws."""
        s = self.state
        links = self.associations.strongest_links()
        with s.lock:
            nodes = [{"id": n.id, "kind": n.kind, "text": n.text, "strength": round(n.strength, 2), "parents": n.parents,
                      "open": bool(n.meta.get("open")), "visits": n.meta.get("visits", 0), "used": n.hits}
                     for n in s.mem.readable()]
            return {
                "nodes": nodes, "links": links, "sub": s.sub_activity, "main": s.main_activity,
                "thoughts": s.thoughts[-30:], "enabled": self.enabled, "responding": self.responding, "status": self.status,
                "pairs_tried": len(s.tried), "paths_learned": len(s.learned), "round": self.subconscious.rounds + 1,
                "small_tokens": s.small_tokens, "main_tokens": self.main_llm.tokens_used, "main_calls": self.main_llm.calls,
                "main_model": self.main_llm.model, "small_model": self.small_llm.model, "meaning_search": self.associations.meaning_working,
                "error": self.error,
            }

    def snapshot(self) -> dict:
        s = self.state
        with s.lock:
            nodes = list(s.mem.nodes.values())
            left = None
            if self.limit_minutes is not None:
                left = max(0, round((self.limit_minutes * 60 + SETTLE_SECONDS - self._quiet_for()) / 60))
            return {
                "turns": s.turns,
                "chats": [{"id": c["id"], "title": c["title"] or "New chat"} for c in reversed(s.chats)],
                "chat_id": s.current,
                "thoughts": s.thoughts[-30:],
                "hunches": [{"id": n.id, "text": n.text, "strength": round(n.strength, 2), "used": n.hits}
                            for n in nodes if n.kind == "connection"],
                "memories": sum(n.kind == "fact" for n in nodes),
                "pairs_tried": len(s.tried),
                "paths_learned": len(s.learned),
                "meaning_search": self.associations.meaning_working,
                "reminders": sorted((r for r in s.reminders if not r["fired"]), key=lambda r: r["at"]),
                "enabled": self.enabled,
                "limit_minutes": self.limit_minutes,
                "minutes_left": left,
                "responding": self.responding,
                "status": self.status,
                "main_model": self.main_llm.model,
                "small_model": self.small_llm.model,
                "error": self.error,
            }
