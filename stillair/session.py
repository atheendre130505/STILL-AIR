"""Runs the two minds together: the conscious answers, the subconscious fills the silence."""

from __future__ import annotations

import threading
import time

from .conscious import MAX_UNANSWERED, NOTED, Conscious, unanswered_messages
from .links import Links
from .state import SharedState
from .subconscious import Subconscious

SETTLE_SECONDS = 3.0  # give the person a moment after a reply before the subconscious starts
ERROR_PAUSE_SECONDS = 30.0
VERIFY_EVERY_SECONDS = 120.0  # the large model is called rarely: at most this often while idle
VERIFY_QUIET_SECONDS = 15.0  # never right after a reply
CLOCK_TICK_SECONDS = 5.0
LIMIT_CHOICES = (None, 5, 15, 60)  # minutes of thinking after each message; None = until the next message


class ChatSession:
    def __init__(self, main_llm, small_llm, embedder=None, path=None, autostart=True, outside=None):
        self.state = SharedState(path)
        self.links = Links(self.state, embedder)
        self.conscious = Conscious(main_llm, self.state, self.links)
        self.subconscious = Subconscious(small_llm, self.state, self.links, outside=outside)
        self.main_llm, self.small_llm = main_llm, small_llm
        self.conscious_on = True
        self.checking = False
        self._missed: list = []  # what was said while the large model was off
        self.limit_minutes = None
        self.responding = False
        self.status = "waiting"
        self.error = None
        self._quiet_since = time.monotonic()
        self._retry_at = 0.0
        self._last_verify = 0.0
        self._stop = threading.Event()
        if autostart:
            threading.Thread(target=self._idle_loop, daemon=True).start()
            threading.Thread(target=self._clock_loop, daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def _quiet_for(self) -> float:
        return time.monotonic() - self._quiet_since

    def _time_up(self) -> bool:
        return self.limit_minutes is not None and self._quiet_for() >= self.limit_minutes * 60 + SETTLE_SECONDS

    def _wait_status(self) -> str:
        return "time limit reached; your next message starts it again" if self._time_up() else "waiting for you to finish"

    def _ready_to_think(self) -> bool:
        return (not self.responding and self._quiet_for() >= SETTLE_SECONDS
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

    def _may_verify(self) -> bool:
        return (not self.responding and self._quiet_for() >= VERIFY_QUIET_SECONDS and bool(self.state.escalations)
                and time.monotonic() - self._last_verify >= VERIFY_EVERY_SECONDS and unanswered_messages(self.state) < MAX_UNANSWERED)

    def post_unchecked(self) -> None:
        """With the large model off, conclusions go straight to the chat, marked as unchecked."""
        with self.state.lock:
            while self.state.escalations:
                insight = self.state.mem.nodes.get(self.state.escalations.pop(0))
                if insight is None:
                    continue
                about = self.state.mem.nodes.get(insight.meta.get("about"))
                if about is not None and about.kind == "problem":
                    about.meta["status"] = "solved"
                self.state.add_turn("assistant", f"Worked out by the small model alone, not checked: {insight.text}", unprompted=True)
            self.state.save()

    def _clock_loop(self) -> None:
        while not self._stop.wait(CLOCK_TICK_SECONDS):
            if not self.conscious_on:
                self.post_unchecked()
                continue
            if not self._may_verify():
                continue
            self._last_verify = time.monotonic()
            try:
                self.conscious.verify_next()
            except Exception as e:
                self.error = f"The main model failed while checking: {e}"

    # ---- controls -----------------------------------------------------------
    def _catch_up(self) -> None:
        self.checking = True
        try:
            missed, self._missed = self._missed, []
            self.conscious.catch_up(missed)
            self.error = None
        except Exception as e:
            self.error = f"The main model failed while checking: {e}"
        finally:
            self.checking = False

    def configure(self, limit_minutes=..., conscious=None) -> None:
        if conscious is True and not self.conscious_on:
            self.conscious_on = True
            threading.Thread(target=self._catch_up, daemon=True).start()
        elif conscious is not None:
            self.conscious_on = conscious
        if limit_minutes is not ...:
            self.limit_minutes = limit_minutes

    def _unless_replying(self, action) -> bool:
        """-> False while a reply is in flight, since it belongs to the chat it was asked in."""
        if self.responding:
            return False
        with self.state.lock:
            action()
        return True

    def new_chat(self) -> bool:
        return self._unless_replying(self.state.new_chat)

    def clear_chat(self) -> bool:
        return self._unless_replying(self.state.clear_chat)

    def open_chat(self, chat_id: int) -> bool:
        if self.responding:
            return False
        with self.state.lock:
            return self.state.open_chat(chat_id)

    def clear_memory(self) -> bool:
        def forget():
            self.state.clear_memory()
            self.links.forget()
            self.subconscious.chain = None
        return self._unless_replying(forget)

    def say(self, text: str) -> str:
        self.responding = True
        try:
            if not self.conscious_on:
                self.conscious.note_only(text)
                self._missed.append(text)
                return NOTED
            reply = self.conscious.respond(text)
            self.error = None
            return reply
        except Exception as e:
            self.error = f"The main model failed: {e}"
            raise
        finally:
            self.responding = False
            self._quiet_since = time.monotonic()

    # ---- what the pages draw --------------------------------------------------
    def backend_snapshot(self) -> dict:
        s = self.state
        links = self.links.strongest_links()
        with s.lock:
            nodes = [{"id": n.id, "kind": n.kind, "text": n.text, "strength": round(n.strength, 2), "parents": n.parents,
                      "status": n.meta.get("status"), "visits": n.meta.get("visits", 0), "used": n.hits,
                      "confirmed": bool(n.meta.get("confirmed"))} for n in s.mem.readable()]
            return {
                "nodes": nodes, "links": links, "sub": s.sub_activity, "main": s.main_activity,
                "thoughts": s.thoughts[-30:], "conscious": self.conscious_on, "checking": self.checking, "responding": self.responding, "status": self.status,
                "open_problems": len(s.open_problems()), "chains": self.subconscious.chains_started,
                "small_tokens": s.small_tokens, "main_tokens": self.main_llm.tokens_used, "main_calls": self.main_llm.calls,
                "main_model": self.main_llm.name, "small_model": self.small_llm.name, "meaning_search": self.links.meaning_working,
                "error": self.error,
            }

    def snapshot(self) -> dict:
        s = self.state
        with s.lock:
            nodes = s.mem.readable()
            left = None
            if self.limit_minutes is not None:
                left = max(0, round((self.limit_minutes * 60 + SETTLE_SECONDS - self._quiet_for()) / 60))
            return {
                "turns": s.turns,
                "chats": [{"id": c["id"], "title": c["title"] or "New chat"} for c in reversed(s.chats)],
                "chat_id": s.current,
                "thoughts": s.thoughts[-30:],
                "hunches": [{"id": n.id, "text": n.text, "strength": round(n.strength, 2), "used": n.hits, "confirmed": bool(n.meta.get("confirmed"))}
                            for n in s.mem.connections()],
                "memories": sum(n.kind == "fact" for n in nodes),
                "problems": [{"id": n.id, "text": n.text, "attempts": n.meta.get("attempts", 0), "resting": n.meta.get("rest_until", 0.0) > time.time()} for n in s.open_problems()],
                "meaning_search": self.links.meaning_working,
                "conscious": self.conscious_on,
                "checking": self.checking,
                "limit_minutes": self.limit_minutes,
                "minutes_left": left,
                "responding": self.responding,
                "status": self.status,
                "main_model": self.main_llm.name,
                "small_model": self.small_llm.name,
                "error": self.error,
            }
