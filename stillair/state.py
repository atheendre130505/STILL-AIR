"""What the conscious and subconscious share: memory, the chats, and what has been tried."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

from .memory import Memory

MAX_TURNS, MAX_THOUGHTS = 40, 60
TITLE_LENGTH = 40


def _title(turns: list) -> str:
    first = next((t["text"] for t in turns if t["role"] == "user"), "")
    return first if len(first) <= TITLE_LENGTH else first[:TITLE_LENGTH].rstrip() + "…"


class SharedState:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.lock = threading.RLock()
        self.mem = Memory()
        self.chats: list = [{"id": 1, "title": "", "turns": []}]  # memory is shared across all of them
        self.current = 1
        self.thoughts: list = []
        self.undigested: list = []  # exchanges the subconscious has not read yet
        self.escalations: list = []  # conclusions waiting for the large model's check
        self.sub_activity: dict = {"phase": "idle"}  # what the backend view shows; not saved
        self.main_activity: dict = {"phase": "idle", "ids": []}
        self.small_tokens = 0  # this run only
        if path and os.path.isfile(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.mem = Memory.from_json(d["memory"])
        old_turns = d.get("turns", [])  # files saved before chats existed held a single conversation
        self.chats = d.get("chats") or [{"id": 1, "title": _title(old_turns), "turns": old_turns}]
        self.current = d.get("current", self.chats[-1]["id"])
        self.thoughts = d["thoughts"]
        self.undigested = d.get("undigested", [])
        self.escalations = d.get("escalations", [])

    def save(self) -> None:
        if not self.path:
            return
        data = {"memory": self.mem.to_json(), "chats": self.chats, "current": self.current, "thoughts": self.thoughts,
                "undigested": self.undigested, "escalations": self.escalations}
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def open_problems(self) -> list:
        return [n for n in self.mem.nodes.values() if n.kind == "problem" and n.meta.get("status") == "open"]

    # ---- chats -------------------------------------------------------------
    def _chat(self) -> dict:
        return next(c for c in self.chats if c["id"] == self.current)

    @property
    def turns(self) -> list:
        return self._chat()["turns"]

    def add_turn(self, role: str, text: str, unprompted: bool = False) -> None:
        chat = self._chat()
        chat["turns"] = (chat["turns"] + [{"role": role, "text": text, "unprompted": unprompted}])[-MAX_TURNS:]
        chat["title"] = chat["title"] or _title(chat["turns"])

    def new_chat(self) -> None:
        """Start a blank conversation. An already blank one is reused, so they cannot pile up."""
        blank = next((c for c in self.chats if not c["turns"]), None)
        if blank is None:
            blank = {"id": max(c["id"] for c in self.chats) + 1, "title": "", "turns": []}
            self.chats.append(blank)
        self.current = blank["id"]
        self.save()

    def open_chat(self, chat_id: int) -> bool:
        if not any(c["id"] == chat_id for c in self.chats):
            return False
        self.current = chat_id
        self.save()
        return True

    def clear_chat(self) -> None:
        """Empty the open conversation. Long-term memory is kept."""
        chat = self._chat()
        chat["turns"], chat["title"] = [], ""
        self.save()

    def clear_memory(self) -> None:
        """Forget all memories, conclusions and problems. Chat transcripts are kept."""
        self.mem.clear()
        self.undigested = []
        self.escalations = []
        self.thoughts = []
        self.sub_activity = {"phase": "idle"}
        self.main_activity = {"phase": "idle", "ids": []}
        self.save()

    def log_thought(self, a: str, b: str, link: str, outcome: str, tokens: int) -> None:
        self.small_tokens += tokens
        entry = {"at": time.time(), "a": a, "b": b, "link": link, "outcome": outcome, "tokens": tokens}
        self.thoughts = (self.thoughts + [entry])[-MAX_THOUGHTS:]
