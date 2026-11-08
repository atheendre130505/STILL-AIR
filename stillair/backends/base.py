"""The interface every LLM backend implements, and the values it returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional



@dataclass
class Thought:
    kind: str  # derive | analogy | restate | none
    text: str
    claim: Optional[tuple] = None
    parents: list = field(default_factory=list)  # node ids
    tokens: int = 0


@dataclass
class Answer:
    text: Optional[str] = None
    need: Optional[str] = None  # entity to look up next (a second hop)
    via: list = field(default_factory=list)
    tokens: int = 0


class LLM:
    name = "base"
    usage_includes_input = False  # True when reported tokens already count the memories the model read

    def extract(self, text: str) -> Optional[tuple]:
        raise NotImplementedError

    def associate(self, seeds: list, nudge=None) -> Thought:
        raise NotImplementedError

    def followups(self, seeds: list) -> list:
        """Memory lookups the model wants before thinking: 'what else do I know about X?'"""
        return []

    def judge(self, thought: Thought, parents: list):
        """-> (validity in [0,1], tokens used)"""
        raise NotImplementedError

    def answer(self, question: str, context: list, scratch: dict) -> Answer:
        raise NotImplementedError
