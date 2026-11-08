"""Selector + grounding (analogue: salience / sleep-like consolidation, and executive control).

Every candidate thought gets two scores:

    validity : grounding. Is the claim supported by the memories it cites? (LLM judge)
    novelty  : is it new? A claim already in memory scores 0; text that is nearly a copy
               of something stored (or of a very recent thought) scores low.

and one of three outcomes:

    keep      valid and novel          -> written to memory as a new connection
    reinforce valid but already known  -> existing memory strengthened, nothing new written
    discard   not valid, or a repeat   -> dropped (the "rest fade")
              with nothing to strengthen

`enabled=False` is condition B / the no-selector ablation: everything is accepted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..backends.base import LLM, Thought
from .memory import Memory, Node


@dataclass
class Decision:
    action: str  # keep | reinforce | discard
    validity: float
    novelty: float
    reason: str
    judge_tokens: int = 0
    existing: Optional[Node] = None


class Selector:
    def __init__(self, llm: LLM, enabled: bool = True, validity_min: float = 0.5, novelty_min: float = 0.5, recent: int = 10):
        self.llm = llm
        self.enabled = enabled
        self.validity_min = validity_min
        self.novelty_min = novelty_min
        self.recent = deque(maxlen=recent)

    def novelty(self, mem: Memory, t: Thought) -> tuple:
        if t.claim:  # a claim is known only if memory holds it; a recent repeat of an unstored claim is still new
            ex = mem.find_claim(t.claim)
            return (0.0, ex) if ex else (1.0, None)
        if t.text in self.recent:
            return 0.0, None
        return 1.0 - mem.max_similarity(t.text), None

    def evaluate(self, mem: Memory, t: Thought) -> Decision:
        if t.kind == "none":
            return Decision("discard", 0.0, 0.0, "empty thought")
        if not self.enabled:
            ex = mem.find_claim(t.claim) if t.claim else None
            self.recent.append(t.text)
            return Decision("keep", 1.0, 1.0, "selector disabled: accept all", 0, ex)
        parents = [mem.nodes[i] for i in t.parents if i in mem.nodes]
        validity, jt = self.llm.judge(t, parents)
        nov, ex = self.novelty(mem, t)
        self.recent.append(t.text)
        if validity < self.validity_min:
            return Decision("discard", validity, nov, "failed grounding", jt)
        if nov < self.novelty_min:
            if ex is None:  # a claimless near-repeat points at no memory, so there is nothing to strengthen
                return Decision("discard", validity, nov, "repeats a recent thought", jt)
            return Decision("reinforce", validity, nov, "valid but already known", jt, ex)
        return Decision("keep", validity, nov, "valid and novel", jt)
