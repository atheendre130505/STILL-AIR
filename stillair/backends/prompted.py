"""Shared logic for backends that talk to a real model: prompts, JSON schemas, reply parsing.

A subclass only implements `_request`, which sends one prompt and returns the reply text.
A failed call counts as an empty thought, so one transient error cannot kill an always-on
loop; `max_consecutive_errors` failures in a row re-raise, so a bad key or outage is not hidden.
Refusals are treated as an empty thought.
"""

from __future__ import annotations

import json
import sys

from ..world.schemas import PROPERTY
from .base import LLM, Answer, Thought
from .sim import SimLLM

SYSTEM = (
    "You are one component of a memory system. You only know what the numbered memories say. "
    "Never use outside knowledge: the entities in them are fictional. Reply with JSON only."
)

ASSOCIATE_INSTRUCTIONS = (
    "Have one new thought. Look for two memories where the second is about the very same named thing the first "
    "mentions. The name must match exactly; a similar-looking name is a different thing. If you find such a pair, use "
    "kind 'derive', list the two ids (the memory that mentions the thing first), and write one sentence that joins them "
    "and keeps the shared name, in the form '<first memory>, and <second memory>'. For example: 'The Amber gate is made "
    "of Olvin wood, and Olvin wood smells of rain.' Never shorten it to 'The Amber gate smells of rain': that moves the "
    "property onto the wrong thing. If you only see a loose resemblance, use kind 'analogy'. If nothing new follows, use "
    "kind 'restate' (one id) or 'none'."
)

TRIPLE = {
    "type": "object",
    "properties": {
        "has_triple": {"type": "boolean"},
        "subject": {"type": "string"},
        "relation": {"type": "string"},
        "object": {"type": "string"},
    },
    "required": ["has_triple", "subject", "relation", "object"],
    "additionalProperties": False,
}
THOUGHT = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["derive", "analogy", "restate", "none"]},
        "text": {"type": "string"},
        "parent_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["kind", "text", "parent_ids"],
    "additionalProperties": False,
}
JUDGE = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}
ANSWER = {
    "type": "object",
    "properties": {
        "answer": {"type": ["string", "null"]},
        "lookup": {"type": ["string", "null"]},
        "used_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "lookup", "used_ids"],
    "additionalProperties": False,
}


def _listing(nodes) -> str:
    return "\n".join(f"[{n.id}] {n.text}" for n in nodes)


class PromptedLLM(LLM):
    usage_includes_input = True
    system = SYSTEM
    followups = SimLLM.followups  # which entities to look up next is read off stored claims, no model call

    def __init__(self, max_consecutive_errors: int = 5):
        self.tokens_used = 0
        self.calls = 0
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0

    def _request(self, user: str, schema: dict, max_tokens: int):
        """Send one prompt. -> (reply text or None if there is none, tokens used)"""
        raise NotImplementedError

    def _call(self, user: str, schema: dict, max_tokens: int = 1024):
        self.calls += 1
        try:
            text, used = self._request(user, schema, max_tokens)
        except Exception as e:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise
            if self.consecutive_errors == 1:  # an empty thought should not look like the model had nothing to say
                print(f"warning: model call failed, treating it as empty: {e}", file=sys.stderr)
            return None, 0
        self.consecutive_errors = 0
        self.tokens_used += used
        if text is None:
            return None, used
        try:
            return json.loads(text), used
        except json.JSONDecodeError:
            return None, used

    def ask_json(self, user: str, schema: dict, max_tokens: int = 1024):
        """One structured model call. -> (parsed reply or None, tokens used)"""
        return self._call(user, schema, max_tokens)

    def extract(self, text: str):
        d, _ = self._call(
            f"Sentence: {text}\nExtract one (subject, relation, object) triple. If the sentence states a property of "
            f"an entity, use relation 'property' and put the property phrase in object. Otherwise use a short "
            f"snake_case relation. Set has_triple=false if there is no clear triple.",
            TRIPLE,
        )
        if not d or not d.get("has_triple"):
            return None
        return (d["subject"], (d["relation"],), d["object"])

    def associate(self, seeds: list, nudge=None) -> Thought:
        if not seeds:
            return Thought("none", "", None, [], 1)
        d, used = self._call(
            "Numbered memories:\n" + _listing(seeds) + "\n\n" + ASSOCIATE_INSTRUCTIONS,
            THOUGHT,
        )
        if not d:
            return Thought("none", "", None, [], used)
        if d.get("kind") not in ("derive", "analogy", "restate", "none") or not isinstance(d.get("text"), str):
            return Thought("none", "", None, [], used)
        by_id = {n.id: n for n in seeds}
        parents = [by_id[i] for i in d.get("parent_ids") or [] if i in by_id]
        claim = None
        if d["kind"] == "derive" and len(parents) == 2 and all(p.claim for p in parents):
            a, b = parents
            if a.claim[1] == (PROPERTY,) and b.claim[1] != (PROPERTY,):
                a, b = b, a
            claim = (a.claim[0], tuple(a.claim[1]) + tuple(b.claim[1]), b.claim[2])
            parents = [a, b]
        elif d["kind"] == "restate" and len(parents) == 1:
            claim = parents[0].claim
        return Thought(d["kind"], d["text"], claim, [p.id for p in parents], used)

    def judge(self, thought: Thought, parents: list):
        if thought.kind == "none":
            return 0.0, 0
        if thought.kind == "restate":
            return 1.0, 0
        d, used = self._call(
            "Memories:\n" + _listing(parents) + f"\n\nCandidate thought: {thought.text}\n\n"
            "Is the candidate logically entailed by these memories alone (the entity linking them must be the same "
            "entity, not a similar-looking one)? Score 1.0 if clearly entailed, 0.0 if not supported. Be strict.",
            JUDGE,
        )
        if not d:
            return 0.0, used
        try:
            return max(0.0, min(1.0, float(d["score"]))), used
        except (KeyError, TypeError, ValueError):  # an unreadable verdict is not a pass
            return 0.0, used

    def answer(self, question: str, context: list, scratch: dict) -> Answer:
        seen = scratch.setdefault("ctx", {})
        for n in context:
            seen[n.id] = n
        d, used = self._call(
            "Memories:\n" + _listing(seen.values()) + f"\n\nQuestion: {question}\n"
            "Answer using only these memories. If answering needs facts about another entity they mention, set "
            "answer=null and lookup to that entity's name. If the memories do not contain the answer, set both null.",
            ANSWER,
        )
        if not d:
            return Answer(None, None, [], used)
        return Answer(d.get("answer"), d.get("lookup") if not d.get("answer") else None, list(d.get("used_ids") or []), used)
