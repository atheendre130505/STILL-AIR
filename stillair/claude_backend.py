"""Real-LLM backend (optional). Same interface as `SimLLM`, driven by the Anthropic API.

    pip install anthropic
    export ANTHROPIC_API_KEY=...            # or `ant auth login`
    python3 -m stillair demo --backend claude

Model: `claude-opus-5-5` by default. Override with STILLAIR_MODEL (e.g. `claude-haiku-5-5` is far
cheaper for the many small calls an always-on loop makes).

STATUS: exercised only against a fake client in tests/test_claude_backend.py. It has NOT been run
against the live API from the development machine (no credentials were available).
Refusals (`stop_reason == "refusal"`) are treated as an empty thought; no fallback model is configured.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .llm import LLM, Answer, SimLLM, Thought
from .memory import Node
from .schemas import PROPERTY

SYSTEM = (
    "You are one component of a memory system. You only know what the numbered memories say. "
    "Never use outside knowledge: the entities in them are fictional. Reply with JSON only."
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


class ClaudeLLM(LLM):
    name = "claude"
    followups = SimLLM.followups  # which entities to look up next is read off stored claims, no model call

    def __init__(self, model: Optional[str] = None, client=None, effort: str = "low"):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("STILLAIR_MODEL", "claude-opus-5-5")
        self.effort = effort
        self.tokens_used = 0

    def _call(self, user: str, schema: dict, max_tokens: int = 1024):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        used = getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0)
        self.tokens_used += used
        if resp.stop_reason == "refusal":
            return None, used
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            return None, used
        try:
            return json.loads(text), used
        except json.JSONDecodeError:
            return None, used

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
            "Numbered memories:\n" + _listing(seeds) + "\n\nHave one new thought. If exactly two memories combine to imply a "
            "new fact (the second is about something the first mentions), write it as kind 'derive' and list the two "
            "ids in order (relation fact first). If you only see a loose resemblance, kind 'analogy'. If nothing new, "
            "kind 'restate' (one id) or 'none'. Write the thought as one plain sentence.",
            THOUGHT,
        )
        if not d:
            return Thought("none", "", None, [], used)
        by_id = {n.id: n for n in seeds}
        parents = [by_id[i] for i in d["parent_ids"] if i in by_id]
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
        return max(0.0, min(1.0, float(d["score"]))), used

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
