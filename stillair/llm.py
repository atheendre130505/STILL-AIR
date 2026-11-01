"""LLM backends.

`SimLLM` is a deterministic, seeded stand-in for a language model. It exists so the
demo and the experiment run offline and reproducibly. It is *not* evidence about real
models: its two imperfections are explicit, tunable parameters that the experiment
sweeps rather than hides.

    hallucination_rate : how often an idle thought is a confabulated link
                         (joining facts that do not actually chain)
    verifier_miss      : how often the grounding judge is fooled (and, at half
                         that rate, wrongly rejects a valid thought)

`ClaudeLLM` (claude_backend.py) implements the same interface against the real API.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .memory import Node, ntok
from .schemas import PROPERTY, SCHEMAS, parse_fact, parse_question


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


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.lower(), b.lower()):
        if x != y:
            break
        n += 1
    return n


def _sentence(text: str) -> str:
    return text.rstrip(".")


class SimLLM(LLM):
    name = "sim"

    def __init__(self, seed: int = 0, hallucination_rate: float = 0.3, verifier_miss: float = 0.15):
        self.rng = random.Random(seed)
        self.h = hallucination_rate
        self.m = verifier_miss

    # ---- ingestion -----------------------------------------------------
    def extract(self, text: str):
        return parse_fact(text)

    # ---- idle thought --------------------------------------------------
    def followups(self, seeds: list) -> list:
        ents = []
        for n in seeds:
            if n.kind != "fact" or not n.claim:
                continue
            e = n.claim[0] if n.claim[1] == (PROPERTY,) else n.claim[2]  # the shared "bridge" entity
            if e not in ents:
                ents.append(e)
        return ents

    def associate(self, seeds: list, nudge=None) -> Thought:
        facts = [n for n in seeds if n.kind == "fact" and n.claim]
        rels = [n for n in facts if n.claim[1][0] != PROPERTY]
        props = [n for n in facts if n.claim[1] == (PROPERTY,)]
        chains = [(r, p) for r in rels for p in props if r.claim[2] == p.claim[0]]

        if chains:
            r, p = chains[0]
            if self.rng.random() < self.h:
                # confabulate: join the relation fact to a look-alike entity's property
                decoys = [q for q in props if q.claim[0] != r.claim[2]]
                if decoys:
                    p = max(decoys, key=lambda q: _common_prefix(q.claim[0], r.claim[2]))
            return self._derive(r, p)

        if rels and props and self.rng.random() < self.h:
            return self._derive(rels[0], props[0])  # a forced leap between unrelated facts

        if len(seeds) >= 2 and self.rng.random() < 0.5:
            a, b = seeds[0], seeds[1]
            shared = sorted(set(a.text.lower().split()) & set(b.text.lower().split()) - {"the", "is", "a"})
            word = shared[0] if shared else "something"
            text = f"Perhaps the first note and the second note are related: both involve '{word}'."
            return Thought("analogy", text, None, [a.id, b.id], ntok(text) + 5)

        if seeds:
            s = seeds[0]
            text = f"Recall: {s.text}"
            return Thought("restate", text, s.claim, [s.id], ntok(text) + 5)
        return Thought("none", "", None, [], 1)

    def _derive(self, r: Node, p: Node) -> Thought:
        rel = r.claim[1][0]
        sent = _sentence(p.text)
        if sent.startswith("The "):
            sent = "the " + sent[4:]
        text = f"{_sentence(r.text)}, and {sent}."
        claim = (r.claim[0], (rel, PROPERTY), p.claim[2])
        return Thought("derive", text, claim, [r.id, p.id], ntok(text) + 5)

    # ---- grounding judge -----------------------------------------------
    def judge(self, thought: Thought, parents: list):
        toks = ntok(thought.text) + sum(ntok(p.text) for p in parents)
        if thought.kind == "none":
            return 0.0, toks
        if thought.kind == "restate":
            return 1.0, toks
        if thought.kind == "analogy" or thought.claim is None:
            return 0.2, toks  # unfalsifiable surface similarity: weak support
        # derive: structurally check the claim against its cited parents
        truly_valid = False
        if len(parents) == 2 and parents[0].claim and parents[1].claim:
            r, p = parents
            s, path, o = thought.claim
            truly_valid = (
                r.claim[0] == s
                and r.claim[1] == (path[0],)
                and r.claim[2] == p.claim[0]  # bridge entity must match exactly
                and p.claim[1] == (PROPERTY,)
                and p.claim[2] == o
            )
        if truly_valid:
            ok = self.rng.random() >= self.m / 2
            return (0.95 if ok else 0.1), toks
        fooled = self.rng.random() < self.m
        return (0.9 if fooled else 0.05), toks

    # ---- task mode -----------------------------------------------------
    def answer(self, question: str, context: list, scratch: dict) -> Answer:
        toks = 3
        parsed = parse_question(question)
        if not parsed:
            return Answer(None, None, [], toks)
        kind, ent, rel = parsed
        if kind == "direct":
            for n in context:
                if n.claim and n.claim[0] == ent and n.claim[1] == (PROPERTY,):
                    return Answer(n.claim[2], None, [n.id], toks + ntok(n.claim[2]))
            return Answer(None, None, [], toks)
        # chain question: A -rel-> M -property-> P
        # a stored connection answers in one step; if memory holds conflicting beliefs,
        # prefer the better-supported one (replay reinforces true links, confabulations rarely recur)
        direct = [n for n in context if n.claim and n.claim[0] == ent and n.claim[1] == (rel, PROPERTY)]
        if direct:
            n = max(direct, key=lambda x: x.strength)
            return Answer(n.claim[2], None, [n.id], toks + ntok(n.claim[2]))
        bridge = scratch.get("bridge")
        if bridge:
            for n in context:
                if n.claim and n.claim[0] == bridge and n.claim[1] == (PROPERTY,):
                    return Answer(n.claim[2], None, scratch.get("via", []) + [n.id], toks + ntok(n.claim[2]))
            return Answer(None, None, [], toks)
        for n in context:
            if n.claim and n.claim[0] == ent and n.claim[1] == (rel,):
                scratch["via"] = [n.id]
                scratch["bridge"] = n.claim[2]
                return Answer(None, n.claim[2], [n.id], toks)
        return Answer(None, None, [], toks)
