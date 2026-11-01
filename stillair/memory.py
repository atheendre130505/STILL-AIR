"""Persistent memory: facts, derived connections, open questions and a self-model.

Retrieval is TF-IDF cosine over word tokens (pure Python, no embedding model needed
for the demo). Connections written by the idle loop carry a *strength* that decays
each loop step unless reinforced by retrieval or re-derivation: "the rest fade".
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

STOP = frozenset(
    "a an the of in on at to is are was were be and or that this it its with for by "
    "from as what which who when where how do does".split()
)

KINDS_READABLE = ("fact", "connection")


def tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


def ntok(text: str) -> int:
    """Cheap token proxy used for compute accounting in the simulated backend."""
    return len(text.split())


def claim_key(claim) -> Optional[tuple]:
    if not claim:
        return None
    s, path, o = claim
    return (s, tuple(path), o)


@dataclass
class Node:
    id: int
    kind: str  # fact | connection | question
    text: str
    claim: Optional[tuple] = None
    strength: float = 1.0
    born: int = 0  # idle-loop step (or session clock) at creation
    session: int = 0
    source: str = "user"  # user | idle
    parents: list = field(default_factory=list)
    hits: int = 0
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["claim"] = None if self.claim is None else [self.claim[0], list(self.claim[1]), self.claim[2]]
        return d

    @staticmethod
    def from_json(d: dict) -> "Node":
        d = dict(d)
        c = d.get("claim")
        d["claim"] = None if c is None else (c[0], tuple(c[1]), c[2])
        return Node(**d)


def _norm(v: dict) -> dict:
    n = math.sqrt(sum(x * x for x in v.values()))
    return {k: x / n for k, x in v.items()} if n else {}


def mix(parts) -> dict:
    """Weighted sum of sparse vectors, L2-normalised. parts = [(weight, vec), ...]."""
    out: dict = {}
    for w, vec in parts:
        for k, x in vec.items():
            out[k] = out.get(k, 0.0) + w * x
    return _norm(out)


class Memory:
    def __init__(self):
        self.nodes: dict = {}
        self._next = 1
        self.df: Counter = Counter()
        self._wcache: dict = {}
        self._cache_ver = -1
        self._claims: dict = {}  # claim_key -> node id (readable kinds only)
        self.self_model: dict = {
            "kept": 0, "reinforced": 0, "discarded": 0, "pruned": 0,
            "curiosity": 0.3, "uncertainty": 0.0,
        }

    # ---- writing -------------------------------------------------------
    def add(self, kind, text, claim=None, source="user", parents=(), born=0, session=0, meta=None) -> Node:
        n = Node(self._next, kind, text, claim, 1.0, born, session, source, list(parents), 0, meta or {})
        self._next += 1
        self.nodes[n.id] = n
        if kind in KINDS_READABLE:
            for t in set(tokens(text)):
                self.df[t] += 1
            k = claim_key(claim)
            if k is not None:
                self._claims.setdefault(k, n.id)
        return n

    def remove(self, node_id: int) -> None:
        n = self.nodes.pop(node_id, None)
        if n is None:
            return
        self._wcache.pop(node_id, None)
        if n.kind in KINDS_READABLE:
            for t in set(tokens(n.text)):
                self.df[t] -= 1
            k = claim_key(n.claim)
            if k is not None and self._claims.get(k) == node_id:
                del self._claims[k]
                for other in self.nodes.values():  # promote another holder of the same claim
                    if other.kind in KINDS_READABLE and claim_key(other.claim) == k:
                        self._claims[k] = other.id
                        break

    def reinforce(self, node: Node, amount: float = 0.5) -> None:
        node.strength = min(2.0, node.strength + amount)
        node.hits += 1

    def decay_and_prune(self, rate: float, floor: float) -> list:
        pruned = []
        for n in list(self.nodes.values()):
            if n.kind != "connection":
                continue
            n.strength -= rate
            if n.strength < floor:
                pruned.append(n.id)
                self.remove(n.id)
        self.self_model["pruned"] += len(pruned)
        return pruned

    # ---- reading -------------------------------------------------------
    def find_claim(self, claim) -> Optional[Node]:
        k = claim_key(claim)
        nid = self._claims.get(k) if k is not None else None
        return self.nodes.get(nid) if nid is not None else None

    def _idf(self, t: str) -> float:
        return math.log((len(self.nodes) + 1) / (self.df.get(t, 0) + 0.5))

    def vec(self, text: str) -> dict:
        tf = Counter(tokens(text))
        return _norm({t: (1 + math.log(c)) * self._idf(t) for t, c in tf.items()})

    def _wvec(self, n: Node) -> dict:
        ver = len(self.nodes) // 16  # idf refreshed every 16 additions: cheap and close enough
        if ver != self._cache_ver:
            self._wcache.clear()
            self._cache_ver = ver
        v = self._wcache.get(n.id)
        if v is None:
            v = self._wcache[n.id] = self.vec(n.text)
        return v

    def retrieve(self, query, k=3, kinds=KINDS_READABLE, noise=0.0, rng=None, exclude=(), penalty=None):
        q = self.vec(query) if isinstance(query, str) else query
        scored = []
        for n in self.nodes.values():
            if n.kind not in kinds or n.id in exclude:
                continue
            v = self._wvec(n)
            s = sum(x * v.get(t, 0.0) for t, x in q.items())
            if noise and rng is not None:
                s += rng.gauss(0.0, noise)
            if penalty:
                s -= penalty.get(n.id, 0.0)
            scored.append((s, n))
        scored.sort(key=lambda p: -p[0])
        return [p for p in scored[:k] if p[0] > -1e9]

    def max_similarity(self, text: str, kinds=KINDS_READABLE) -> float:
        q = self.vec(text)
        best = 0.0
        for n in self.nodes.values():
            if n.kind in kinds:
                v = self._wvec(n)
                best = max(best, sum(x * v.get(t, 0.0) for t, x in q.items()))
        return best

    def readable(self, kinds=KINDS_READABLE) -> list:
        return [n for n in self.nodes.values() if n.kind in kinds]

    def open_questions(self) -> list:
        return [n for n in self.nodes.values() if n.kind == "question" and n.meta.get("status") == "open"]

    def counts(self) -> dict:
        c = Counter(n.kind for n in self.nodes.values())
        return {"fact": c["fact"], "connection": c["connection"], "question": c["question"]}

    # ---- persistence ---------------------------------------------------
    def to_json(self) -> dict:
        return {"next": self._next, "self_model": self.self_model, "nodes": [n.to_json() for n in self.nodes.values()]}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Memory":
        with open(path) as f:
            d = json.load(f)
        m = cls()
        m._next = d["next"]
        m.self_model = d["self_model"]
        for nd in d["nodes"]:
            n = Node.from_json(nd)
            m.nodes[n.id] = n
            if n.kind in KINDS_READABLE:
                for t in set(tokens(n.text)):
                    m.df[t] += 1
                k = claim_key(n.claim)
                if k is not None:
                    m._claims.setdefault(k, n.id)
        return m
