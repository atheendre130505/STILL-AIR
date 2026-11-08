"""What makes two memories related: shared words, similar meaning, closeness in time, and past use."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from itertools import combinations
from typing import Optional

from ..core.memory import Node
from .state import SharedState

MEANING_FLOOR = 0.2  # all-minilm: unrelated sentence pairs measured at most 0.16, related mostly 0.33-0.51
TIME_WEIGHT = 0.15  # a time link is real but weaker than a link of content
TIME_SCALE_SECONDS = 30 * 60
LEARNING_STEP = 0.25
MAX_LEARNED = 1.0  # a much-used path can at most double in strength
USED_STRENGTH, PARENT_STRENGTH = 0.3, 0.1


class OllamaEmbedder:
    """Sentence-to-meaning-vector model running in the local Ollama server."""

    def __init__(self, model: str, base_url: str, timeout: float = 120.0):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    def embed(self, texts: list) -> list:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(f"{self.base_url}/embeddings", body, {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            items = sorted(json.loads(r.read())["data"], key=lambda d: d["index"])
        return [_unit(d["embedding"]) for d in items]


def _unit(vector: list) -> list:
    size = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [round(x / size, 5) for x in vector]


def _meaning(a: Optional[list], b: Optional[list]) -> float:
    if not a or not b:
        return 0.0
    cosine = sum(x * y for x, y in zip(a, b))
    return max(0.0, (cosine - MEANING_FLOOR) / (1.0 - MEANING_FLOOR))


class Associations:
    def __init__(self, state: SharedState, embedder=None):
        self.state, self.mem, self.embedder = state, state.mem, embedder
        self.meaning_working = False
        self._links_key, self._links = None, []

    # ---- how related two memories are -----------------------------------
    def _words(self, a: Node, b: Node) -> float:
        va, vb = self.mem._wvec(a), self.mem._wvec(b)
        return sum(x * vb.get(t, 0.0) for t, x in va.items())

    @staticmethod
    def _time(a: Node, b: Node) -> float:
        if a.kind != "fact" or b.kind != "fact" or "at" not in a.meta or "at" not in b.meta:
            return 0.0
        return TIME_WEIGHT * math.exp(-abs(a.meta["at"] - b.meta["at"]) / TIME_SCALE_SECONDS)

    def weight(self, a: Node, b: Node) -> float:
        """Strength of the path from a to b, whichever kind of relatedness is strongest."""
        linked = 1.0 if a.id in b.parents or b.id in a.parents else 0.0
        base = max(linked, self._words(a, b), _meaning(a.meta.get("vec"), b.meta.get("vec")), self._time(a, b))
        return base * b.strength * (1.0 + self.state.learned.get(frozenset((a.id, b.id)), 0.0))

    def relation(self, a: Node, b: Node) -> tuple:
        """-> (kind of link, its strength before learning), naming the strongest reason the two are related."""
        reasons = {
            "link": 1.0 if a.id in b.parents or b.id in a.parents else 0.0,
            "words": self._words(a, b),
            "meaning": _meaning(a.meta.get("vec"), b.meta.get("vec")),
            "time": self._time(a, b),
        }
        kind = max(reasons, key=reasons.get)
        return kind, reasons[kind]

    def strongest_links(self, per_node: int = 3, min_strength: float = 0.08) -> list:
        """The links worth drawing: each memory's few strongest. Cached until memory or learning changes."""
        with self.state.lock:
            nodes = self.mem.readable()
            key = (tuple(n.id for n in nodes), sum("vec" in n.meta for n in nodes), sum(self.state.learned.values()))
            if key == self._links_key:
                return self._links
            chosen = {}
            for a in nodes:
                ranked = sorted(((*self.relation(a, b), b) for b in nodes if b.id != a.id), key=lambda r: -r[1])
                for kind, strength, b in ranked[:per_node]:
                    if strength >= min_strength:
                        chosen[frozenset((a.id, b.id))] = (kind, strength)
            self._links = [{"a": min(p), "b": max(p), "kind": kind, "strength": round(strength, 3), "learned": self.state.learned.get(p, 0.0)}
                           for p, (kind, strength) in chosen.items()]
            self._links_key = key
            return self._links

    # ---- meaning vectors --------------------------------------------------
    def embed_missing(self, batch: int = 16) -> None:
        """Give memories their meaning vectors. If the embedder is down, word matching carries on alone."""
        if self.embedder is None:
            return
        with self.state.lock:
            missing = [n for n in self.mem.readable() if "vec" not in n.meta][:batch]
        if not missing:
            return
        try:
            vectors = self.embedder.embed([n.text for n in missing])
        except Exception:
            self.meaning_working = False
            return
        self.meaning_working = True
        with self.state.lock:
            for node, vector in zip(missing, vectors):
                node.meta["vec"] = vector

    def recall(self, text: str, k: int) -> list:
        """The k memories most related to `text`, by words or meaning."""
        query = None
        if self.embedder is not None:
            try:
                query = self.embedder.embed([text])[0]
                self.meaning_working = True
            except Exception:
                self.meaning_working = False
        with self.state.lock:
            words = self.mem.vec(text)
            scored = []
            for n in self.mem.readable():
                vec = self.mem._wvec(n)
                score = max(sum(x * vec.get(t, 0.0) for t, x in words.items()), _meaning(query, n.meta.get("vec")))
                if score > 0:
                    scored.append((score * n.strength, n))
            scored.sort(key=lambda p: -p[0])
            return [n for _, n in scored[:k]]

    # ---- learning from what the main model actually used -------------------
    def _boost(self, a: int, b: int) -> None:
        pair = frozenset((a, b))
        self.state.learned[pair] = min(MAX_LEARNED, self.state.learned.get(pair, 0.0) + LEARNING_STEP)

    def learn_from_use(self, ids: list) -> None:
        used = [i for i in dict.fromkeys(ids) if i in self.mem.nodes]
        for a, b in combinations(used, 2):
            self._boost(a, b)
        for i in used:
            node = self.mem.nodes[i]
            self.mem.reinforce(node, USED_STRENGTH)
            for parent in node.parents:
                if parent in self.mem.nodes:
                    self.mem.reinforce(self.mem.nodes[parent], PARENT_STRENGTH)
                    self._boost(i, parent)


def now() -> float:
    return time.time()
