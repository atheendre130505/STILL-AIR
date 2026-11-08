"""The maths of memory: how related two memories are, and how activation spreads between them."""

from __future__ import annotations

import json
import math
import time
import urllib.request

from .memory import Node
from .state import SharedState

MEANING_FLOOR = 0.2  # all-minilm: unrelated sentence pairs measured at most 0.16, related mostly 0.33-0.51
USED_STRENGTH, PARENT_STRENGTH = 0.3, 0.1
SEARCH_SEEDS, SEARCH_LIMIT = 3, 12
MIN_SEARCH_SCORE, MAX_FACTS, MAX_CONNECTIONS = 0.1, 6, 3


def now() -> float:
    return time.time()


class Embedder:
    """Sentence-to-meaning-vector model running in the local Ollama server."""

    def __init__(self, name: str, base_url: str, timeout: float = 120.0):
        self.name, self.base_url, self.timeout = name, base_url.rstrip("/"), timeout

    def embed(self, texts: list) -> list:
        body = json.dumps({"model": self.name, "input": texts}).encode()
        req = urllib.request.Request(f"{self.base_url}/embeddings", body, {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            items = sorted(json.loads(r.read())["data"], key=lambda d: d["index"])
        return [_unit(d["embedding"]) for d in items]


def _unit(vector: list) -> list:
    size = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [round(x / size, 5) for x in vector]


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) if a and b else 0.0


def _meaning(a, b) -> float:
    return max(0.0, (cosine(a, b) - MEANING_FLOOR) / (1.0 - MEANING_FLOOR))


class Links:
    def __init__(self, state: SharedState, embedder=None):
        self.state, self.mem, self.embedder = state, state.mem, embedder
        self.meaning_working = False
        self._relations: dict = {}  # a pair's reason and strength never change, so each is worked out once
        self._drawn_key, self._drawn = None, []

    def forget(self) -> None:
        self._relations.clear()

    # ---- how related two memories are -----------------------------------
    def _words(self, a: Node, b: Node) -> float:
        va, vb = self.mem.word_vector(a), self.mem.word_vector(b)
        return sum(x * vb.get(t, 0.0) for t, x in va.items())

    def relation(self, a: Node, b: Node) -> tuple:
        """-> (kind of link, strength): the strongest reason the two are related."""
        pair = frozenset((a.id, b.id))
        known = self._relations.get(pair)
        if known is not None:
            return known
        reasons = {
            "link": 1.0 if a.id in b.parents or b.id in a.parents else 0.0,
            "words": self._words(a, b),
            "meaning": _meaning(a.meta.get("vec"), b.meta.get("vec")),
        }
        kind = max(reasons, key=reasons.get)
        if self.embedder is None or ("vec" in a.meta and "vec" in b.meta):
            self._relations[pair] = (kind, reasons[kind])
        return kind, reasons[kind]

    def weight(self, a: Node, b: Node) -> float:
        return self.relation(a, b)[1] * b.strength

    # ---- spreading activation (adapted from Synapse, arXiv 2601.02744) -----
    def spread(self, start: Node, hops: int = 2, decay: float = 0.5, min_edge: float = 0.08, keep: int = 7) -> dict:
        """-> {node id: activation} for memories reached from `start`, excluding `start`."""
        pool = self.mem.readable()
        by_id = {n.id: n for n in pool}
        active = {start.id: 1.0}
        for _ in range(hops):
            reached = dict(active)
            for node_id, level in active.items():
                node = by_id[node_id]
                edges = {o.id: w for o in pool if o.id != node_id and (w := self.weight(node, o)) >= min_edge}
                total = sum(edges.values())
                for other, w in edges.items():
                    reached[other] = max(reached.get(other, 0.0), decay * level * w / total)
            active = dict(sorted(reached.items(), key=lambda kv: -kv[1])[:keep])
        active.pop(start.id, None)
        return active

    # ---- searching from a message ------------------------------------------
    def embed_one(self, text: str):
        if self.embedder is None:
            return None
        try:
            vector = self.embedder.embed([text])[0]
            self.meaning_working = True
            return vector
        except Exception:
            self.meaning_working = False
            return None

    def search(self, text: str, seeds: int = SEARCH_SEEDS, limit: int = SEARCH_LIMIT, include_outside: bool = False) -> list:
        """Search from some text through memories and what was worked out from them. -> [(score, node)], best first."""
        query = self.embed_one(text)
        with self.state.lock:
            usable = [n for n in self.mem.readable() if n.kind in ("fact", "connection") or (include_outside and n.kind == "reading")]
            words = self.mem.vec(text)
            scored = []
            for n in usable:
                vector = self.mem.word_vector(n)
                score = max(sum(x * vector.get(t, 0.0) for t, x in words.items()), _meaning(query, n.meta.get("vec")))
                if score > 0:
                    scored.append((score * n.strength, n))
            scored.sort(key=lambda p: -p[0])
            found = {n.id: (score, n) for score, n in scored[:seeds]}
            allowed = {n.id for n in usable}
            for score, seed in scored[:seeds]:
                for other_id, level in self.spread(seed).items():
                    if other_id in allowed and score * level > found.get(other_id, (0.0,))[0]:
                        found[other_id] = (score * level, self.mem.nodes[other_id])
        return sorted(found.values(), key=lambda p: -p[0])[:limit]

    @staticmethod
    def most_relevant(found: list) -> list:
        """What the main model gets to read, favouring the person's own words."""
        relevant = [n for score, n in found if score >= MIN_SEARCH_SCORE]
        return [n for n in relevant if n.kind == "fact"][:MAX_FACTS] + [n for n in relevant if n.kind == "connection"][:MAX_CONNECTIONS]

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

    # ---- learning from what the main model actually used -------------------
    def learn_from_use(self, ids: list) -> None:
        for i in [i for i in dict.fromkeys(ids) if i in self.mem.nodes]:
            node = self.mem.nodes[i]
            self.mem.reinforce(node, USED_STRENGTH)
            for parent in node.parents:
                if parent in self.mem.nodes:
                    self.mem.reinforce(self.mem.nodes[parent], PARENT_STRENGTH)

    # ---- for the backend view ---------------------------------------------
    def strongest_links(self, per_node: int = 3, min_strength: float = 0.08) -> list:
        """The links worth drawing: each memory's few strongest. Cached until memory changes."""
        with self.state.lock:
            nodes = self.mem.readable()
            key = (tuple(n.id for n in nodes), sum("vec" in n.meta for n in nodes))
            if key == self._drawn_key:
                return self._drawn
            chosen = {}
            for a in nodes:
                ranked = sorted(((*self.relation(a, b), b) for b in nodes if b.id != a.id), key=lambda r: -r[1])
                for kind, strength, b in ranked[:per_node]:
                    if strength >= min_strength:
                        chosen[frozenset((a.id, b.id))] = (kind, strength)
            self._drawn = [{"a": min(p), "b": max(p), "kind": kind, "strength": round(strength, 3)}
                           for p, (kind, strength) in chosen.items()]
            self._drawn_key = key
            return self._drawn
