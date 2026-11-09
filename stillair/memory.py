"""Long-term memory: the person's own words ("fact"), related Wikipedia pages ("reading"), and connections between them."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, fields

STOP = frozenset(
    "a an the of in on at to is are was were be and or that this it its with for by "
    "from as what which who when where how do does "
    "i me my mine am im you your u we our us nd btw".split()
)
MAX_STRENGTH = 2.0


def tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


@dataclass
class Node:
    id: int
    kind: str  # fact | reading | connection
    text: str
    source: str = "user"
    parents: list = field(default_factory=list)
    strength: float = 1.0
    hits: int = 0  # how often the main model has relied on it
    meta: dict = field(default_factory=dict)


def _unit(vector: dict) -> dict:
    size = math.sqrt(sum(x * x for x in vector.values()))
    return {k: x / size for k, x in vector.items()} if size else {}


class Memory:
    def __init__(self):
        self.nodes: dict = {}
        self._next = 1
        self.df: Counter = Counter()
        self._vectors: dict = {}
        self._vectors_for = -1

    def add(self, kind, text, source="user", parents=(), meta=None) -> Node:
        node = Node(self._next, kind, text, source, list(parents), meta=meta or {})
        self._next += 1
        self.nodes[node.id] = node
        self.df.update(set(tokens(text)))
        return node

    def remove(self, node_id: int) -> None:
        node = self.nodes.pop(node_id, None)
        if node is not None:
            self._vectors.pop(node_id, None)
            self.df.subtract(set(tokens(node.text)))

    def clear(self) -> None:
        """Forget everything. Ids keep counting up, so work already in flight cannot be mistaken for a new memory."""
        self.nodes.clear()
        self.df.clear()
        self._vectors.clear()

    def reinforce(self, node: Node, amount: float) -> None:
        node.strength = min(MAX_STRENGTH, node.strength + amount)
        node.hits += 1

    def connections(self) -> list:
        return [n for n in self.nodes.values() if n.kind == "connection"]

    def fade(self, rate: float, floor: float) -> None:
        """Connections weaken unless used, and are dropped once weak enough. The person's own words never fade."""
        for node in self.connections():
            node.strength -= rate
            if node.strength < floor:
                self.remove(node.id)

    def keep_strongest(self, limit: int) -> None:
        """Memory does not hold every connection: past `limit`, the weakest are dropped."""
        for node in sorted(self.connections(), key=lambda n: (n.strength, n.hits, n.id))[:-limit]:
            self.remove(node.id)

    def readable(self) -> list:
        return list(self.nodes.values())

    # ---- word vectors: rarer words count for more --------------------------
    def vec(self, text: str) -> dict:
        count = Counter(tokens(text))
        n = len(self.nodes) + 1
        return _unit({t: (1 + math.log(c)) * math.log(n / (self.df.get(t, 0) + 0.5)) for t, c in count.items()})

    def word_vector(self, node: Node) -> dict:
        generation = len(self.nodes) // 16  # word rarity is refreshed every 16 additions: cheap and close enough
        if generation != self._vectors_for:
            self._vectors.clear()
            self._vectors_for = generation
        vector = self._vectors.get(node.id)
        if vector is None:
            vector = self._vectors[node.id] = self.vec(node.text)
        return vector

    def max_similarity(self, text: str) -> float:
        query = self.vec(text)
        return max((sum(x * self.word_vector(n).get(t, 0.0) for t, x in query.items()) for n in self.nodes.values()), default=0.0)

    # ---- saving ------------------------------------------------------------
    def to_json(self) -> dict:
        return {"next": self._next, "nodes": [dict(n.__dict__) for n in self.nodes.values()]}

    @classmethod
    def from_json(cls, d: dict) -> "Memory":
        memory = cls()
        memory._next = d["next"]
        known = {f.name for f in fields(Node)}
        for item in d["nodes"]:
            node = Node(**{k: v for k, v in item.items() if k in known})
            memory.nodes[node.id] = node
            memory.df.update(set(tokens(node.text)))
        return memory
