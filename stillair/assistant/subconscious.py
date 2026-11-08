"""The subconscious: a small local model driven by the idle algorithm.

The algorithm decides what to think about, with no language model involved:

    1. start from the memory most in need of attention: unresolved, then recent, then least revisited
    2. spread activation from it, through words, meaning, time and paths that proved useful
    3. skip pairs already tried or linked, so effort goes to new combinations

Only then is the small model asked whether the two memories connect, and asked again to
check its own answer. Passing links are stored as hunches: unverified until the conscious
model reviews one. When every worthwhile pair has been tried it goes round again, reaching
further through memory each time. It runs on the local CPU, so it needs no quota.
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.activation import spread
from ..core.memory import Node
from .associations import Associations, now
from .state import SharedState

CONNECT = {
    "type": "object",
    "properties": {"connected": {"type": "boolean"}, "link": {"type": "string"}},
    "required": ["connected", "link"],
    "additionalProperties": False,
}
CHECK = {
    "type": "object",
    "properties": {"follows": {"type": "boolean"}},
    "required": ["follows"],
    "additionalProperties": False,
}
MIN_ACTIVATION = 0.05
MAX_SIMILARITY = 0.85  # a link this close to an existing memory says nothing new
BASE_HOPS, MAX_HOPS = 2, 4
OPEN_BONUS, RECENCY_SECONDS, VISIT_PENALTY = 2.0, 3600.0, 0.3


class Subconscious:
    system = "You quietly look for connections between a person's memories. Reply with JSON only."

    def __init__(self, llm, state: SharedState, associations: Associations):
        self.llm, self.state, self.associations = llm, state, associations
        llm.system = self.system
        self.mem = state.mem
        self.rounds = 0  # how many times every pair has been tried; each round reaches further

    # ---- the idle algorithm: no language model involved ---------------------
    def salience(self, node: Node) -> float:
        age = max(0.0, now() - node.meta.get("at", 0.0))
        return OPEN_BONUS * bool(node.meta.get("open")) + math.exp(-age / RECENCY_SECONDS) - VISIT_PENALTY * node.meta.get("visits", 0)

    def _untried_partner(self, node: Node) -> Optional[Node]:
        reached = spread(self.mem, node, self.associations.weight, hops=min(MAX_HOPS, BASE_HOPS + self.rounds))
        self._reached = reached
        threshold = MIN_ACTIVATION / (1 + self.rounds)
        for other_id, level in sorted(reached.items(), key=lambda kv: -kv[1]):
            other = self.mem.nodes[other_id]
            pair = frozenset((node.id, other.id))
            if level >= threshold and pair not in self.state.tried and node.id not in other.parents and other.id not in node.parents:
                return other
        return None

    def next_pair(self) -> Optional[tuple]:
        """The next two memories worth a model call, or None if there is nothing left to think about."""
        for node in sorted(self.mem.readable(), key=self.salience, reverse=True):
            other = self._untried_partner(node)
            if other is not None:
                return node, other
        return None

    # ---- the small model --------------------------------------------------
    def _propose(self, a: Node, b: Node):
        d, used = self.llm.ask_json(
            f"Memory A: {a.text}\nMemory B: {b.text}\n\nDo these two memories connect in a way that tells the person "
            "something useful that neither says alone? Only say yes for a real connection, not a shared word. If yes, "
            "write the connection as one short sentence naming the specifics from both. If no, leave link empty.",
            CONNECT, 200,
        )
        link = d.get("link", "").strip() if d and d.get("connected") else ""
        return link, used

    def _holds(self, link: str, a: Node, b: Node):
        d, used = self.llm.ask_json(
            f"Memory A: {a.text}\nMemory B: {b.text}\nProposed connection: {link}\n\nDoes the proposed connection follow "
            "from these two memories without inventing anything? Be strict.",
            CHECK, 100,
        )
        return bool(d and d.get("follows") is True), used

    def _store(self, link: str, a: Node, b: Node) -> str:
        if self.mem.max_similarity(link) >= MAX_SIMILARITY:
            return "already known"
        self.mem.add("connection", link, None, "subconscious", [a.id, b.id], meta={"seen": False, "at": now()})
        a.meta["open"] = b.meta["open"] = False
        return "hunch stored"

    def think_once(self) -> bool:
        """One idle step. -> False when there is nothing left to think about."""
        self.associations.embed_missing()
        with self.state.lock:
            pair = self.next_pair()
            if pair is None and self.state.tried:  # everything tried: go round again, reaching further
                self.state.tried.clear()
                self.rounds += 1
                pair = self.next_pair()
            if pair is None:
                return False
            a, b = pair
            self.state.tried.add(frozenset((a.id, b.id)))
            a.meta["visits"] = a.meta.get("visits", 0) + 1
            activation = {i: round(v, 3) for i, v in self._reached.items()}
            activity = self.state.sub_activity = {"phase": "asking", "a": a.id, "b": b.id, "activation": activation, "link": ""}
        link, used = self._propose(a, b)
        outcome = "no connection"
        if link:
            self.state.sub_activity = {**activity, "phase": "checking", "link": link}
            holds, checked = self._holds(link, a, b)
            used += checked
            outcome = "rejected by its own check"
            if holds:
                with self.state.lock:
                    outcome = self._store(link, a, b)
        with self.state.lock:
            self.state.sub_activity = {**activity, "phase": "done", "link": link, "outcome": outcome}
            self.state.log_thought(a.text, b.text, link, outcome, used)
            self.state.save()
        return True
