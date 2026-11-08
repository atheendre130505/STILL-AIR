"""Spreading activation over memory: find related memories without calling a model.

Activation starts at one memory and flows along weighted edges, split by how much weight
leaves a node, fading each hop, keeping only the strongest few. This can reach a memory that
shares nothing with the start but does relate to a memory in between. Adapted from Synapse
(arXiv 2601.02744). The caller decides what makes two memories related, through `weight`.
"""

from __future__ import annotations

from typing import Callable

from .memory import Memory, Node


def spread(mem: Memory, start: Node, weight: Callable[[Node, Node], float], hops: int = 2, decay: float = 0.5,
           min_edge: float = 0.08, keep: int = 7) -> dict:
    """-> {node id: activation} for memories reached from `start`, excluding `start`."""
    pool = mem.readable()
    by_id = {n.id: n for n in pool}
    active = {start.id: 1.0}
    for _ in range(hops):
        reached = dict(active)
        for node_id, level in active.items():
            node = by_id[node_id]
            edges = {o.id: w for o in pool if o.id != node_id and (w := weight(node, o)) >= min_edge}
            total = sum(edges.values())
            for other, w in edges.items():
                reached[other] = max(reached.get(other, 0.0), decay * level * w / total)
        active = dict(sorted(reached.items(), key=lambda kv: -kv[1])[:keep])
    active.pop(start.id, None)
    return active
