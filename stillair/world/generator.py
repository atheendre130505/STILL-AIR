"""Synthetic multi-session knowledge world used by the evaluation and the demo.

Each *chain* is a two-hop fact pair, delivered in different sessions:

    session i : "The Kelpmer bridge is built from Zorvian stone."     (A -built_from-> M)
    session j : "Zorvian stone conducts heat extremely well."         (M -property-> P)

Nobody ever states the combined fact. A later question asks about A and needs P,
so an agent either does two retrievals at test time (reusing nothing) or has
already *connected* the two facts during idle time.

Dangling facts (a relation with no property, or a property nobody refers to) and
near-name decoys (Zorvian vs Zorvane) give the idle loop plenty of plausible but
wrong connections to make, which is what the selector is for.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .schemas import PROPERTY, SCHEMAS

_SYL = [
    "zor", "vi", "an", "brin", "dle", "kel", "mer", "tha", "lo", "mu", "ren", "ska",
    "ar", "wy", "ix", "dro", "pel", "nu", "sav", "qu", "fen", "gor", "hel", "jas",
    "lum", "nor", "bal", "cre", "dun", "eth", "fal", "gri", "hov", "ilm", "jor",
    "kyr", "lav", "mor", "ned", "ost", "pra", "rav", "sil", "tor", "ulm", "vel",
]


@dataclass
class Fact:
    text: str
    claim: tuple
    role: str  # chain_f1 | chain_f2 | dangling_rel | dangling_prop
    chain: int | None = None


@dataclass
class Question:
    text: str
    answer: str
    kind: str  # chain | control
    chain: int


@dataclass
class World:
    seed: int
    sessions: list  # list[list[Fact]]
    questions: list  # list[Question]
    valid_derived: set = field(default_factory=set)  # true two-hop claims
    base_claims: set = field(default_factory=set)

    @property
    def n_chains(self) -> int:
        return len({q.chain for q in self.questions if q.kind == "chain"})

    def chain_questions(self):
        return [q for q in self.questions if q.kind == "chain"]

    def control_questions(self):
        return [q for q in self.questions if q.kind == "control"]


def _names(rng: random.Random, n: int, taken: set) -> list:
    out = []
    while len(out) < n:
        w = "".join(rng.choice(_SYL) for _ in range(rng.choice((2, 2, 3)))).capitalize()
        if len(w) >= 5 and w not in taken:
            taken.add(w)
            out.append(w)
    return out


def _decoy(rng: random.Random, name: str, taken: set) -> str:
    """A name that looks like `name` (same 4-letter stem) but is not it."""
    for _ in range(200):
        w = name[:4] + "".join(rng.choice(_SYL) for _ in range(1)).lower()
        w = w.capitalize() if w[0].islower() else w
        if w not in taken and len(w) >= 5:
            taken.add(w)
            return w
    return _names(rng, 1, taken)[0]


def generate(
    seed: int = 0,
    n_chains: int = 24,
    n_sessions: int = 5,
    n_dangling_rel: int = 12,
    n_dangling_prop: int = 12,
    n_controls: int = 12,
) -> World:
    capacity = sum(len(s.props) for s in SCHEMAS.values())
    if n_chains + n_dangling_prop > capacity:  # each property is used once, so answers stay unique
        raise ValueError(f"n_chains + n_dangling_prop must be at most {capacity}, got {n_chains + n_dangling_prop}")
    if n_sessions < 2:
        raise ValueError("n_sessions must be at least 2: a chain's two facts arrive in different sessions")
    if n_dangling_prop and not n_chains:
        raise ValueError("near-name decoys need at least one chain to imitate")
    rng = random.Random(seed)
    taken: set = set()
    schemas = list(SCHEMAS.values())
    sessions: list = [[] for _ in range(n_sessions)]
    questions: list = []
    valid: set = set()
    base: set = set()

    props_pool = {s.rel: rng.sample(s.props, len(s.props)) for s in schemas}
    ms: list = []  # (schema, M) for chain bridges, for decoys later

    for c in range(n_chains):
        sc = schemas[c % len(schemas)]
        a, m = _names(rng, 2, taken)
        p = props_pool[sc.rel].pop()
        f1 = Fact(sc.f1.format(A=a, M=m), (a, (sc.rel,), m), "chain_f1", c)
        f2 = Fact(sc.f2.format(M=m, P=p), (m, (PROPERTY,), p), "chain_f2", c)
        if rng.random() < 0.5:
            f1, f2 = f2, f1  # sometimes learn the property before the relation
        s1 = rng.randrange(0, n_sessions - 1)
        s2 = rng.randrange(s1 + 1, n_sessions)
        sessions[s1].append(f1)
        sessions[s2].append(f2)
        valid.add((a, (sc.rel, PROPERTY), p))
        base.add(f1.claim)
        base.add(f2.claim)
        questions.append(Question(sc.q_chain.format(A=a), p, "chain", c))
        ms.append((sc, m, c, p))

    # near-name decoys: a property fact about a *different* entity with a similar name
    for i in range(n_dangling_prop):
        sc, m, _, _ = ms[i % len(ms)]
        dm = _decoy(rng, m, taken)
        sc2 = sc
        if not props_pool[sc2.rel]:
            sc2 = next(s for s in schemas if props_pool[s.rel])
        p = props_pool[sc2.rel].pop()
        f = Fact(sc2.f2.format(M=dm, P=p), (dm, (PROPERTY,), p), "dangling_prop")
        sessions[rng.randrange(n_sessions)].append(f)
        base.add(f.claim)

    for i in range(n_dangling_rel):
        sc = schemas[i % len(schemas)]
        a, m = _names(rng, 2, taken)
        f = Fact(sc.f1.format(A=a, M=m), (a, (sc.rel,), m), "dangling_rel")
        sessions[rng.randrange(n_sessions)].append(f)
        base.add(f.claim)

    # single-hop controls: answerable from one stored fact; idle thinking must not hurt these
    for c in rng.sample(range(n_chains), min(n_controls, n_chains)):
        sc, m, _, p = ms[c]
        questions.append(Question(sc.q_direct.format(M=m), p, "control", c))

    for s in sessions:
        rng.shuffle(s)
    return World(seed, sessions, questions, valid, base)
