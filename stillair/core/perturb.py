"""Perturbation source: what nudges the idle loop (analogue: neural noise, neuromodulation,
interoception).

A nudge is a *query* into memory. It mixes two things:

    drive : where the system's own state points it (the focus of its last thoughts, or an
            open question it could not answer), weighted by (1 - amplitude)
    noise : a random jump to unrelated memories, weighted by amplitude

amplitude=0  -> pure drift: the loop keeps circling the same material (rumination)
amplitude=1  -> pure noise: random jumps with no direction (what condition B uses)

Internal state variables drift too: `curiosity` rises while nothing useful is found and
falls after a success; `uncertainty` grows with the number of open questions. Both bias
the drive towards open questions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .memory import Memory, mix


@dataclass
class Nudge:
    query: dict
    source: str  # noise | drift | open_question
    amplitude: float
    note: str = ""


class Perturbation:
    EXPLORE_AMPLITUDE = 0.75  # adaptive mode: the amplitude used when not chasing an open question

    def __init__(self, rng: random.Random, amplitude: float = 0.5, use_state: bool = True, noise_nodes: int = 2, adaptive: bool = False):
        self.rng = rng
        self.amplitude = amplitude
        self.adaptive = adaptive  # focus fully on open questions, explore otherwise
        self.use_state = use_state
        self.noise_nodes = noise_nodes
        self.focus: dict = {}
        self.curiosity = 0.3
        self.uncertainty = 0.0

    def _noise(self, mem: Memory) -> dict:
        pool = mem.readable()
        if not pool:
            return {}
        picks = self.rng.sample(pool, min(self.noise_nodes, len(pool)))
        return mix([(1.0, mem.vec(n.text)) for n in picks])

    def draw(self, mem: Memory) -> Nudge:
        noise = self._noise(mem)
        if not self.use_state:
            return Nudge(noise, "noise", 1.0, "random jump")
        oq = mem.open_questions()
        self.uncertainty = min(1.0, len(oq) / 5.0)
        mem.self_model["curiosity"] = round(self.curiosity, 3)
        mem.self_model["uncertainty"] = round(self.uncertainty, 3)
        drive, source, note = self.focus, "drift", "continuing own focus"
        if oq and self.rng.random() < 0.6 * self.uncertainty + 0.3 * self.curiosity:
            q = self.rng.choice(oq)
            drive, source, note = mem.vec(q.text), "open_question", q.text
        if not drive:
            return Nudge(noise, "noise", 1.0, "no focus yet")
        a = self.amplitude
        if self.adaptive:
            a = 0.0 if source == "open_question" else self.EXPLORE_AMPLITUDE
        return Nudge(mix([(1 - a, drive), (a, noise)]), source, a, note)

    def seed(self, vec: dict) -> None:
        """Resume idle thinking seeded with what a task just surfaced."""
        self.focus = vec

    def update(self, thought_vec: dict, accepted: bool) -> None:
        if thought_vec:
            self.focus = mix([(0.6, self.focus), (0.4, thought_vec)]) if self.focus else thought_vec
        if accepted:
            self.curiosity = max(0.0, self.curiosity * 0.6)
        else:
            self.curiosity = min(1.0, self.curiosity + 0.05)
