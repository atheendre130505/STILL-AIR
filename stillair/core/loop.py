"""The idle loop: perturb -> associate -> evaluate -> consolidate -> repeat.

    1. Perturb      draw a nudge, retrieve related memory (the seeds)
    2. Associate    the model produces a thought from the seeds
    3. Evaluate     selector + grounding score the thought
    4. Consolidate  keep -> write a connection; reinforce -> strengthen; discard -> fade
    5. Repeat       at a controlled rate and token budget

Every step is logged (the concept document's oversight requirement for version 1:
"log every thought and memory write"). The loop is thought-only: it reads and writes
memory and takes no external action.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from ..backends.base import LLM, Thought
from .memory import Memory, ntok
from .perturb import Perturbation
from .selector import Selector


@dataclass
class LoopConfig:
    seed_k: int = 4  # memories retrieved per nudge
    write_memory: bool = True  # False = ablation "remove memory writes"
    decay: float = 0.002  # strength lost per step by unreinforced connections
    prune_floor: float = 0.2
    habituation: float = 0.12  # inhibition of return: boredom with recently visited memories
    visit_decay: float = 0.9
    max_followups: int = 1  # lookups the model may issue per step
    skip_unlinked: bool = False  # no model call unless two seed facts share an entity
    trial_expiry: int = 0  # steps an unconfirmed connection may survive; 0 = no expiry


def seeds_share_entity(seeds: list) -> bool:
    seen = set()
    for n in seeds:
        if n.kind != "fact" or not n.claim:
            continue
        entities = {n.claim[0], n.claim[2]}
        if seen & entities:
            return True
        seen |= entities
    return False


class IdleLoop:
    def __init__(
        self,
        memory: Memory,
        llm: LLM,
        perturb: Perturbation,
        selector: Selector,
        cfg: Optional[LoopConfig] = None,
        log_path: Optional[str] = None,
        on_event: Optional[Callable] = None,
    ):
        self.mem, self.llm, self.perturb, self.selector = memory, llm, perturb, selector
        self.cfg = cfg or LoopConfig()
        self.step_no = 0
        self.tokens = 0
        self.visits: dict = {}
        self.events: list = []
        self.log_path = log_path
        self.on_event = on_event
        self.lock = threading.RLock()  # held while a step runs; task mode takes it to pause the loop

    def step(self) -> dict:
        with self.lock:
            self.step_no += 1
            nudge = self.perturb.draw(self.mem)
            boredom = (0.5 + self.perturb.curiosity) * self.cfg.habituation if self.perturb.use_state else 0.0
            penalty = {i: v * boredom for i, v in self.visits.items()}
            seeds = [n for _, n in self.mem.retrieve(nudge.query, k=self.cfg.seed_k, penalty=penalty)]
            for i in list(self.visits):
                self.visits[i] *= self.cfg.visit_decay
                if self.visits[i] < 0.05:
                    del self.visits[i]
            for s in seeds:
                self.visits[s.id] = self.visits.get(s.id, 0.0) + 1.0
            # follow the thread: the model may look up related memory before it thinks
            for q in self.llm.followups(seeds)[: self.cfg.max_followups]:
                for _, n in self.mem.retrieve(q, k=2):
                    if n not in seeds:
                        seeds.append(n)
            skipped = self.cfg.skip_unlinked and not seeds_share_entity(seeds)
            thought = Thought("none", "", None, [], 1) if skipped else self.llm.associate(seeds, nudge)
            dup = bool(thought.claim and self.mem.find_claim(thought.claim))
            dec = self.selector.evaluate(self.mem, thought)
            read = 0 if skipped or self.llm.usage_includes_input else sum(ntok(s.text) for s in seeds)
            toks = read + thought.tokens + dec.judge_tokens

            wrote = None
            if dec.action == "keep" and self.cfg.write_memory and thought.kind != "none":
                node = self.mem.add(
                    "connection", thought.text, thought.claim, "idle", thought.parents, self.step_no,
                    meta={"thought_kind": thought.kind},
                )
                wrote = node.id
                self.mem.self_model["kept"] += 1
            elif dec.action == "reinforce":
                if dec.existing is not None and self.cfg.write_memory:
                    self.mem.reinforce(dec.existing)
                self.mem.self_model["reinforced"] += 1
            elif dec.action == "discard":
                self.mem.self_model["discarded"] += 1

            accepted = wrote is not None
            self.perturb.update(self.mem.vec(thought.text) if thought.text else {}, accepted)
            pruned = self.mem.decay_and_prune(self.cfg.decay, self.cfg.prune_floor)
            if self.cfg.trial_expiry:
                pruned += self.mem.expire_unconfirmed(self.step_no, self.cfg.trial_expiry)
            self.tokens += toks

            ev = {
                "step": self.step_no,
                "source": nudge.source,
                "note": nudge.note,
                "seeds": [s.id for s in seeds],
                "kind": thought.kind,
                "text": thought.text,
                "claim": thought.claim,
                "parents": thought.parents,
                "dup": dup,
                "action": dec.action,
                "reason": dec.reason,
                "validity": round(dec.validity, 3),
                "novelty": round(dec.novelty, 3),
                "wrote": wrote,
                "pruned": pruned,
                "tokens": toks,
                "curiosity": round(self.perturb.curiosity, 3),
            }
            self.events.append(ev)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev) + "\n")
            if self.on_event:
                self.on_event(ev)
            return ev

    def run(self, steps: Optional[int] = None, token_budget: Optional[int] = None) -> int:
        """Run until `steps` steps or `token_budget` tokens are spent. Returns tokens used."""
        start = self.tokens
        n = 0
        while True:
            if steps is not None and n >= steps:
                break
            if token_budget is not None and self.tokens - start >= token_budget:
                break
            if steps is None and token_budget is None:
                break
            self.step()
            n += 1
        return self.tokens - start

