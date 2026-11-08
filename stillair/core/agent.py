"""Task mode: answer a question from memory under a test-time budget.

The budget has two knobs: `max_hops` (how many retrieval rounds) and `k` (how many
memories are read per round). Tokens read + tokens generated are counted, so
"test-time tokens needed for equal accuracy" can be measured.

If the question cannot be answered, it is stored as an *open question*. The idle loop
treats open questions as drives, which is how a task seeds later idle thinking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..backends.base import LLM, Answer, Thought
from .memory import KINDS_READABLE, Memory, Node, ntok


@dataclass
class Trace:
    answer: Optional[str]
    hops: int
    tokens: int
    retrieved: list = field(default_factory=list)  # list of lists of node ids, one per hop
    via: list = field(default_factory=list)


class Agent:
    def __init__(
        self,
        memory: Memory,
        llm: LLM,
        k: int = 3,
        max_hops: int = 2,
        retrieval_noise: float = 0.0,
        rng: Optional[random.Random] = None,
        readonly: bool = True,
        kinds=KINDS_READABLE,
        trial_links: bool = False,
        verify_on_use: bool = False,
    ):
        self.mem, self.llm = memory, llm
        self.k, self.max_hops = k, max_hops
        self.noise = retrieval_noise  # models an imperfect embedding model
        self.rng = rng or random.Random(0)
        self.readonly = readonly
        self.kinds = kinds
        self.trial_links = trial_links  # links never re-derived cannot answer yet
        self.verify_on_use = verify_on_use  # re-check a link against its parents before trusting it

    def _connection_used(self, ans: Answer) -> Optional[Node]:
        for nid in ans.via:
            n = self.mem.nodes.get(nid)
            if n is not None and n.kind == "connection":
                return n
        return None

    def _link_holds(self, link: Node):
        """-> (does the link still check out against its parent facts?, tokens spent checking)"""
        parents = [self.mem.nodes[i] for i in link.parents if i in self.mem.nodes]
        kind = link.meta.get("thought_kind", "derive")
        validity, used = self.llm.judge(Thought(kind, link.text, link.claim, link.parents), parents)
        return validity >= 0.5, used

    def ask(self, question: str) -> Trace:
        scratch: dict = {}
        query = question
        tokens = ntok(question)
        tr = Trace(None, 0, 0)
        excluded = self.mem.unconfirmed() if self.trial_links else set()
        for hop in range(1, self.max_hops + 1):
            hits = self.mem.retrieve(query, self.k, self.kinds, self.noise, self.rng, exclude=excluded)
            items = [n for _, n in hits]
            tr.hops = hop
            tr.retrieved.append([n.id for n in items])
            if not self.llm.usage_includes_input:
                tokens += sum(ntok(n.text) for n in items)
            ans = self.llm.answer(question, items, scratch)
            tokens += ans.tokens
            while self.verify_on_use and ans.text is not None:
                link = self._connection_used(ans)
                if link is None:
                    break
                holds, used = self._link_holds(link)
                tokens += used
                if holds:
                    break
                # drop the failed link and fall back on what is left, including the two-hop path
                excluded.add(link.id)
                items = [n for n in items if n.id != link.id]
                scratch.get("ctx", {}).pop(link.id, None)
                ans = self.llm.answer(question, items, scratch)
                tokens += ans.tokens
            if ans.text is not None:
                tr.answer, tr.via = ans.text, ans.via
                break
            if ans.need is None:
                break
            query = ans.need  # second hop: look up the bridge entity
        tr.tokens = tokens
        if not self.readonly:
            for nid in tr.via:
                n = self.mem.nodes.get(nid)
                if n is not None:
                    self.mem.reinforce(n, 0.2)
            if tr.answer is None:
                if not any(q.text == question for q in self.mem.open_questions()):
                    self.mem.add("question", question, None, "user", meta={"status": "open"})
            else:
                for q in self.mem.open_questions():
                    if q.text == question:
                        q.meta["status"] = "closed"
        return tr
