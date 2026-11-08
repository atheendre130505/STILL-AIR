"""Build an idle loop for one experimental condition or ablation."""

from __future__ import annotations

import random
from typing import Callable, Optional

from ..backends.base import LLM
from ..backends.sim import SimLLM
from .loop import IdleLoop, LoopConfig
from .memory import Memory
from .perturb import Perturbation
from .selector import Selector


def build_loop(
    memory: Memory,
    seed: int = 0,
    mode: str = "full",
    amplitude: float = 0.5,
    hallucination_rate: float = 0.3,
    verifier_miss: float = 0.15,
    llm: Optional[LLM] = None,
    log_path: Optional[str] = None,
    on_event: Optional[Callable] = None,
    skip_unlinked: bool = False,
    trial_expiry: int = 0,
    adaptive_noise: bool = False,
) -> IdleLoop:
    """Factory for the conditions / ablations used in the paper.

    mode: full          - Still Air (condition C)
          noise_only    - random noise, no selector (condition B)
          no_selector   - full perturbation, accept everything (ablation)
          no_writes     - full, but nothing is written to memory (ablation)
          noise_selector- random noise, selector on (ablation)
    """
    llm = llm or SimLLM(seed + 101, hallucination_rate, verifier_miss)
    rng = random.Random(seed + 7)
    if mode == "noise_only":
        pert, sel, cfg = Perturbation(rng, 1.0, use_state=False), Selector(llm, enabled=False), LoopConfig()
    elif mode == "no_selector":
        pert, sel, cfg = Perturbation(rng, amplitude), Selector(llm, enabled=False), LoopConfig()
    elif mode == "no_writes":
        pert, sel, cfg = Perturbation(rng, amplitude), Selector(llm), LoopConfig(write_memory=False)
    elif mode == "noise_selector":
        pert, sel, cfg = Perturbation(rng, 1.0, use_state=False), Selector(llm), LoopConfig()
    elif mode == "full":
        pert, sel, cfg = Perturbation(rng, amplitude), Selector(llm), LoopConfig()
    else:
        raise ValueError(mode)
    cfg.skip_unlinked, cfg.trial_expiry = skip_unlinked, trial_expiry
    pert.adaptive = adaptive_noise
    return IdleLoop(memory, llm, pert, sel, cfg, log_path, on_event)
