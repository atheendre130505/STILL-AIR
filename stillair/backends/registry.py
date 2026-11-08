"""Pick a backend by name. Real backends import lazily, so the simulator needs nothing installed."""

from __future__ import annotations

BACKENDS = ("sim", "claude", "openai")


def make_backend(name: str):
    """-> a real backend, or None for the simulator (callers build seeded SimLLMs themselves)."""
    if name == "sim":
        return None
    if name == "claude":
        from .claude import ClaudeLLM

        return ClaudeLLM()
    if name == "openai":
        from .openai_compat import OpenAICompatLLM

        return OpenAICompatLLM()
    raise ValueError(name)
