"""Real-LLM backend (optional), driven by the Anthropic API.

    pip install anthropic
    export ANTHROPIC_API_KEY=...            # or `ant auth login`
    python3 -m stillair demo --backend claude

Model: `claude-opus-5-5` by default. Override with STILLAIR_MODEL (e.g. `claude-haiku-5-5` is far
cheaper for the many small calls an always-on loop makes).

STATUS: exercised only against fake clients in tests/. It has NOT been run against the live API
from the development machine (no credentials were available). No fallback model is configured.
"""

from __future__ import annotations

import os
from typing import Optional

from .prompted import PromptedLLM


class ClaudeLLM(PromptedLLM):
    name = "claude"

    def __init__(self, model: Optional[str] = None, client=None, effort: str = "low", max_consecutive_errors: int = 5):
        super().__init__(max_consecutive_errors)
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("STILLAIR_MODEL", "claude-opus-5-5")
        self.effort = effort

    def _request(self, user: str, schema: dict, max_tokens: int):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        used = getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0)
        if resp.stop_reason == "refusal":
            return None, used
        return next((b.text for b in resp.content if b.type == "text"), None), used
