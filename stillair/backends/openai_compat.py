"""Backend for any OpenAI-compatible chat API: free hosted tiers or a local Ollama. Standard library only.

Groq's free tier (the default):

    STILLAIR_API_KEY=...                    # in the environment, or in a gitignored .env file
    python3 -m stillair demo --backend openai

Local Ollama, no key needed:

    STILLAIR_BASE_URL=http://localhost:11434/v1
    STILLAIR_MODEL=qwen3.5:4b

STILLAIR_RPM and STILLAIR_TPM limit requests and tokens per minute (defaults 25 and 6000 with a key, unlimited without one).

STATUS: exercised only against a fake HTTP server in tests/. It has NOT been run against a real provider.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from .prompted import PromptedLLM

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_RPM_WITH_KEY = 25
DEFAULT_TPM_WITH_KEY = 6000
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 60.0  # a longer Retry-After means a daily quota is gone; waiting would just hang
RETRYABLE = (429, 500, 502, 503, 504)


class OpenAICompatLLM(PromptedLLM):
    name = "openai"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 rpm: Optional[float] = None, max_consecutive_errors: int = 5, timeout: float = 120.0, sleep=time.sleep,
                 tpm: Optional[float] = None):
        super().__init__(max_consecutive_errors)
        self.base_url = (base_url or os.environ.get("STILLAIR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("STILLAIR_API_KEY", "")
        if self.base_url == DEFAULT_BASE_URL and not self.api_key:
            raise ValueError("Set STILLAIR_API_KEY (or STILLAIR_BASE_URL for a local server such as Ollama).")
        self.model = model or os.environ.get("STILLAIR_MODEL") or DEFAULT_MODEL
        rpm = rpm if rpm is not None else float(os.environ.get("STILLAIR_RPM") or (DEFAULT_RPM_WITH_KEY if self.api_key else 0))
        self.min_interval = 60.0 / rpm if rpm else 0.0
        tpm = tpm if tpm is not None else float(os.environ.get("STILLAIR_TPM") or (DEFAULT_TPM_WITH_KEY if self.api_key else 0))
        self.seconds_per_token = 60.0 / tpm if tpm else 0.0
        self.timeout = timeout
        self.sleep = sleep
        self.use_json_schema = True  # dropped for good if the provider rejects it for this model
        self._pace_lock = threading.Lock()  # the dashboard calls from more than one thread
        self._next_slot = 0.0

    def _pace(self) -> None:
        with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self.min_interval
        if start > now:
            self.sleep(start - now)

    def _body(self, user: str, schema: dict, max_tokens: int) -> dict:
        if self.use_json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        else:
            response_format = {"type": "json_object"}
            user += "\n\nReply with one JSON object matching this schema:\n" + json.dumps(schema)
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
            "response_format": response_format,
        }

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "stillair/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _retry_wait(self, error: urllib.error.HTTPError, attempt: int) -> float:
        try:
            return float(error.headers.get("Retry-After"))
        except (TypeError, ValueError):
            return 2.0 * 2 ** attempt

    def _request(self, user: str, schema: dict, max_tokens: int):
        attempt = 0
        while True:
            self._pace()
            try:
                data = self._post(self._body(user, schema, max_tokens))
                break
            except urllib.error.HTTPError as e:
                with e:
                    detail = e.read().decode("utf-8", "replace")
                if e.code == 400 and self.use_json_schema and ("response_format" in detail or "json_schema" in detail):
                    self.use_json_schema = False
                    continue
                if e.code == 400 and "json_validate_failed" in detail and attempt < MAX_RETRIES:
                    attempt += 1  # the model produced malformed JSON this time; ask again
                    continue
                wait = self._retry_wait(e, attempt)
                if e.code not in RETRYABLE or attempt >= MAX_RETRIES or wait > MAX_WAIT_SECONDS:
                    raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {detail[:300]}") from None
                attempt += 1
                self.sleep(wait)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        used = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        with self._pace_lock:  # a big call pushes the next one back, keeping us under the per-minute token limit
            self._next_slot = max(self._next_slot, time.monotonic() + used * self.seconds_per_token)
        if choice.get("finish_reason") == "content_filter":
            return None, used
        return (choice.get("message") or {}).get("content"), used
