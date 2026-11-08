"""One client for any OpenAI-compatible chat API: Groq for the large model, local Ollama for the small one."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

MAX_RETRIES = 3
MAX_WAIT_SECONDS = 60.0  # a longer Retry-After means a daily quota is gone; waiting would just hang
RETRYABLE = (429, 500, 502, 503, 504)


class Model:
    def __init__(self, name: str, base_url: str, system: str, api_key: str = "", timeout: float = 120.0,
                 max_consecutive_errors: int = 5, sleep=time.sleep):
        self.name, self.base_url, self.system, self.api_key = name, base_url.rstrip("/"), system, api_key
        self.timeout, self.max_consecutive_errors, self.sleep = timeout, max_consecutive_errors, sleep
        self.use_json_schema = True  # dropped for good if the provider rejects it for this model
        self.consecutive_errors = 0
        self.tokens_used = 0
        self.calls = 0

    def ask_json(self, prompt: str, schema: dict, max_tokens: int = 1024):
        """One structured call. -> (parsed reply or None, tokens used). A lone failure counts as no answer;
        `max_consecutive_errors` in a row re-raise, so a bad key or an outage is not hidden."""
        self.calls += 1
        try:
            text, used = self._request(prompt, schema, max_tokens)
        except Exception as e:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise
            if self.consecutive_errors == 1:
                print(f"warning: model call failed, treating it as empty: {e}", file=sys.stderr)
            return None, 0
        self.consecutive_errors = 0
        self.tokens_used += used
        try:
            return (json.loads(text) if text is not None else None), used
        except json.JSONDecodeError:
            return None, used

    def _body(self, prompt: str, schema: dict, max_tokens: int) -> dict:
        if self.use_json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        else:
            response_format = {"type": "json_object"}
            prompt += "\n\nReply with one JSON object matching this schema:\n" + json.dumps(schema)
        return {"model": self.name, "max_tokens": max_tokens, "response_format": response_format,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]}

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "stillair/0.2"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _retry_wait(error: urllib.error.HTTPError, attempt: int) -> float:
        try:
            return float(error.headers.get("Retry-After"))
        except (TypeError, ValueError):
            return 2.0 * 2 ** attempt

    def _request(self, prompt: str, schema: dict, max_tokens: int):
        attempt = 0
        while True:
            try:
                data = self._post(self._body(prompt, schema, max_tokens))
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
        if choice.get("finish_reason") == "content_filter":
            return None, used
        return (choice.get("message") or {}).get("content"), used
