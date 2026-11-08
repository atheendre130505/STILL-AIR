"""Outside reading for the subconscious: short Wikipedia summaries, some on topic and some at random.

Only a few keywords ever leave the computer, never whole messages. What comes back is raw
material to think about, and is never treated as instructions.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional

HEADERS = {"User-Agent": "StillAir/0.2 (student research prototype)", "Accept": "application/json"}
SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
RANDOM_URL = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
MAX_CHARACTERS = 500


class Wikipedia:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def _get(self, url: str) -> dict:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=self.timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _reading(page: dict) -> Optional[dict]:
        text = str(page.get("extract") or "").strip()
        return {"title": str(page.get("title") or ""), "text": text[:MAX_CHARACTERS]} if text else None

    def about(self, keywords: list) -> Optional[dict]:
        query = urllib.parse.urlencode({"q": " ".join(keywords), "limit": 1})
        pages = self._get(f"{SEARCH_URL}?{query}").get("pages") or []
        if not pages:
            return None
        return self._reading(self._get(SUMMARY_URL + urllib.parse.quote(pages[0]["key"], safe="")))

    def random(self) -> Optional[dict]:
        return self._reading(self._get(RANDOM_URL))
