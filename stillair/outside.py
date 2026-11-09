"""Outside reading: Wikipedia, searched only for topics the person mentioned in the chat.

Only those short topics leave the computer. What comes back is raw material to
think about, and is never treated as instructions.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional

HEADERS = {"User-Agent": "StillAir/0.2 (student research prototype)", "Accept": "application/json"}
SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
MAX_CHARACTERS = 300
RESULTS_PER_SEARCH = 5


class Wikipedia:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def _get(self, url: str) -> dict:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=self.timeout) as r:
            return json.loads(r.read())

    def search(self, query: str, skip=frozenset()) -> list:
        """-> candidate pages not read yet: [{"title", "key", "excerpt"}]"""
        params = urllib.parse.urlencode({"q": query, "limit": RESULTS_PER_SEARCH})
        pages = self._get(f"{SEARCH_URL}?{params}").get("pages") or []
        return [{"title": p.get("title", ""), "key": p["key"], "excerpt": re.sub(r"<[^>]+>", "", p.get("excerpt") or "")}
                for p in pages if p.get("title") not in skip]

    def page(self, key: str) -> Optional[dict]:
        data = self._get(SUMMARY_URL + urllib.parse.quote(key, safe=""))
        text = str(data.get("extract") or "").strip()
        return {"title": str(data.get("title") or ""), "text": text[:MAX_CHARACTERS]} if text else None
