"""Tiny HTTP server for the dashboard: static page plus a JSON API over a LiveSession."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .session import BadRequest, LiveSession

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")


def _reset(live: LiveSession, data: dict) -> dict:
    seed = data.get("seed", live.seed)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BadRequest("seed must be an integer")
    live.reset(seed)
    return {"ok": True}


def _routes(live: LiveSession) -> dict:
    return {
        "feed": lambda data: live.feed(),
        "ask": lambda data: live.ask(data.get("question")),
        "config": live.config,
        "reset": lambda data: _reset(live, data),
    }


def make_server(port: int = 8765, seed: int = 0, llm=None, token_budget: int = 50_000):
    """Bind the dashboard server (port 0 picks a free port). Returns (server, session)."""
    live = LiveSession(seed, llm, token_budget)
    routes = _routes(live)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _path(self):
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def do_GET(self):
            path = self._path()
            if path == "/api/state":
                return self._json(live.snapshot())
            if path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            path = self._path()
            route = routes.get(path[len("/api/"):]) if path.startswith("/api/") else None
            try:  # read the body even for a bad route: closing with it unread resets the connection on Windows
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
            except ValueError:
                return self._json({"error": "bad length"}, 400)
            if route is None:
                return self.send_error(404)
            try:
                data = json.loads(body or b"{}")
            except ValueError:  # undecodable bytes or malformed JSON
                return self._json({"error": "bad json"}, 400)
            if not isinstance(data, dict):
                return self._json({"error": "body must be a JSON object"}, 400)
            try:
                return self._json(route(data))
            except BadRequest as e:
                return self._json({"error": str(e)}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler), live


def serve(port: int = 8765, seed: int = 0, llm=None, token_budget: int = 50_000):
    srv, _ = make_server(port, seed, llm, token_budget)
    print(f"Still Air dashboard: http://127.0.0.1:{srv.server_address[1]}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
