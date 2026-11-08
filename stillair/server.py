"""Starts the models and serves the chat page and the backend view."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import conscious, subconscious
from .links import Embedder
from .model import Model
from .outside import Wikipedia
from .session import LIMIT_CHOICES, ChatSession

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
PAGES = {"/": "chat.html", "/backend": "backend.html"}
GROQ_URL = "https://api.groq.com/openai/v1"
LOCAL_URL = "http://127.0.0.1:11434/v1"
OLLAMA_EXE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "StillAir", "ollama", "app", "ollama.exe")


def build_models() -> tuple:
    """-> (large model on the API, small model on local Ollama, local embedder)"""
    key = os.environ.get("STILLAIR_API_KEY", "")
    base_url = os.environ.get("STILLAIR_BASE_URL", GROQ_URL)
    if base_url == GROQ_URL and not key:
        raise SystemExit("Set STILLAIR_API_KEY in a .env file in this folder. See README.md.")
    local = os.environ.get("STILLAIR_LOCAL_URL", LOCAL_URL)
    large = Model(os.environ.get("STILLAIR_MAIN_MODEL", "openai/gpt-oss-120b"), base_url, conscious.SYSTEM, api_key=key)
    small = Model(os.environ.get("STILLAIR_SMALL_MODEL", "phi4-mini"), local, subconscious.SYSTEM, timeout=600.0)
    return large, small, Embedder(os.environ.get("STILLAIR_EMBED_MODEL", "all-minilm"), local)


def _local_server_up(version_url: str) -> bool:
    try:
        with urllib.request.urlopen(version_url, timeout=2):
            return True
    except OSError:
        return False


def _start_local_server():
    """Start the bundled Ollama if nothing is serving yet. -> the process we started, or None."""
    version_url = os.environ.get("STILLAIR_LOCAL_URL", LOCAL_URL).rsplit("/v1", 1)[0] + "/api/version"
    if _local_server_up(version_url):
        return None
    if not os.path.isfile(OLLAMA_EXE):
        print("warning: Ollama is not running, so the subconscious cannot think until it is started.")
        return None
    process = subprocess.Popen([OLLAMA_EXE, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if _local_server_up(version_url):
            break
        time.sleep(0.5)
    return process


def make_server(port: int, session: ChatSession) -> ThreadingHTTPServer:
    def toggle(data):
        if "conscious" in data and not isinstance(data["conscious"], bool):
            return {"error": "conscious must be true or false"}, 400
        if "limit_minutes" in data and data["limit_minutes"] not in LIMIT_CHOICES:
            return {"error": f"limit_minutes must be one of {LIMIT_CHOICES}"}, 400
        session.configure(data.get("limit_minutes", ...), data.get("conscious"))
        return {"ok": True}, 200

    def say(data):
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"error": "text must be a non-empty string"}, 400
        try:
            return {"reply": session.say(text.strip())}, 200
        except Exception as e:
            return {"error": str(e)}, 502

    def open_chat(data):
        chat_id = data.get("id")
        if isinstance(chat_id, bool) or not isinstance(chat_id, int):
            return {"error": "id must be an integer"}, 400
        return ({"ok": True}, 200) if session.open_chat(chat_id) else ({"error": "could not open that chat"}, 409)

    def unless_replying(action):
        return lambda data: ({"ok": True}, 200) if action() else ({"error": "a reply is in progress"}, 409)

    routes = {
        "/api/say": say, "/api/subconscious": toggle, "/api/chat/open": open_chat,
        "/api/chat/new": unless_replying(session.new_chat), "/api/chat/clear": unless_replying(session.clear_chat),
        "/api/memory/clear": unless_replying(session.clear_memory),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                return self._json(session.snapshot())
            if path == "/api/backend":
                return self._json(session.backend_snapshot())
            if path in PAGES:
                with open(os.path.join(WEB, PAGES[path]), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._json({"error": "bad request"}, 400)
            route = routes.get(self.path.split("?", 1)[0])
            if route is None:
                return self.send_error(404)
            if not isinstance(data, dict):
                return self._json({"error": "body must be a JSON object"}, 400)
            self._json(*route(data))

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def chat(port: int, path: str) -> None:
    large, small, embedder = build_models()
    ollama = _start_local_server()
    session = ChatSession(large, small, embedder, path, outside=Wikipedia())
    srv = make_server(port, session)
    print(f"Still Air chat: http://127.0.0.1:{srv.server_address[1]}   (Ctrl-C to stop)")
    print(f"conscious: {large.name}   subconscious: {small.name} (local)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        session.close()
        if ollama is not None:
            ollama.terminate()
