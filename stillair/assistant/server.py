"""HTTP server for the chat page."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..backends.openai_compat import OpenAICompatLLM
from .associations import OllamaEmbedder
from .session import LIMIT_CHOICES, ChatSession

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web", "chat.html")
BACKEND_PAGE = os.path.join(os.path.dirname(PAGE), "backend.html")
DEFAULT_MAIN_MODEL = "openai/gpt-oss-120b"
DEFAULT_SMALL_MODEL = "phi4-mini"
DEFAULT_EMBED_MODEL = "all-minilm"
LOCAL_URL = "http://127.0.0.1:11434/v1"
OLLAMA_EXE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "StillAir", "ollama", "app", "ollama.exe")


def build_models():
    """-> (main model on the hosted API, small model and embedding model on the local Ollama server)"""
    local_url = os.environ.get("STILLAIR_LOCAL_URL", LOCAL_URL)
    main = OpenAICompatLLM(model=os.environ.get("STILLAIR_MAIN_MODEL", DEFAULT_MAIN_MODEL))
    small = OpenAICompatLLM(model=os.environ.get("STILLAIR_SMALL_MODEL", DEFAULT_SMALL_MODEL), base_url=local_url, api_key="", timeout=600.0)
    return main, small, OllamaEmbedder(os.environ.get("STILLAIR_EMBED_MODEL", DEFAULT_EMBED_MODEL), local_url)


def make_server(port: int, session: ChatSession) -> ThreadingHTTPServer:
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
            if path in ("/", "/backend"):
                with open(PAGE if path == "/" else BACKEND_PAGE, "rb") as f:
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
            path = self.path.split("?", 1)[0]
            if not isinstance(data, dict):
                return self._json({"error": "body must be a JSON object"}, 400)
            if path == "/api/say":
                text = data.get("text")
                if not isinstance(text, str) or not text.strip():
                    return self._json({"error": "text must be a non-empty string"}, 400)
                try:
                    return self._json({"reply": session.say(text.strip())})
                except Exception as e:
                    return self._json({"error": str(e)}, 502)
            if path == "/api/chat/new":
                return self._json({"ok": True}) if session.new_chat() else self._json({"error": "a reply is in progress"}, 409)
            if path == "/api/chat/open":
                chat_id = data.get("id")
                if isinstance(chat_id, bool) or not isinstance(chat_id, int):
                    return self._json({"error": "id must be an integer"}, 400)
                return self._json({"ok": True}) if session.open_chat(chat_id) else self._json({"error": "could not open that chat"}, 409)
            if path == "/api/subconscious":
                if "on" in data and not isinstance(data["on"], bool):
                    return self._json({"error": "on must be true or false"}, 400)
                if "limit_minutes" in data and data["limit_minutes"] not in LIMIT_CHOICES:
                    return self._json({"error": f"limit_minutes must be one of {LIMIT_CHOICES}"}, 400)
                session.configure(data.get("on"), data.get("limit_minutes", ...))
                return self._json({"ok": True})
            self.send_error(404)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


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


def chat(port: int, path: str) -> None:
    main, small, embedder = build_models()
    ollama = _start_local_server()
    session = ChatSession(main, small, embedder, path)
    srv = make_server(port, session)
    print(f"Still Air chat: http://127.0.0.1:{srv.server_address[1]}   (Ctrl-C to stop)")
    print(f"conscious: {main.model}   subconscious: {small.model} (local)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        session.close()
        if ollama is not None:
            ollama.terminate()
