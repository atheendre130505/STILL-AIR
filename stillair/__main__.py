"""Run with: python -m stillair"""

import argparse
import os
import sys

from .server import chat


def _load_dotenv(path: str = ".env") -> None:
    """Read STILLAIR_* settings from a gitignored .env file; real environment variables win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and key.startswith("STILLAIR_"):
                os.environ.setdefault(key, value.strip().strip("\"'"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="stillair", description=__doc__)
    parser.add_argument("command", nargs="?", default="chat", choices=["chat"])
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--memory", default="stillair_memory.json", help="where the conversation and memories are saved")
    args = parser.parse_args()
    chat(args.port, args.memory)


main()
