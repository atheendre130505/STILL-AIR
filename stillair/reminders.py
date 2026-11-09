"""Reminders: set by the large model while it replies, delivered by the clock with no model at all."""

from __future__ import annotations

from datetime import datetime

from .state import SharedState


def schedule_reminders(state: SharedState, items: list, now_epoch: float = None) -> int:
    """Store the proposed reminders, skipping unreadable, past or duplicate ones. -> how many were added."""
    now_epoch = datetime.now().timestamp() if now_epoch is None else now_epoch
    added = 0
    for item in items:
        try:
            at = datetime.fromisoformat(str(item["remind_at"]).strip().rstrip("Zz")).replace(tzinfo=None).timestamp()
        except (KeyError, TypeError, ValueError, OSError):
            continue
        text = str(item.get("text", "")).strip()
        duplicate = any(r["text"].lower() == text.lower() and not r["fired"] for r in state.reminders)
        if text and at > now_epoch and not duplicate:
            state.reminders.append({"text": text, "at": at, "fired": False})
            added += 1
    return added


def due_reminders(state: SharedState, now_epoch: float = None) -> list:
    """Every reminder whose time has come, including any missed while the app was closed. Each is handed out once."""
    now_epoch = datetime.now().timestamp() if now_epoch is None else now_epoch
    due = [r for r in state.reminders if not r["fired"] and r["at"] <= now_epoch]
    for reminder in due:
        reminder["fired"] = True
    return due


def with_connections(links, text: str, limit: int = 2) -> str:
    """Add what the subconscious connected to the memory this reminder is about."""
    found = links.search(text)
    with links.state.lock:
        about = next((n for _, n in found if n.kind == "fact"), None)
        if about is None:
            return text
        linked = [n for _, n in found if n.kind == "connection" and about.id in n.parents][:limit]
    return text + "".join(f"\nAlso: {n.text}" for n in linked)
