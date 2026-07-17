"""Append-only event log — what the system did, and what it wants approved.

Two jobs:
  * tell the user what happened ("your rear scan was merged into bobs-civic");
  * hold pending duplicate merges awaiting a human decision when auto_merge is
    off, which is the testing mode this project defaults to.

JSONL on disk: survives restarts, trivially greppable, no database to run.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Event:
    kind: str                 # observation | merge | merge_pending | merge_rejected | error
    message: str
    vehicle: str = ""
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    data: dict = field(default_factory=dict)


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> Event:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        return event

    def all(self, limit: int | None = None) -> list[dict]:
        """Newest first."""
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        events.reverse()
        return events[:limit] if limit else events

    def pending_merges(self) -> list[dict]:
        """Flagged duplicates not yet approved or rejected."""
        resolved = set()
        pending: dict[str, dict] = {}
        for event in self.all():  # newest first
            if event["kind"] in ("merge", "merge_rejected"):
                resolved.add(event["data"].get("pending_id", ""))
            elif event["kind"] == "merge_pending":
                if event["event_id"] not in resolved:
                    pending[event["event_id"]] = event
        return list(pending.values())

    def find(self, event_id: str) -> dict | None:
        return next((e for e in self.all() if e["event_id"] == event_id), None)
