"""Per-vehicle job queue: serializes fusion so concurrent devices can't corrupt.

Fusion is read-modify-write on one asset. Five cameras uploading the same car at
once would interleave into a lost-update race — one device's evidence silently
overwritten by another's stale copy. So work for a given vehicle runs strictly
one job at a time, while different vehicles proceed in parallel.

In-process threads are the right size for LAN use (one laptop, a few devices).
The swap point for Redis/RQ is `submit` — the API only awaits a result, so
distributing the workers later changes nothing above this line.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    job_id: str
    vehicle: str
    future: Future
    kind: str = "fusion"


@dataclass
class QueueStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    active: dict = field(default_factory=dict)


class VehicleQueue:
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self.stats = QueueStats()

    def _lock_for(self, vehicle: str) -> threading.Lock:
        with self._registry_lock:
            return self._locks.setdefault(vehicle, threading.Lock())

    def submit(self, vehicle: str, fn: Callable[[], object], kind: str = "fusion") -> Job:
        """Queue `fn` for `vehicle`; it runs once no other job for it is running."""
        job_id = uuid.uuid4().hex[:12]
        lock = self._lock_for(vehicle)

        def run():
            with lock:  # serializes per vehicle; other vehicles run concurrently
                self.stats.active[vehicle] = job_id
                try:
                    result = fn()
                    self.stats.completed += 1
                    return result
                except Exception:
                    self.stats.failed += 1
                    traceback.print_exc()
                    raise
                finally:
                    self.stats.active.pop(vehicle, None)

        self.stats.submitted += 1
        return Job(job_id=job_id, vehicle=vehicle, future=self._executor.submit(run), kind=kind)

    def run_sync(self, vehicle: str, fn: Callable[[], object], timeout: float | None = None):
        """Submit and wait — what the HTTP ingest path uses."""
        return self.submit(vehicle, fn).future.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
