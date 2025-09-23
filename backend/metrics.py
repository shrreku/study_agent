"""Lightweight in-memory metrics collector for local/dev use.

Provides counters and simple timing histograms. This is intentionally
simple to avoid pulling heavyweight dependencies; production deployments
should replace this with Prometheus/OTel exporters.
"""
from typing import Dict, Any
import threading


class MetricsCollector:
    _global = None

    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, list] = {}

    @classmethod
    def get_global(cls) -> "MetricsCollector":
        if cls._global is None:
            cls._global = MetricsCollector()
        return cls._global

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def timing(self, name: str, ms: int) -> None:
        with self._lock:
            self.timers.setdefault(name, []).append(ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"counters": dict(self.counters), "timers": {k: list(v) for k, v in self.timers.items()}}


__all__ = ["MetricsCollector"]


