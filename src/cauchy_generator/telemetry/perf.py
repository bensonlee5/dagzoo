"""Low-overhead runtime performance telemetry collection."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
import math
import time
from typing import Any

_ACTIVE_PERF_TELEMETRY: contextvars.ContextVar[PerfTelemetry | None] = contextvars.ContextVar(
    "active_perf_telemetry",
    default=None,
)


@dataclass(slots=True)
class PerfTelemetry:
    """Collect run-level timers and counters for performance diagnostics."""

    enabled: bool = True
    timers_seconds: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)

    def add_time(self, key: str, seconds: float) -> None:
        """Accumulate one timer duration in seconds."""

        if not self.enabled:
            return
        seconds_value = float(seconds)
        if not math.isfinite(seconds_value) or seconds_value < 0.0:
            return
        self.timers_seconds[key] = float(self.timers_seconds.get(key, 0.0)) + seconds_value

    def increment(self, key: str, value: float = 1.0) -> None:
        """Accumulate one numeric counter."""

        if not self.enabled:
            return
        increment_value = float(value)
        if not math.isfinite(increment_value):
            return
        self.counters[key] = float(self.counters.get(key, 0.0)) + increment_value

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        """Context manager for recording one timed section."""

        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(key, time.perf_counter() - start)

    def snapshot(self) -> dict[str, Any]:
        """Build JSON-safe telemetry payload."""

        if not self.enabled:
            return {"enabled": False, "timers_ms": {}, "counters": {}}
        timers_ms = {
            key: float(value * 1000.0)
            for key, value in sorted(self.timers_seconds.items(), key=lambda item: item[0])
        }
        counters = {
            key: float(value)
            for key, value in sorted(self.counters.items(), key=lambda item: item[0])
        }
        return {"enabled": True, "timers_ms": timers_ms, "counters": counters}


def get_active_perf_telemetry() -> PerfTelemetry | None:
    """Return active performance telemetry context, when set."""

    return _ACTIVE_PERF_TELEMETRY.get()


@contextmanager
def activate_perf_telemetry(telemetry: PerfTelemetry | None) -> Iterator[None]:
    """Temporarily set active performance telemetry for nested calls."""

    token = _ACTIVE_PERF_TELEMETRY.set(telemetry)
    try:
        yield
    finally:
        _ACTIVE_PERF_TELEMETRY.reset(token)
