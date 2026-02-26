"""Metric helpers for benchmark reporting and regression checks."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np

from cauchy_generator.bench.constants import MILLISECONDS_PER_SECOND
from cauchy_generator.math_utils import to_numpy as _to_numpy
from cauchy_generator.types import DatasetBundle

HIGHER_IS_BETTER_METRICS = frozenset({"datasets_per_second", "datasets_per_minute"})
LOWER_IS_BETTER_METRICS = frozenset(
    {
        "elapsed_seconds",
        "latency_mean_ms",
        "latency_p95_ms",
        "peak_rss_mb",
        "peak_cuda_allocated_mb",
        "peak_cuda_reserved_mb",
    }
)


def percent_change(current: float, baseline: float) -> float | None:
    """Return percent change from baseline to current, or ``None`` for invalid baselines."""

    if not math.isfinite(current) or not math.isfinite(baseline) or baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0


def degradation_percent(metric: str, current: float, baseline: float) -> float | None:
    """Return positive percentage when performance degrades for the given metric direction."""

    change = percent_change(current, baseline)
    if change is None:
        return None

    if metric in HIGHER_IS_BETTER_METRICS:
        return -change
    if metric in LOWER_IS_BETTER_METRICS:
        return change
    return None


def summarize_latencies(latencies_seconds: Iterable[float]) -> dict[str, float]:
    """Summarize per-dataset latency samples in milliseconds."""

    values = np.asarray(list(latencies_seconds), dtype=np.float64)
    if values.size == 0:
        return {
            "latency_samples": 0.0,
            "latency_mean_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    ms = values * MILLISECONDS_PER_SECOND
    return {
        "latency_samples": float(ms.size),
        "latency_mean_ms": float(np.mean(ms)),
        "latency_p95_ms": float(np.percentile(ms, 95.0)),
        "latency_min_ms": float(np.min(ms)),
        "latency_max_ms": float(np.max(ms)),
    }


def reproducibility_signature(bundles: Iterable[DatasetBundle]) -> str:
    """Build a deterministic digest for a sequence or stream of dataset bundles."""

    h = hashlib.blake2s(digest_size=16)
    for bundle in bundles:
        for arr in (bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test):
            np_arr = _to_numpy(arr)
            h.update(str(np_arr.shape).encode("utf-8"))
            h.update(str(np_arr.dtype).encode("utf-8"))
            h.update(np.ascontiguousarray(np_arr).tobytes())

        for ft in bundle.feature_types:
            h.update(ft.encode("utf-8"))
            h.update(b"|")

        seed = bundle.metadata.get("seed")
        attempt = bundle.metadata.get("attempt_used")
        h.update(str(seed).encode("utf-8"))
        h.update(b":")
        h.update(str(attempt).encode("utf-8"))
        h.update(b";")

    return h.hexdigest()
