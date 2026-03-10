"""Metric helpers for benchmark reporting and regression checks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from dagzoo.bench.constants import MILLISECONDS_PER_SECOND
from dagzoo.math_utils import to_numpy as _to_numpy
from dagzoo.types import DatasetBundle

HIGHER_IS_BETTER_METRICS = frozenset(
    {
        "datasets_per_second",
        "datasets_per_minute",
        "generation_datasets_per_minute",
        "write_datasets_per_minute",
        "filter_datasets_per_minute",
        "filter_acceptance_rate_dataset_level",
    }
)
LOWER_IS_BETTER_METRICS = frozenset(
    {
        "elapsed_seconds",
        "latency_mean_ms",
        "latency_p95_ms",
        "peak_rss_mb",
        "peak_cuda_allocated_mb",
        "peak_cuda_reserved_mb",
        "filter_rejection_rate_dataset_level",
        "filter_rejection_rate_attempt_level",
        "filter_retry_dataset_rate",
        "retry_dataset_rate",
        "mean_attempts_per_dataset",
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


def _update_digest(digest: Any, *values: object) -> None:
    """Append normalized values to one digest."""

    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"|")


def reproducibility_signatures(bundles: Iterable[DatasetBundle]) -> tuple[str, str]:
    """Build content and workload digests for a sequence or stream of bundles."""

    content = hashlib.blake2s(digest_size=16)
    workload = hashlib.blake2s(digest_size=16)
    for bundle in bundles:
        for arr in (bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test):
            np_arr = _to_numpy(arr)
            _update_digest(content, np_arr.shape, np_arr.dtype)
            content.update(np.ascontiguousarray(np_arr).tobytes())
            _update_digest(workload, np_arr.shape, np_arr.dtype)

        for ft in bundle.feature_types:
            _update_digest(content, ft)
            _update_digest(workload, ft)

        seed = bundle.metadata.get("seed")
        dataset_seed = bundle.metadata.get("dataset_seed")
        dataset_index = bundle.metadata.get("dataset_index")
        attempt = bundle.metadata.get("attempt_used")
        _update_digest(content, seed, dataset_seed, dataset_index, attempt)
        _update_digest(
            workload,
            bundle.metadata.get("layout_signature"),
            bundle.metadata.get("layout_plan_signature"),
            bundle.metadata.get("n_features"),
            bundle.metadata.get("n_categorical_features"),
            bundle.metadata.get("n_classes"),
            bundle.metadata.get("graph_nodes"),
            bundle.metadata.get("graph_edges"),
            bundle.metadata.get("graph_depth_nodes"),
            dataset_index,
            attempt,
        )
        noise_distribution = bundle.metadata.get("noise_distribution")
        if isinstance(noise_distribution, dict):
            _update_digest(
                workload,
                noise_distribution.get("family_sampled"),
                noise_distribution.get("family_requested"),
            )
        else:
            _update_digest(workload, None)

    return content.hexdigest(), workload.hexdigest()


def reproducibility_signature(bundles: Iterable[DatasetBundle]) -> str:
    """Build a deterministic content digest for a sequence or stream of bundles."""

    content, _ = reproducibility_signatures(bundles)
    return content


def reproducibility_workload_signature(bundles: Iterable[DatasetBundle]) -> str:
    """Build a workload-shape digest for a sequence or stream of bundles."""

    _, workload = reproducibility_signatures(bundles)
    return workload
