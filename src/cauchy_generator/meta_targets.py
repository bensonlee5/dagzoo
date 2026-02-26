"""Shared helpers for meta-feature target specs and diagnostics bands."""

from __future__ import annotations

import math

WeightedTargetSpec = tuple[float, float, float]
TargetBand = tuple[float, float]


def coerce_weighted_target_specs(
    raw: object,
    *,
    supported_metrics: set[str] | frozenset[str],
) -> dict[str, WeightedTargetSpec]:
    """Normalize target payload into finite `(min, max, weight)` tuples."""

    normalized: dict[str, WeightedTargetSpec] = {}
    if not isinstance(raw, dict):
        return normalized
    for metric_name, value in raw.items():
        if not isinstance(metric_name, str) or metric_name not in supported_metrics:
            continue
        if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
            continue
        try:
            lo = float(value[0])
            hi = float(value[1])
            weight = float(value[2]) if len(value) == 3 else 1.0
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(weight)):
            continue
        if weight <= 0.0:
            continue
        normalized[metric_name] = (lo, hi, weight) if lo <= hi else (hi, lo, weight)
    return normalized


def merge_weighted_target_specs(
    *raw_values: object,
    supported_metrics: set[str] | frozenset[str],
) -> dict[str, WeightedTargetSpec]:
    """Merge weighted target spec payloads in order, where later entries win."""

    merged: dict[str, WeightedTargetSpec] = {}
    for raw in raw_values:
        merged.update(coerce_weighted_target_specs(raw, supported_metrics=supported_metrics))
    return merged


def collect_unknown_target_metrics(
    *raw_values: object,
    supported_metrics: set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Collect unsupported metric keys found in raw target payloads."""

    unknown: set[str] = set()
    for raw in raw_values:
        if not isinstance(raw, dict):
            continue
        for metric_name in raw:
            if isinstance(metric_name, str) and metric_name not in supported_metrics:
                unknown.add(metric_name)
    return tuple(sorted(unknown))


def validate_target_specs_for_task(
    *,
    task: str,
    target_specs: dict[str, WeightedTargetSpec],
    classification_only_metrics: set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Return metrics that are incompatible with the configured task."""

    if task.strip().lower() != "regression":
        return ()
    incompatible = sorted(set(target_specs).intersection(classification_only_metrics))
    return tuple(incompatible)


def weighted_specs_to_bands(
    target_specs: dict[str, WeightedTargetSpec],
) -> dict[str, TargetBand]:
    """Drop steering weights from target specs for diagnostics band payloads."""

    return {metric_name: (spec[0], spec[1]) for metric_name, spec in target_specs.items()}


def coerce_target_bands(raw: object) -> dict[str, TargetBand]:
    """Normalize target band mappings into finite `(lo, hi)` tuples."""

    normalized: dict[str, TargetBand] = {}
    if not isinstance(raw, dict):
        return normalized
    for metric_name, value in raw.items():
        if not isinstance(metric_name, str):
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        normalized[metric_name] = (lo, hi) if lo <= hi else (hi, lo)
    return normalized


def merge_target_bands(*raw_values: object) -> dict[str, TargetBand]:
    """Merge target bands in order, where later entries win."""

    merged: dict[str, TargetBand] = {}
    for raw in raw_values:
        merged.update(coerce_target_bands(raw))
    return merged


def coerce_quantiles(raw: object) -> tuple[float, ...]:
    """Normalize quantile payload into finite float values."""

    if not isinstance(raw, (list, tuple)):
        return ()
    normalized: list[float] = []
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            normalized.append(value)
    return tuple(normalized)
