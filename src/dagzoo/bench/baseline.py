"""Baseline persistence and regression comparison for benchmark suites."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dagzoo.bench.metrics import degradation_percent

DEFAULT_GATING_METRICS = ("datasets_per_minute",)


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load a benchmark baseline JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Baseline payload must be a JSON object.")
    return payload


def write_baseline(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a benchmark baseline payload to disk."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path


def build_baseline_payload(
    suite_summary: dict[str, Any],
    *,
    metrics: Iterable[str] = DEFAULT_GATING_METRICS,
) -> dict[str, Any]:
    """Extract a compact baseline payload from a benchmark suite summary."""

    metric_names = list(metrics)
    presets: dict[str, dict[str, float]] = {}
    for result in suite_summary.get("preset_results", []):
        preset_key = str(result.get("preset_key"))
        metrics_payload: dict[str, float] = {}
        for metric in metric_names:
            value = result.get(metric)
            if isinstance(value, (int, float)):
                metrics_payload[metric] = float(value)
        presets[preset_key] = metrics_payload

    return {
        "version": 1,
        "suite": suite_summary.get("suite"),
        "metrics": metric_names,
        "presets": presets,
    }


def compare_summary_to_baseline(
    suite_summary: dict[str, Any],
    baseline_payload: dict[str, Any],
    *,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    metrics: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare suite results with baseline and return severity-ranked issues."""

    metric_names = (
        list(metrics) if metrics is not None else list(baseline_payload.get("metrics", []))
    )
    if not metric_names:
        metric_names = list(DEFAULT_GATING_METRICS)

    baseline_presets = baseline_payload.get("presets", {})
    issues: list[dict[str, Any]] = []

    for result in suite_summary.get("preset_results", []):
        preset_key = str(result.get("preset_key"))
        preset_baseline = baseline_presets.get(preset_key)
        if not isinstance(preset_baseline, dict):
            continue

        for metric in metric_names:
            cur_val = result.get(metric)
            base_val = preset_baseline.get(metric)
            if not isinstance(cur_val, (int, float)) or not isinstance(base_val, (int, float)):
                continue

            degr = degradation_percent(metric, float(cur_val), float(base_val))
            if degr is None or degr < warn_threshold_pct:
                continue

            severity = "fail" if degr >= fail_threshold_pct else "warn"
            issues.append(
                {
                    "preset": preset_key,
                    "metric": metric,
                    "current": float(cur_val),
                    "baseline": float(base_val),
                    "degradation_pct": float(degr),
                    "severity": severity,
                }
            )

    status = "pass"
    if any(issue["severity"] == "fail" for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"

    return {
        "status": status,
        "warn_threshold_pct": float(warn_threshold_pct),
        "fail_threshold_pct": float(fail_threshold_pct),
        "issues": sorted(
            issues,
            key=lambda issue: (issue["severity"] != "fail", -float(issue["degradation_pct"])),
        ),
    }
