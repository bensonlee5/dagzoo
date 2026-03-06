"""Benchmark guardrail helpers for lineage and missingness checks."""

from __future__ import annotations

import math
import pickle
import statistics
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dagzoo.bench.constants import (
    LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE,
    LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE_SMOKE,
    LINEAGE_GUARDRAIL_RUNTIME_TRIALS,
    LINEAGE_GUARDRAIL_SEED_OFFSET,
    LINEAGE_GUARDRAIL_SMOKE_SAMPLE_CAP,
    SECONDS_PER_MINUTE,
)
from dagzoo.bench.metrics import degradation_percent
from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core.parallel_generation import generate_parallel_batch_iter
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.math_utils import to_numpy as _to_numpy
from dagzoo.rng import offset_seed32
from dagzoo.types import DatasetBundle


def _lineage_guardrail_sample_count(config: GeneratorConfig, suite: str, num_datasets: int) -> int:
    """Choose dataset count used for lineage overhead guardrail checks."""

    n = max(1, min(int(config.benchmark.latency_num_samples), num_datasets))
    if suite == "smoke":
        return min(n, LINEAGE_GUARDRAIL_SMOKE_SAMPLE_CAP)
    return n


def _bundle_without_lineage(bundle: DatasetBundle) -> DatasetBundle:
    """Clone one bundle while removing lineage metadata for control persistence runs."""

    metadata = dict(bundle.metadata)
    metadata.pop("lineage", None)
    return DatasetBundle(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        feature_types=list(bundle.feature_types),
        metadata=metadata,
    )


def _measure_persistence_datasets_per_minute(
    bundles: Iterable[DatasetBundle],
    *,
    config: GeneratorConfig,
    num_bundles: int,
) -> float:
    """Measure write-path throughput for a fixed number of streamed bundles."""

    if num_bundles <= 0:
        return 0.0

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dagzoo_lineage_guardrail_") as tmp_dir:
        out_dir = Path(tmp_dir) / "shards"
        _ = write_packed_parquet_shards_stream(
            bundles,
            out_dir=out_dir,
            shard_size=max(1, int(config.output.shard_size)),
            compression=str(config.output.compression),
        )
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return 0.0
    return (float(num_bundles) / elapsed) * SECONDS_PER_MINUTE


def _stage_bundle(path: Path, bundle: DatasetBundle, *, strip_lineage: bool) -> None:
    """Serialize one bundle to disk for later replay in persistence timing."""

    staged_bundle = _bundle_without_lineage(bundle) if strip_lineage else bundle
    payload: dict[str, Any] = {
        "X_train": _to_numpy(staged_bundle.X_train),
        "y_train": _to_numpy(staged_bundle.y_train),
        "X_test": _to_numpy(staged_bundle.X_test),
        "y_test": _to_numpy(staged_bundle.y_test),
        "feature_types": list(staged_bundle.feature_types),
        "metadata": dict(staged_bundle.metadata),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _iter_staged_bundles(stage_dir: Path, *, num_bundles: int) -> Iterable[DatasetBundle]:
    """Replay staged bundles from disk as a streaming iterator."""

    for idx in range(max(0, num_bundles)):
        bundle_path = stage_dir / f"bundle_{idx:06d}.pkl"
        with bundle_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid staged bundle payload at {bundle_path}.")

        feature_types_raw = payload.get("feature_types")
        metadata_raw = payload.get("metadata")
        if not isinstance(feature_types_raw, list):
            raise ValueError(f"Staged feature_types must be a list at {bundle_path}.")
        if not isinstance(metadata_raw, dict):
            raise ValueError(f"Staged metadata must be an object at {bundle_path}.")
        feature_types = [str(value) for value in feature_types_raw]
        metadata = dict(metadata_raw)

        yield DatasetBundle(
            X_train=payload["X_train"],
            y_train=payload["y_train"],
            X_test=payload["X_test"],
            y_test=payload["y_test"],
            feature_types=feature_types,
            metadata=metadata,
        )


def _stage_lineage_trial_bundles(
    config: GeneratorConfig,
    *,
    sample_n: int,
    seed: int,
    device: str | None,
    baseline_stage_dir: Path,
    current_stage_dir: Path,
) -> tuple[int, int]:
    """Generate and stage baseline/current trial bundles once outside timed sections."""

    baseline_stage_dir.mkdir(parents=True, exist_ok=True)
    current_stage_dir.mkdir(parents=True, exist_ok=True)
    generator = (
        generate_parallel_batch_iter
        if int(config.runtime.worker_count) > 1
        else generate_batch_iter
    )

    seen = 0
    with_lineage = 0
    for idx, bundle in enumerate(
        generator(
            config,
            num_datasets=sample_n,
            seed=seed,
            device=device,
        )
    ):
        has_lineage = isinstance(bundle.metadata.get("lineage"), dict)
        if has_lineage:
            with_lineage += 1
        file_name = f"bundle_{idx:06d}.pkl"
        _stage_bundle(current_stage_dir / file_name, bundle, strip_lineage=False)
        _stage_bundle(baseline_stage_dir / file_name, bundle, strip_lineage=True)
        seen = idx + 1
    return with_lineage, seen


def _measure_lineage_persistence_trials(
    *,
    baseline_stage_dir: Path,
    current_stage_dir: Path,
    num_bundles: int,
    config: GeneratorConfig,
    trials: int,
) -> tuple[list[float], list[float]]:
    """Collect paired baseline/lineage persistence throughput trials."""

    n_trials = max(1, int(trials))
    baseline_trials: list[float] = []
    current_trials: list[float] = []
    for _ in range(n_trials):
        baseline_bundles = _iter_staged_bundles(baseline_stage_dir, num_bundles=num_bundles)
        baseline_trials.append(
            _measure_persistence_datasets_per_minute(
                baseline_bundles,
                config=config,
                num_bundles=num_bundles,
            )
        )
        current_bundles = _iter_staged_bundles(current_stage_dir, num_bundles=num_bundles)
        current_trials.append(
            _measure_persistence_datasets_per_minute(
                current_bundles,
                config=config,
                num_bundles=num_bundles,
            )
        )
    return baseline_trials, current_trials


def _median_throughput(values: list[float]) -> float:
    """Return robust median throughput across trial values."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(statistics.median(finite))


def _build_guardrail_issue(
    *,
    metric: str,
    severity: str,
    current: float | None,
    baseline: float | None,
    degradation_pct: float | None,
    detail: str,
) -> dict[str, Any]:
    """Create a normalized guardrail issue payload."""

    return {
        "metric": metric,
        "severity": severity,
        "current": current,
        "baseline": baseline,
        "degradation_pct": degradation_pct,
        "detail": detail,
    }


def _severity_from_thresholds(value: float, *, warn: float, fail: float) -> str:
    """Map a numeric value to pass/warn/fail severity."""

    if value >= fail:
        return "fail"
    if value >= warn:
        return "warn"
    return "pass"


def _status_from_issues(issues: list[dict[str, Any]]) -> str:
    """Collapse per-metric issues into an overall status."""

    severities = {str(issue.get("severity", "pass")) for issue in issues}
    if "fail" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"


def _issue_sort_key(issue: dict[str, Any]) -> tuple[int, float]:
    """Sort regression issues by severity then descending degradation percentage."""

    severity = str(issue.get("severity", "warn"))
    rank = 0 if severity == "fail" else 1 if severity == "warn" else 2
    raw_degradation = issue.get("degradation_pct")
    if isinstance(raw_degradation, bool) or not isinstance(raw_degradation, (int, float)):
        return (rank, 0.0)
    degradation = float(raw_degradation)
    if not math.isfinite(degradation):
        return (rank, 0.0)
    return (rank, -degradation)


def _collect_guardrail_regression_issues(
    preset_results: list[dict[str, Any]],
    *,
    guardrail_key: str,
) -> list[dict[str, Any]]:
    """Flatten one guardrail type into regression issue payloads."""

    issues: list[dict[str, Any]] = []
    for result in preset_results:
        guardrails = result.get(guardrail_key)
        if not isinstance(guardrails, dict) or not bool(guardrails.get("enabled")):
            continue
        raw_issues = guardrails.get("issues")
        if not isinstance(raw_issues, list):
            continue
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            merged = dict(issue)
            merged["preset"] = str(result.get("preset_key"))
            issues.append(merged)
    return issues


def _collect_lineage_guardrails(
    config: GeneratorConfig,
    *,
    suite: str,
    num_datasets: int,
    device: str | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> dict[str, Any]:
    """Measure lineage export overhead and convert it into guardrail metrics."""
    sample_n = _lineage_guardrail_sample_count(config, suite=suite, num_datasets=num_datasets)
    if sample_n <= 0:
        return {"enabled": False}

    sample_seed = offset_seed32(config.seed, LINEAGE_GUARDRAIL_SEED_OFFSET)
    with tempfile.TemporaryDirectory(prefix="dagzoo_lineage_guardrail_stage_") as tmp_dir:
        stage_root = Path(tmp_dir)
        baseline_stage_dir = stage_root / "baseline"
        current_stage_dir = stage_root / "current"
        try:
            bundles_with_lineage, bundles_seen = _stage_lineage_trial_bundles(
                config,
                sample_n=sample_n,
                seed=sample_seed,
                device=device,
                baseline_stage_dir=baseline_stage_dir,
                current_stage_dir=current_stage_dir,
            )
        except Exception as exc:
            return {
                "enabled": False,
                "reason": "sampling_failed",
                "detail": str(exc),
            }
        if bundles_seen <= 0:
            return {"enabled": False}

        lineage_coverage_rate = float(bundles_with_lineage) / float(bundles_seen)

        try:
            baseline_trials, current_trials = _measure_lineage_persistence_trials(
                baseline_stage_dir=baseline_stage_dir,
                current_stage_dir=current_stage_dir,
                num_bundles=bundles_seen,
                config=config,
                trials=LINEAGE_GUARDRAIL_RUNTIME_TRIALS,
            )
        except Exception as exc:
            return {
                "enabled": False,
                "reason": "unavailable",
                "detail": str(exc),
            }
    baseline_dpm = _median_throughput(baseline_trials)
    current_dpm = _median_throughput(current_trials)

    runtime_degradation = degradation_percent("datasets_per_minute", current_dpm, baseline_dpm)
    runtime_degradation_value = (
        float(runtime_degradation) if runtime_degradation is not None else 0.0
    )
    runtime_severity = _severity_from_thresholds(
        runtime_degradation_value,
        warn=float(warn_threshold_pct),
        fail=float(fail_threshold_pct),
    )
    runtime_gating_min_sample = (
        LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE_SMOKE
        if suite == "smoke"
        else LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE
    )
    runtime_gating_enabled = sample_n >= runtime_gating_min_sample

    issues: list[dict[str, Any]] = []
    if bundles_with_lineage != bundles_seen:
        issues.append(
            _build_guardrail_issue(
                metric="lineage_metadata_coverage",
                severity="fail",
                current=float(lineage_coverage_rate),
                baseline=1.0,
                degradation_pct=float(max(0.0, (1.0 - lineage_coverage_rate) * 100.0)),
                detail="Lineage metadata must be present for all generated bundles.",
            )
        )
    if runtime_gating_enabled and runtime_severity != "pass":
        issues.append(
            _build_guardrail_issue(
                metric="lineage_export_runtime_degradation_pct",
                severity=runtime_severity,
                current=float(current_dpm),
                baseline=float(baseline_dpm),
                degradation_pct=float(runtime_degradation_value),
                detail=(
                    "Lineage-enabled shard persistence throughput regressed versus a "
                    "lineage-stripped control persistence run."
                ),
            )
        )

    return {
        "enabled": True,
        "sample_datasets": int(sample_n),
        "runtime_gating_enabled": bool(runtime_gating_enabled),
        "runtime_gating_min_sample_datasets": int(runtime_gating_min_sample),
        "runtime_gating_suppressed_reason": (
            None if runtime_gating_enabled else "insufficient_sample_size"
        ),
        "runtime_trials": int(max(1, LINEAGE_GUARDRAIL_RUNTIME_TRIALS)),
        "runtime_baseline_trials_dpm": baseline_trials,
        "runtime_with_lineage_trials_dpm": current_trials,
        "lineage_metadata_coverage_rate": float(lineage_coverage_rate),
        "runtime_baseline_datasets_per_minute": float(baseline_dpm),
        "runtime_with_lineage_datasets_per_minute": float(current_dpm),
        "runtime_degradation_pct": (
            float(runtime_degradation) if runtime_degradation is not None else None
        ),
        "runtime_warn_threshold_pct": float(warn_threshold_pct),
        "runtime_fail_threshold_pct": float(fail_threshold_pct),
        "issues": issues,
        "status": _status_from_issues(issues),
    }
