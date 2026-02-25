"""Benchmark suite orchestration across runtime profiles."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import pickle
import re
import resource
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from cauchy_generator.bench.baseline import compare_summary_to_baseline
from cauchy_generator.bench.micro import run_microbenchmarks
from cauchy_generator.bench.metrics import (
    degradation_percent,
    reproducibility_signature,
    summarize_latencies,
)
from cauchy_generator.bench.throughput import run_throughput_benchmark
from cauchy_generator.config import GeneratorConfig, MISSINGNESS_MECHANISM_NONE
from cauchy_generator.core.dataset import generate_batch_iter, generate_one
from cauchy_generator.diagnostics import (
    CoverageAggregationConfig,
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from cauchy_generator.io.parquet_writer import write_parquet_shards_stream
from cauchy_generator.hardware import (
    HardwareInfo,
    apply_hardware_profile,
    detect_hardware,
)
from cauchy_generator.math_utils import to_numpy as _to_numpy
from cauchy_generator.rng import SeedManager
from cauchy_generator.types import DatasetBundle


DEFAULT_PROFILE_CONFIGS: dict[str, str] = {
    "cpu": "configs/benchmark_cpu.yaml",
    "cuda_desktop": "configs/benchmark_cuda_desktop.yaml",
    "cuda_h100": "configs/benchmark_cuda_h100.yaml",
}
MISSINGNESS_RATE_WARN_ABS_ERROR = 0.03
MISSINGNESS_RATE_FAIL_ABS_ERROR = 0.05
LINEAGE_GUARDRAIL_SMOKE_SAMPLE_CAP = 5
LINEAGE_GUARDRAIL_SEED_OFFSET = 123_000
LINEAGE_GUARDRAIL_RUNTIME_TRIALS = 3
LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE = 5


@dataclass(slots=True)
class ProfileRunSpec:
    """Benchmark execution spec for one profile/config pair."""

    key: str
    config: GeneratorConfig
    device: str | None = None


def _clone_config(config: GeneratorConfig) -> GeneratorConfig:
    """Clone nested benchmark config state to avoid in-place cross-profile mutations."""

    return GeneratorConfig.from_dict(config.to_dict())


def _peak_rss_mb() -> float:
    """Return process max resident set size in MiB."""

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _profile_counts(
    config: GeneratorConfig,
    *,
    profile_key: str,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
) -> tuple[int, int]:
    """Resolve benchmark dataset and warmup counts for a profile and suite mode."""

    profile_map = config.benchmark.profiles.get(profile_key, {})
    num = int(profile_map.get("num_datasets", config.benchmark.num_datasets))
    warmup = int(profile_map.get("warmup_datasets", config.benchmark.warmup_datasets))

    if num_datasets_override is not None:
        num = int(num_datasets_override)
    if warmup_override is not None:
        warmup = int(warmup_override)

    if suite == "smoke":
        num = min(num, 25)
        warmup = min(warmup, 5)

    return max(1, num), max(0, warmup)


def _latency_sample_count(config: GeneratorConfig, suite: str, num_datasets: int) -> int:
    """Choose per-profile latency sample count for the requested suite level."""

    n = max(1, min(int(config.benchmark.latency_num_samples), num_datasets))
    if suite == "smoke":
        return min(n, 5)
    return n


def _collect_latency(
    config: GeneratorConfig,
    *,
    device: str | None,
    num_samples: int,
) -> dict[str, float]:
    """Collect per-dataset latency samples by repeatedly calling ``generate_one``."""

    manager = SeedManager(config.seed + 77_000)
    samples: list[float] = []
    for i in range(max(1, num_samples)):
        seed = manager.child("latency", i)
        start = time.perf_counter()
        _ = generate_one(config, seed=seed, device=device)
        samples.append(time.perf_counter() - start)
    return summarize_latencies(samples)


def _collect_reproducibility(
    config: GeneratorConfig,
    *,
    device: str | None,
    num_datasets: int,
) -> dict[str, Any]:
    """Generate two deterministic runs and compare content digests."""

    n = max(1, num_datasets)
    run_seed = config.seed + 91_000
    sig_a = reproducibility_signature(
        generate_batch_iter(config, num_datasets=n, seed=run_seed, device=device)
    )
    sig_b = reproducibility_signature(
        generate_batch_iter(config, num_datasets=n, seed=run_seed, device=device)
    )
    return {
        "reproducibility_datasets": n,
        "reproducibility_signature": sig_a,
        "reproducibility_match": bool(sig_a == sig_b),
    }


def _prepare_config_for_profile(
    spec: ProfileRunSpec,
    *,
    suite: str,
    no_hardware_aware: bool,
) -> tuple[GeneratorConfig, str, HardwareInfo]:
    """Clone and hardware-tune a profile config before running benchmarks."""

    cfg = _clone_config(spec.config)
    if no_hardware_aware:
        cfg.runtime.hardware_aware = False

    requested_device = spec.device or cfg.runtime.device
    hw = detect_hardware(requested_device)
    cfg = apply_hardware_profile(cfg, hw)

    if suite == "smoke":
        cfg.dataset.n_train = min(cfg.dataset.n_train, 256)
        cfg.dataset.n_test = min(cfg.dataset.n_test, 128)
        cfg.dataset.n_features_min = min(cfg.dataset.n_features_min, 24)
        cfg.dataset.n_features_max = min(cfg.dataset.n_features_max, 24)
        cfg.graph.n_nodes_min = min(cfg.graph.n_nodes_min, 16)
        cfg.graph.n_nodes_max = min(cfg.graph.n_nodes_max, 16)

    return cfg, requested_device, hw


def _coerce_target_bands(raw: object) -> dict[str, tuple[float, float]]:
    """Normalize target band mappings into finite `(lo, hi)` tuples."""

    normalized: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return normalized
    for key, value in raw.items():
        if not isinstance(key, str):
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
        normalized[key] = (lo, hi) if lo <= hi else (hi, lo)
    return normalized


def _resolve_target_bands(config: GeneratorConfig) -> dict[str, tuple[float, float]]:
    """Merge diagnostics and top-level target mappings for coverage aggregation."""

    merged = _coerce_target_bands(config.diagnostics.meta_feature_targets)
    merged.update(_coerce_target_bands(config.meta_feature_targets))
    return merged


def _coerce_quantiles(raw: object) -> tuple[float, ...]:
    """Normalize diagnostics quantiles into finite values."""

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


def _sanitize_profile_key(profile_key: str) -> str:
    """Normalize profile key into a filesystem-safe unique path segment."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", profile_key).strip("._-")
    if not normalized:
        normalized = "profile"
    suffix = hashlib.sha1(profile_key.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}_{suffix}"


def _artifact_pointer(path: Path) -> str:
    """Return a summary-safe pointer for diagnostics artifacts."""

    return str(path.resolve())


def _build_diagnostics_aggregator(config: GeneratorConfig) -> CoverageAggregator:
    """Create a diagnostics coverage aggregator from config."""

    return CoverageAggregator(
        CoverageAggregationConfig(
            include_spearman=bool(config.diagnostics.include_spearman),
            histogram_bins=max(1, int(config.diagnostics.histogram_bins)),
            quantiles=_coerce_quantiles(config.diagnostics.quantiles),
            underrepresented_threshold=float(config.diagnostics.underrepresented_threshold),
            max_values_per_metric=config.diagnostics.max_values_per_metric,
            target_bands=_resolve_target_bands(config),
        )
    )


def _is_missingness_enabled(config: GeneratorConfig) -> bool:
    """Return whether missingness is enabled in config."""

    return bool(
        float(config.dataset.missing_rate) > 0.0
        and str(config.dataset.missing_mechanism).strip().lower() != MISSINGNESS_MECHANISM_NONE
    )


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
    with tempfile.TemporaryDirectory(prefix="cauchy_lineage_guardrail_") as tmp_dir:
        out_dir = Path(tmp_dir) / "shards"
        _ = write_parquet_shards_stream(
            bundles,
            out_dir=out_dir,
            shard_size=max(1, int(config.output.shard_size)),
            compression=str(config.output.compression),
        )
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return 0.0
    return (float(num_bundles) / elapsed) * 60.0


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

    seen = 0
    with_lineage = 0
    for idx, bundle in enumerate(
        generate_batch_iter(
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


def _build_lineage_guardrail_issue(
    *,
    metric: str,
    severity: str,
    current: float | None,
    baseline: float | None,
    degradation_pct: float | None,
    detail: str,
) -> dict[str, Any]:
    """Create a normalized lineage guardrail issue payload."""

    return {
        "metric": metric,
        "severity": severity,
        "current": current,
        "baseline": baseline,
        "degradation_pct": degradation_pct,
        "detail": detail,
    }


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

    sample_seed = int(config.seed + LINEAGE_GUARDRAIL_SEED_OFFSET)
    with tempfile.TemporaryDirectory(prefix="cauchy_lineage_guardrail_stage_") as tmp_dir:
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
    runtime_gating_enabled = sample_n >= LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE

    issues: list[dict[str, Any]] = []
    if bundles_with_lineage != bundles_seen:
        issues.append(
            _build_lineage_guardrail_issue(
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
            _build_lineage_guardrail_issue(
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
        "runtime_gating_min_sample_datasets": int(LINEAGE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE),
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


def _matrix_cell_count(matrix: Any) -> int:
    """Return cell count for a rank-2 matrix-like payload."""

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    try:
        n_rows = max(0, int(shape[0]))
        n_cols = max(0, int(shape[1]))
    except (TypeError, ValueError):
        return 0
    return n_rows * n_cols


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


@dataclass(slots=True)
class _MissingnessAcceptanceCollector:
    """Collect per-bundle missingness metadata for acceptance guardrails."""

    target_rate: float
    bundles_seen: int = 0
    bundles_with_metadata: int = 0
    missing_cells: int = 0
    total_cells: int = 0

    def update(self, bundle: DatasetBundle) -> None:
        """Collect missingness counters for one generated bundle."""

        self.bundles_seen += 1
        payload = bundle.metadata.get("missingness")
        if not isinstance(payload, dict):
            return

        total_cells = _matrix_cell_count(bundle.X_train) + _matrix_cell_count(bundle.X_test)
        if total_cells <= 0:
            return

        missing_count_raw = payload.get("missing_count_overall")
        if isinstance(missing_count_raw, bool) or not isinstance(missing_count_raw, (int, float)):
            return
        missing_count = int(max(0, min(total_cells, int(missing_count_raw))))

        self.bundles_with_metadata += 1
        self.total_cells += total_cells
        self.missing_cells += missing_count

    def build_summary(self) -> dict[str, Any]:
        """Build acceptance guardrail metrics and issues."""

        coverage_rate = (
            float(self.bundles_with_metadata) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )
        realized_rate = (
            float(self.missing_cells) / float(self.total_cells) if self.total_cells > 0 else 0.0
        )
        rate_abs_error = abs(realized_rate - float(self.target_rate))

        issues: list[dict[str, Any]] = []
        if self.bundles_with_metadata != self.bundles_seen:
            issues.append(
                {
                    "metric": "missingness_metadata_coverage",
                    "severity": "fail",
                    "current": float(coverage_rate),
                    "baseline": 1.0,
                    "degradation_pct": float(max(0.0, (1.0 - coverage_rate) * 100.0)),
                    "detail": "Missingness metadata must be present for all generated bundles.",
                }
            )

        rate_error_pp = rate_abs_error * 100.0
        rate_severity = _severity_from_thresholds(
            rate_abs_error,
            warn=MISSINGNESS_RATE_WARN_ABS_ERROR,
            fail=MISSINGNESS_RATE_FAIL_ABS_ERROR,
        )
        if rate_severity != "pass":
            threshold_pp = (
                MISSINGNESS_RATE_FAIL_ABS_ERROR * 100.0
                if rate_severity == "fail"
                else MISSINGNESS_RATE_WARN_ABS_ERROR * 100.0
            )
            issues.append(
                {
                    "metric": "missingness_realized_rate_error_pp",
                    "severity": rate_severity,
                    "current": float(rate_error_pp),
                    "baseline": float(threshold_pp),
                    "degradation_pct": float(rate_error_pp),
                    "detail": "Realized missing rate drifted from configured target.",
                }
            )

        return {
            "metadata_coverage_rate": float(coverage_rate),
            "realized_rate_overall": float(realized_rate),
            "rate_abs_error": float(rate_abs_error),
            "issues": issues,
            "status": _status_from_issues(issues),
        }


def _build_missingness_guardrail_issue(
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


def _collect_missingness_regression_issues(
    profile_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten per-profile missingness guardrail issues into regression issue payloads."""

    issues: list[dict[str, Any]] = []
    for result in profile_results:
        guardrails = result.get("missingness_guardrails")
        if not isinstance(guardrails, dict) or not bool(guardrails.get("enabled")):
            continue
        raw_issues = guardrails.get("issues")
        if not isinstance(raw_issues, list):
            continue
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            merged = dict(issue)
            merged["profile"] = str(result.get("profile_key"))
            issues.append(merged)
    return issues


def _collect_lineage_regression_issues(
    profile_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten per-profile lineage guardrail issues into regression issue payloads."""

    issues: list[dict[str, Any]] = []
    for result in profile_results:
        guardrails = result.get("lineage_guardrails")
        if not isinstance(guardrails, dict) or not bool(guardrails.get("enabled")):
            continue
        raw_issues = guardrails.get("issues")
        if not isinstance(raw_issues, list):
            continue
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            merged = dict(issue)
            merged["profile"] = str(result.get("profile_key"))
            issues.append(merged)
    return issues


def run_profile_benchmark(
    spec: ProfileRunSpec,
    *,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
    collect_memory: bool,
    collect_reproducibility: bool,
    include_micro: bool,
    no_hardware_aware: bool,
    collect_diagnostics: bool,
    diagnostics_root_dir: Path | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    diagnostics_occurrence_index: int,
    diagnostics_occurrence_total: int,
) -> dict[str, Any]:
    """Run one benchmark profile and collect throughput, latency, and optional diagnostics."""

    config, requested_device, hw = _prepare_config_for_profile(
        spec,
        suite=suite,
        no_hardware_aware=no_hardware_aware,
    )

    num_datasets, warmup = _profile_counts(
        config,
        profile_key=spec.key,
        suite=suite,
        num_datasets_override=num_datasets_override,
        warmup_override=warmup_override,
    )

    rss_before = _peak_rss_mb() if collect_memory else 0.0
    if collect_memory and hw.backend == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    diagnostics_enabled = bool(collect_diagnostics and diagnostics_root_dir is not None)
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_aggregator = _build_diagnostics_aggregator(config)

    missingness_enabled = _is_missingness_enabled(config)
    missingness_acceptance = (
        _MissingnessAcceptanceCollector(target_rate=float(config.dataset.missing_rate))
        if missingness_enabled
        else None
    )

    on_bundle_callback = None
    if diagnostics_aggregator is not None or missingness_acceptance is not None:

        def _on_bundle(bundle: DatasetBundle) -> None:
            if diagnostics_aggregator is not None:
                diagnostics_aggregator.update_bundle(bundle)
            if missingness_acceptance is not None:
                missingness_acceptance.update(bundle)

        on_bundle_callback = _on_bundle

    result = run_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        device=requested_device,
        on_bundle=on_bundle_callback,
    )
    result["profile_key"] = spec.key
    result["suite"] = suite
    result["device"] = requested_device
    result["hardware_backend"] = hw.backend
    result["hardware_device_name"] = hw.device_name
    result["hardware_memory_gb"] = hw.total_memory_gb
    result["hardware_peak_flops"] = hw.peak_flops
    result["hardware_profile"] = hw.profile
    result["diagnostics_enabled"] = diagnostics_enabled
    result["diagnostics_artifacts"] = None
    result["missingness_guardrails"] = {"enabled": False}
    result["lineage_guardrails"] = {"enabled": False}

    latency_stats = _collect_latency(
        config,
        device=requested_device,
        num_samples=_latency_sample_count(config, suite, num_datasets),
    )
    result.update(latency_stats)

    if collect_memory:
        result["peak_rss_mb"] = max(0.0, _peak_rss_mb() - rss_before)
        if hw.backend == "cuda" and torch.cuda.is_available():
            try:
                result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024.0**2)
                result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024.0**2)
            except Exception:
                result["peak_cuda_allocated_mb"] = None
                result["peak_cuda_reserved_mb"] = None

    if collect_reproducibility:
        repro_n = min(num_datasets, max(1, int(config.benchmark.reproducibility_num_datasets)))
        result.update(
            _collect_reproducibility(
                config,
                device=requested_device,
                num_datasets=repro_n,
            )
        )

    if include_micro:
        result.update(run_microbenchmarks(config, device=requested_device, repeats=3))

    if missingness_enabled and missingness_acceptance is not None:
        baseline_config = _clone_config(config)
        baseline_config.dataset.missing_rate = 0.0
        baseline_config.dataset.missing_mechanism = MISSINGNESS_MECHANISM_NONE
        baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            baseline_diagnostics_aggregator = _build_diagnostics_aggregator(baseline_config)
        baseline_missingness_acceptance = _MissingnessAcceptanceCollector(
            target_rate=float(config.dataset.missing_rate)
        )

        baseline_on_bundle_callback = None
        if baseline_diagnostics_aggregator is not None:

            def _baseline_on_bundle(bundle: DatasetBundle) -> None:
                baseline_diagnostics_aggregator.update_bundle(bundle)
                baseline_missingness_acceptance.update(bundle)

            baseline_on_bundle_callback = _baseline_on_bundle
        else:

            def _baseline_on_bundle(bundle: DatasetBundle) -> None:
                baseline_missingness_acceptance.update(bundle)

            baseline_on_bundle_callback = _baseline_on_bundle

        baseline_throughput = run_throughput_benchmark(
            baseline_config,
            num_datasets=num_datasets,
            warmup_datasets=warmup,
            device=requested_device,
            on_bundle=baseline_on_bundle_callback,
        )
        baseline_dpm = float(baseline_throughput.get("datasets_per_minute", 0.0))
        current_dpm = float(result.get("datasets_per_minute", 0.0))
        runtime_degradation = degradation_percent("datasets_per_minute", current_dpm, baseline_dpm)
        runtime_degradation_value = (
            float(runtime_degradation) if runtime_degradation is not None else 0.0
        )
        runtime_severity = _severity_from_thresholds(
            runtime_degradation_value,
            warn=float(warn_threshold_pct),
            fail=float(fail_threshold_pct),
        )

        acceptance_summary = missingness_acceptance.build_summary()
        issues = list(acceptance_summary["issues"])
        if runtime_severity != "pass":
            issues.append(
                _build_missingness_guardrail_issue(
                    metric="missingness_runtime_degradation_pct",
                    severity=runtime_severity,
                    current=current_dpm,
                    baseline=baseline_dpm,
                    degradation_pct=runtime_degradation_value,
                    detail=(
                        "Missingness-enabled throughput regressed versus an equivalent "
                        "missingness-disabled control run."
                    ),
                )
            )

        result["missingness_guardrails"] = {
            "enabled": True,
            "mechanism": str(config.dataset.missing_mechanism),
            "target_rate": float(config.dataset.missing_rate),
            "metadata_coverage_rate": float(acceptance_summary["metadata_coverage_rate"]),
            "realized_rate_overall": float(acceptance_summary["realized_rate_overall"]),
            "rate_abs_error": float(acceptance_summary["rate_abs_error"]),
            "rate_warn_abs_error": float(MISSINGNESS_RATE_WARN_ABS_ERROR),
            "rate_fail_abs_error": float(MISSINGNESS_RATE_FAIL_ABS_ERROR),
            "runtime_baseline_datasets_per_minute": baseline_dpm,
            "runtime_degradation_pct": (
                float(runtime_degradation) if runtime_degradation is not None else None
            ),
            "runtime_warn_threshold_pct": float(warn_threshold_pct),
            "runtime_fail_threshold_pct": float(fail_threshold_pct),
            "issues": issues,
            "status": _status_from_issues(issues),
        }

    result["lineage_guardrails"] = _collect_lineage_guardrails(
        config,
        suite=suite,
        num_datasets=num_datasets,
        device=requested_device,
        warn_threshold_pct=float(warn_threshold_pct),
        fail_threshold_pct=float(fail_threshold_pct),
    )

    if (
        diagnostics_enabled
        and diagnostics_aggregator is not None
        and diagnostics_root_dir is not None
    ):
        profile_segment = _sanitize_profile_key(spec.key)
        if diagnostics_occurrence_total > 1:
            profile_segment = f"{profile_segment}_run{diagnostics_occurrence_index + 1}"
        profile_diagnostics_dir = diagnostics_root_dir / "diagnostics" / profile_segment
        summary = diagnostics_aggregator.build_summary()
        json_path = write_coverage_summary_json(
            summary, profile_diagnostics_dir / "coverage_summary.json"
        )
        md_path = write_coverage_summary_markdown(
            summary, profile_diagnostics_dir / "coverage_summary.md"
        )
        result["diagnostics_artifacts"] = {
            "json": _artifact_pointer(json_path),
            "markdown": _artifact_pointer(md_path),
        }

    return result


def resolve_profile_run_specs(
    *,
    profile_keys: list[str] | None,
    config_path: str | None,
) -> list[ProfileRunSpec]:
    """Resolve requested profile keys into concrete benchmark run specs."""

    keys = list(profile_keys or [])
    if "all" in keys:
        keys = ["cpu", "cuda_desktop", "cuda_h100"]

    if not keys:
        if config_path:
            keys = ["custom"]
        else:
            keys = ["cpu", "cuda_desktop", "cuda_h100"]

    resolved: list[ProfileRunSpec] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)

        if key == "custom":
            if not config_path:
                raise ValueError("Profile 'custom' requires --config.")
            config = GeneratorConfig.from_yaml(config_path)
            profile_key = config.benchmark.profile_name or "custom"
            resolved.append(
                ProfileRunSpec(key=profile_key, config=config, device=config.runtime.device)
            )
            continue

        config_file = DEFAULT_PROFILE_CONFIGS.get(key)
        if not config_file:
            raise ValueError(f"Unknown benchmark profile key: {key}")

        config = GeneratorConfig.from_yaml(config_file)
        profile_device = str(
            config.benchmark.profiles.get(key, {}).get("device", config.runtime.device)
        )
        resolved.append(ProfileRunSpec(key=key, config=config, device=profile_device))

    return resolved


def run_benchmark_suite(
    profile_specs: list[ProfileRunSpec],
    *,
    suite: str,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    baseline_payload: dict[str, Any] | None,
    num_datasets_override: int | None,
    warmup_override: int | None,
    collect_memory: bool,
    collect_reproducibility: bool,
    collect_diagnostics: bool,
    diagnostics_root_dir: Path | None,
    fail_on_regression: bool,
    no_hardware_aware: bool,
) -> dict[str, Any]:
    """Run a benchmark suite over one or more profiles and attach regression diagnostics."""

    normalized_suite = suite.lower().strip()
    if normalized_suite not in {"smoke", "standard", "full"}:
        raise ValueError(f"Unsupported suite: {suite}")
    if collect_diagnostics and diagnostics_root_dir is None:
        raise ValueError("Benchmark diagnostics collection requires a diagnostics_root_dir.")

    include_micro = normalized_suite == "full"
    enable_repro = collect_reproducibility or normalized_suite == "full"
    key_totals: Counter[str] = Counter(spec.key for spec in profile_specs)
    key_seen: dict[str, int] = {}

    profile_results: list[dict[str, Any]] = []
    for spec in profile_specs:
        occurrence_index = key_seen.get(spec.key, 0)
        key_seen[spec.key] = occurrence_index + 1
        profile_results.append(
            run_profile_benchmark(
                spec,
                suite=normalized_suite,
                num_datasets_override=num_datasets_override,
                warmup_override=warmup_override,
                collect_memory=collect_memory,
                collect_reproducibility=enable_repro,
                collect_diagnostics=collect_diagnostics,
                diagnostics_root_dir=diagnostics_root_dir,
                warn_threshold_pct=float(warn_threshold_pct),
                fail_threshold_pct=float(fail_threshold_pct),
                include_micro=include_micro,
                no_hardware_aware=no_hardware_aware,
                diagnostics_occurrence_index=occurrence_index,
                diagnostics_occurrence_total=key_totals[spec.key],
            )
        )

    summary: dict[str, Any] = {
        "suite": normalized_suite,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "profile_results": profile_results,
    }

    regression: dict[str, Any]
    if baseline_payload is not None:
        regression = compare_summary_to_baseline(
            summary,
            baseline_payload,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        )
    else:
        regression = {
            "status": "pass",
            "warn_threshold_pct": float(warn_threshold_pct),
            "fail_threshold_pct": float(fail_threshold_pct),
            "issues": [],
        }

    missingness_issues = _collect_missingness_regression_issues(profile_results)
    lineage_issues = _collect_lineage_regression_issues(profile_results)
    additional_issues = [*missingness_issues, *lineage_issues]
    if additional_issues:
        existing_issues = regression.get("issues", [])
        if not isinstance(existing_issues, list):
            existing_issues = []
        all_issues = [*existing_issues, *additional_issues]
        regression["issues"] = sorted(all_issues, key=_issue_sort_key)
        regression["status"] = _status_from_issues(regression["issues"])

    regression["fail_on_regression"] = bool(fail_on_regression)
    regression["hard_fail"] = bool(fail_on_regression and regression.get("status") == "fail")
    summary["regression"] = regression
    return summary
