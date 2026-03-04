"""Benchmark suite orchestration across runtime presets."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import copy
import re
import resource
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dagzoo.bench.baseline import compare_summary_to_baseline
from dagzoo.bench.constants import (
    DIAGNOSTICS_DUPLICATE_PRESET_SUFFIX_BASE,
    KIB,
    LATENCY_SEED_OFFSET,
    MIB,
    MICROBENCH_REPEATS,
    MISSINGNESS_RATE_FAIL_ABS_ERROR,
    MISSINGNESS_RATE_WARN_ABS_ERROR,
    PRESET_KEY_HASH_SUFFIX_LEN,
    REPRODUCIBILITY_SEED_OFFSET,
    SMOKE_LATENCY_SAMPLES_CAP,
    SMOKE_N_FEATURES_CAP,
    SMOKE_N_NODES_CAP,
    SMOKE_N_TEST_CAP,
    SMOKE_N_TRAIN_CAP,
    SMOKE_NUM_DATASETS_CAP,
    SMOKE_WARMUP_DATASETS_CAP,
    SHIFT_GUARDRAIL_DIRECTIONAL_GATING_MIN_SAMPLE,
    SHIFT_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE,
    NOISE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE,
)
from dagzoo.bench.micro import run_microbenchmarks
from dagzoo.bench.metrics import (
    degradation_percent,
    percent_change,
    reproducibility_signature,
    summarize_latencies,
)
from dagzoo.bench.collectors import (
    _MissingnessAcceptanceCollector,
    _NoiseGuardrailCollector,
    _ShiftGuardrailCollector,
    _compose_bundle_callback,
)
from dagzoo.bench.guardrails import (
    _build_guardrail_issue,
    _collect_lineage_guardrails,
    _collect_guardrail_regression_issues,
    _issue_sort_key,
    _severity_from_thresholds,
    _status_from_issues,
)
from dagzoo.bench.throughput import run_throughput_benchmark
from dagzoo.config import (
    GeneratorConfig,
    MISSINGNESS_MECHANISM_NONE,
    NOISE_FAMILY_GAUSSIAN,
    SHIFT_MODE_OFF,
)
from dagzoo.core.config_resolution import (
    BenchmarkSmokeCaps,
    resolve_benchmark_preset_config,
    serialize_resolution_events,
)
from dagzoo.core.dataset import generate_batch_iter, generate_one
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.diagnostics import (
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics_targets import (
    build_diagnostics_aggregation_config,
)
from dagzoo.rng import SeedManager, offset_seed32


DEFAULT_PRESET_CONFIGS: dict[str, str] = {
    "cpu": "configs/benchmark_cpu.yaml",
    "cuda_desktop": "configs/benchmark_cuda_desktop.yaml",
    "cuda_h100": "configs/benchmark_cuda_h100.yaml",
}


@dataclass(slots=True)
class PresetRunSpec:
    """Benchmark execution spec for one preset/config pair."""

    key: str
    config: GeneratorConfig
    device: str | None = None


def _copy_runtime_config(config: GeneratorConfig) -> GeneratorConfig:
    """Copy an already validated runtime config without re-running schema validation."""

    return copy.deepcopy(config)


def _peak_rss_mb() -> float:
    """Return process max resident set size in MiB."""

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / MIB
    return rss / KIB


def _preset_counts(
    config: GeneratorConfig,
    *,
    preset_key: str,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
) -> tuple[int, int]:
    """Resolve benchmark dataset and warmup counts for a preset and suite mode."""

    preset_map = config.benchmark.presets.get(preset_key, {})
    num = int(preset_map.get("num_datasets", config.benchmark.num_datasets))
    warmup = int(preset_map.get("warmup_datasets", config.benchmark.warmup_datasets))

    if num_datasets_override is not None:
        num = int(num_datasets_override)
    if warmup_override is not None:
        warmup = int(warmup_override)

    if suite == "smoke":
        num = min(num, SMOKE_NUM_DATASETS_CAP)
        warmup = min(warmup, SMOKE_WARMUP_DATASETS_CAP)

    return max(1, num), max(0, warmup)


def _latency_sample_count(config: GeneratorConfig, suite: str, num_datasets: int) -> int:
    """Choose per-preset latency sample count for the requested suite level."""

    n = max(1, min(int(config.benchmark.latency_num_samples), num_datasets))
    if suite == "smoke":
        return min(n, SMOKE_LATENCY_SAMPLES_CAP)
    return n


def _collect_latency(
    config: GeneratorConfig,
    *,
    device: str | None,
    num_samples: int,
) -> dict[str, float]:
    """Collect per-dataset latency samples by repeatedly calling ``generate_one``."""

    manager = SeedManager(offset_seed32(config.seed, LATENCY_SEED_OFFSET))
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
    run_seed = offset_seed32(config.seed, REPRODUCIBILITY_SEED_OFFSET)
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


def _sanitize_preset_key(preset_key: str) -> str:
    """Normalize preset key into a filesystem-safe unique path segment."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", preset_key).strip("._-")
    if not normalized:
        normalized = "preset"
    suffix = hashlib.sha1(preset_key.encode("utf-8")).hexdigest()[:PRESET_KEY_HASH_SUFFIX_LEN]
    return f"{normalized}_{suffix}"


def _artifact_pointer(path: Path) -> str:
    """Return a summary-safe pointer for diagnostics artifacts."""

    return str(path.resolve())


def _build_diagnostics_aggregator(config: GeneratorConfig) -> CoverageAggregator:
    """Create a diagnostics coverage aggregator from config."""

    return CoverageAggregator(build_diagnostics_aggregation_config(config.diagnostics))


def _is_missingness_enabled(config: GeneratorConfig) -> bool:
    """Return whether missingness is enabled in config."""

    return bool(
        float(config.dataset.missing_rate) > 0.0
        and str(config.dataset.missing_mechanism).strip().lower() != MISSINGNESS_MECHANISM_NONE
    )


def _is_shift_enabled(config: GeneratorConfig) -> bool:
    """Return whether shift controls are enabled in config."""

    return bool(config.shift.enabled)


def _is_noise_enabled(config: GeneratorConfig) -> bool:
    """Return whether non-gaussian noise controls are enabled in config."""

    return str(config.noise.family).strip().lower() != NOISE_FAMILY_GAUSSIAN


def _build_shift_directional_check(
    *,
    metric: str,
    enabled: bool,
    gating_enabled: bool,
    current: float | None,
    baseline: float | None,
    detail: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build directional check payload and optional issue for one shift metric."""

    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "gating_enabled": bool(gating_enabled),
        "current": current,
        "baseline": baseline,
        "status": "not_applicable",
        "detail": detail,
    }
    if not enabled:
        payload["reason"] = "axis_inactive"
        return payload, None
    if not gating_enabled:
        payload["status"] = "suppressed"
        payload["reason"] = "insufficient_sample_size"
        return payload, None
    if current is None or baseline is None:
        payload["status"] = "fail"
        issue = _build_guardrail_issue(
            metric=f"shift_{metric}_directionality_unavailable",
            severity="fail",
            current=current,
            baseline=baseline,
            degradation_pct=None,
            detail=f"{detail} Directional check could not be computed from benchmark samples.",
        )
        return payload, issue
    if float(current) > float(baseline):
        payload["status"] = "pass"
        return payload, None

    payload["status"] = "fail"
    current_value = float(current)
    baseline_value = float(baseline)
    raw_change = percent_change(current_value, baseline_value)
    raw_degradation = -raw_change if raw_change is not None else None
    issue = _build_guardrail_issue(
        metric=f"shift_{metric}_directionality",
        severity="fail",
        current=current_value,
        baseline=baseline_value,
        degradation_pct=(float(raw_degradation) if raw_degradation is not None else None),
        detail=detail,
    )
    return payload, issue


def run_preset_benchmark(
    spec: PresetRunSpec,
    *,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
    collect_memory: bool,
    collect_reproducibility: bool,
    include_micro: bool,
    hardware_policy: str,
    collect_diagnostics: bool,
    diagnostics_root_dir: Path | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    diagnostics_occurrence_index: int,
    diagnostics_occurrence_total: int,
) -> dict[str, Any]:
    """Run one benchmark preset and collect throughput, latency, and optional diagnostics."""

    resolved_preset = resolve_benchmark_preset_config(
        preset_key=spec.key,
        config=spec.config,
        preset_device=spec.device,
        suite=suite,
        hardware_policy=hardware_policy,
        smoke_caps=BenchmarkSmokeCaps(
            n_train=SMOKE_N_TRAIN_CAP,
            n_test=SMOKE_N_TEST_CAP,
            n_features=SMOKE_N_FEATURES_CAP,
            n_nodes=SMOKE_N_NODES_CAP,
        ),
    )
    config = resolved_preset.config
    requested_device = resolved_preset.requested_device
    hw = resolved_preset.hardware

    num_datasets, warmup = _preset_counts(
        config,
        preset_key=spec.key,
        suite=suite,
        num_datasets_override=num_datasets_override,
        warmup_override=warmup_override,
    )

    rss_before = _peak_rss_mb() if collect_memory else 0.0
    if collect_memory and hw.backend == "cuda" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()

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
    shift_enabled = _is_shift_enabled(config)
    shift_guardrails = _ShiftGuardrailCollector() if shift_enabled else None
    noise_enabled = _is_noise_enabled(config)
    noise_guardrails = (
        _NoiseGuardrailCollector(expected_family_requested=str(config.noise.family))
        if noise_enabled
        else None
    )

    on_bundle_callback = _compose_bundle_callback(
        diagnostics_aggregator=diagnostics_aggregator,
        missingness_acceptance=missingness_acceptance,
        shift_guardrails=shift_guardrails,
        noise_guardrails=noise_guardrails,
    )

    result = run_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        device=requested_device,
        on_bundle=on_bundle_callback,
    )
    result["preset_key"] = spec.key
    result["suite"] = suite
    result["device"] = requested_device
    result["hardware_backend"] = hw.backend
    result["hardware_device_name"] = hw.device_name
    result["hardware_memory_gb"] = hw.total_memory_gb
    result["hardware_peak_flops"] = hw.peak_flops
    result["hardware_tier"] = hw.tier
    result["hardware_policy"] = str(hardware_policy)
    result["effective_config"] = config.to_dict()
    result["effective_config_trace"] = serialize_resolution_events(resolved_preset.trace_events)
    result["diagnostics_enabled"] = diagnostics_enabled
    result["diagnostics_artifacts"] = None
    result["missingness_guardrails"] = {"enabled": False}
    result["lineage_guardrails"] = {"enabled": False}
    result["shift_guardrails"] = {"enabled": False}
    result["noise_guardrails"] = {"enabled": False}

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
                result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / MIB
                result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / MIB
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
        result.update(
            run_microbenchmarks(config, device=requested_device, repeats=MICROBENCH_REPEATS)
        )

    if missingness_enabled and missingness_acceptance is not None:
        baseline_config = _copy_runtime_config(config)
        baseline_config.dataset.missing_rate = 0.0
        baseline_config.dataset.missing_mechanism = MISSINGNESS_MECHANISM_NONE
        missingness_baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            missingness_baseline_diagnostics_aggregator = _build_diagnostics_aggregator(
                baseline_config
            )
        baseline_missingness_acceptance = _MissingnessAcceptanceCollector(
            target_rate=float(config.dataset.missing_rate)
        )
        # Keep control-run callback instrumentation equivalent so runtime delta
        # reflects missingness overhead instead of callback overhead skew.
        baseline_on_bundle_callback = _compose_bundle_callback(
            diagnostics_aggregator=missingness_baseline_diagnostics_aggregator,
            missingness_acceptance=baseline_missingness_acceptance,
            shift_guardrails=_ShiftGuardrailCollector() if shift_enabled else None,
            noise_guardrails=(
                _NoiseGuardrailCollector(expected_family_requested=str(config.noise.family))
                if noise_enabled
                else None
            ),
        )

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
                _build_guardrail_issue(
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

    if shift_enabled and shift_guardrails is not None:
        shift_params = resolve_shift_runtime_params(config)
        baseline_config = _copy_runtime_config(config)
        baseline_config.shift.enabled = False
        baseline_config.shift.mode = SHIFT_MODE_OFF
        baseline_config.shift.graph_scale = None
        baseline_config.shift.mechanism_scale = None
        baseline_config.shift.variance_scale = None
        shift_baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            shift_baseline_diagnostics_aggregator = _build_diagnostics_aggregator(baseline_config)
        shift_baseline_missingness_acceptance = (
            _MissingnessAcceptanceCollector(target_rate=float(config.dataset.missing_rate))
            if missingness_acceptance is not None
            else None
        )
        baseline_shift_guardrails = _ShiftGuardrailCollector()
        baseline_on_bundle_callback = _compose_bundle_callback(
            diagnostics_aggregator=shift_baseline_diagnostics_aggregator,
            missingness_acceptance=shift_baseline_missingness_acceptance,
            shift_guardrails=baseline_shift_guardrails,
            noise_guardrails=(
                _NoiseGuardrailCollector(expected_family_requested=str(config.noise.family))
                if noise_enabled
                else None
            ),
        )

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
        runtime_gating_enabled = num_datasets >= SHIFT_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE
        directional_gating_enabled = num_datasets >= SHIFT_GUARDRAIL_DIRECTIONAL_GATING_MIN_SAMPLE

        current_summary = shift_guardrails.build_summary()
        baseline_summary = baseline_shift_guardrails.build_summary()

        shift_issues: list[dict[str, Any]] = []
        metadata_coverage_rate = float(current_summary["metadata_coverage_rate"])
        if metadata_coverage_rate < 1.0:
            shift_issues.append(
                _build_guardrail_issue(
                    metric="shift_metadata_coverage",
                    severity="fail",
                    current=metadata_coverage_rate,
                    baseline=1.0,
                    degradation_pct=float(max(0.0, (1.0 - metadata_coverage_rate) * 100.0)),
                    detail="Shift metadata must be present for all shift-enabled bundles.",
                )
            )
        shift_enabled_coverage_rate = float(current_summary["shift_enabled_coverage_rate"])
        if shift_enabled_coverage_rate < 1.0:
            shift_issues.append(
                _build_guardrail_issue(
                    metric="shift_enabled_metadata_coverage",
                    severity="fail",
                    current=shift_enabled_coverage_rate,
                    baseline=1.0,
                    degradation_pct=float(max(0.0, (1.0 - shift_enabled_coverage_rate) * 100.0)),
                    detail="Shift-enabled benchmark runs must emit shift.enabled=true metadata.",
                )
            )
        if runtime_gating_enabled and runtime_severity != "pass":
            shift_issues.append(
                _build_guardrail_issue(
                    metric="shift_runtime_degradation_pct",
                    severity=runtime_severity,
                    current=current_dpm,
                    baseline=baseline_dpm,
                    degradation_pct=runtime_degradation_value,
                    detail=(
                        "Shift-enabled throughput regressed versus an equivalent "
                        "shift-disabled control run."
                    ),
                )
            )

        graph_check, graph_issue = _build_shift_directional_check(
            metric="graph_edge_density",
            enabled=float(shift_params.graph_scale) > 0.0,
            gating_enabled=directional_gating_enabled,
            current=current_summary.get("mean_graph_edge_density"),
            baseline=baseline_summary.get("mean_graph_edge_density"),
            detail=(
                "Graph shift should increase mean graph edge density "
                "relative to a shift-disabled control run."
            ),
        )
        mechanism_check, mechanism_issue = _build_shift_directional_check(
            metric="mechanism_nonlinear_mass",
            enabled=float(shift_params.mechanism_scale) > 0.0,
            gating_enabled=directional_gating_enabled,
            current=current_summary.get("mean_mechanism_nonlinear_mass"),
            baseline=baseline_summary.get("mean_mechanism_nonlinear_mass"),
            detail=(
                "Mechanism shift should increase nonlinear family mass "
                "relative to a shift-disabled control run."
            ),
        )
        noise_check, noise_issue = _build_shift_directional_check(
            metric="noise_variance_multiplier",
            enabled=float(shift_params.variance_scale) > 0.0,
            gating_enabled=directional_gating_enabled,
            current=current_summary.get("mean_noise_variance_multiplier"),
            baseline=baseline_summary.get("mean_noise_variance_multiplier"),
            detail=(
                "Noise shift should increase noise variance multiplier "
                "relative to a shift-disabled control run."
            ),
        )
        for maybe_issue in (graph_issue, mechanism_issue, noise_issue):
            if maybe_issue is not None:
                shift_issues.append(maybe_issue)

        result["shift_guardrails"] = {
            "enabled": True,
            "mode": str(shift_params.mode),
            "graph_scale": float(shift_params.graph_scale),
            "mechanism_scale": float(shift_params.mechanism_scale),
            "variance_scale": float(shift_params.variance_scale),
            "sample_datasets": int(num_datasets),
            "metadata_coverage_rate": metadata_coverage_rate,
            "shift_enabled_coverage_rate": shift_enabled_coverage_rate,
            "runtime_gating_enabled": bool(runtime_gating_enabled),
            "runtime_gating_min_sample_datasets": int(SHIFT_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE),
            "runtime_gating_suppressed_reason": (
                None if runtime_gating_enabled else "insufficient_sample_size"
            ),
            "directional_gating_enabled": bool(directional_gating_enabled),
            "directional_gating_min_sample_datasets": int(
                SHIFT_GUARDRAIL_DIRECTIONAL_GATING_MIN_SAMPLE
            ),
            "directional_gating_suppressed_reason": (
                None if directional_gating_enabled else "insufficient_sample_size"
            ),
            "current_means": {
                "graph_edge_density": current_summary.get("mean_graph_edge_density"),
                "edge_odds_multiplier": current_summary.get("mean_edge_odds_multiplier"),
                "mechanism_nonlinear_mass": current_summary.get("mean_mechanism_nonlinear_mass"),
                "noise_variance_multiplier": current_summary.get("mean_noise_variance_multiplier"),
            },
            "baseline_means": {
                "graph_edge_density": baseline_summary.get("mean_graph_edge_density"),
                "edge_odds_multiplier": baseline_summary.get("mean_edge_odds_multiplier"),
                "mechanism_nonlinear_mass": baseline_summary.get("mean_mechanism_nonlinear_mass"),
                "noise_variance_multiplier": baseline_summary.get("mean_noise_variance_multiplier"),
            },
            "directional_checks": {
                "graph_edge_density": graph_check,
                "mechanism_nonlinear_mass": mechanism_check,
                "noise_variance_multiplier": noise_check,
            },
            "runtime_baseline_datasets_per_minute": baseline_dpm,
            "runtime_with_shift_datasets_per_minute": current_dpm,
            "runtime_degradation_pct": (
                float(runtime_degradation) if runtime_degradation is not None else None
            ),
            "runtime_warn_threshold_pct": float(warn_threshold_pct),
            "runtime_fail_threshold_pct": float(fail_threshold_pct),
            "issues": shift_issues,
            "status": _status_from_issues(shift_issues),
        }

    if noise_enabled and noise_guardrails is not None:
        baseline_config = _copy_runtime_config(config)
        baseline_config.noise.family = NOISE_FAMILY_GAUSSIAN
        baseline_config.noise.base_scale = 1.0
        baseline_config.noise.student_t_df = 5.0
        baseline_config.noise.mixture_weights = None
        noise_baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            noise_baseline_diagnostics_aggregator = _build_diagnostics_aggregator(baseline_config)
        noise_baseline_missingness_acceptance = (
            _MissingnessAcceptanceCollector(target_rate=float(config.dataset.missing_rate))
            if missingness_acceptance is not None
            else None
        )
        noise_baseline_shift_guardrails = _ShiftGuardrailCollector() if shift_enabled else None
        baseline_noise_guardrails = _NoiseGuardrailCollector(
            expected_family_requested=str(baseline_config.noise.family)
        )
        baseline_on_bundle_callback = _compose_bundle_callback(
            diagnostics_aggregator=noise_baseline_diagnostics_aggregator,
            missingness_acceptance=noise_baseline_missingness_acceptance,
            shift_guardrails=noise_baseline_shift_guardrails,
            noise_guardrails=baseline_noise_guardrails,
        )
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
        runtime_gating_enabled = num_datasets >= NOISE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE

        current_summary = noise_guardrails.build_summary()
        baseline_summary = baseline_noise_guardrails.build_summary()
        noise_issues = list(current_summary["issues"])
        if runtime_gating_enabled and runtime_severity != "pass":
            noise_issues.append(
                _build_guardrail_issue(
                    metric="noise_runtime_degradation_pct",
                    severity=runtime_severity,
                    current=current_dpm,
                    baseline=baseline_dpm,
                    degradation_pct=runtime_degradation_value,
                    detail=(
                        "Non-gaussian noise throughput regressed versus an equivalent "
                        "gaussian-noise control run."
                    ),
                )
            )

        result["noise_guardrails"] = {
            "enabled": True,
            "family_requested": str(config.noise.family),
            "sample_datasets": int(num_datasets),
            "metadata_coverage_rate": float(current_summary["metadata_coverage_rate"]),
            "metadata_valid_rate": float(current_summary["metadata_valid_rate"]),
            "valid_metadata_count": int(current_summary["valid_metadata_count"]),
            "sampled_family_counts": dict(current_summary["sampled_family_counts"]),
            "invalid_reason_counts": dict(current_summary["invalid_reason_counts"]),
            "runtime_gating_enabled": bool(runtime_gating_enabled),
            "runtime_gating_min_sample_datasets": int(NOISE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE),
            "runtime_gating_suppressed_reason": (
                None if runtime_gating_enabled else "insufficient_sample_size"
            ),
            "runtime_baseline_family_requested": str(baseline_config.noise.family),
            "runtime_baseline_sampled_family_counts": dict(
                baseline_summary["sampled_family_counts"]
            ),
            "runtime_baseline_datasets_per_minute": baseline_dpm,
            "runtime_with_noise_datasets_per_minute": current_dpm,
            "runtime_degradation_pct": (
                float(runtime_degradation) if runtime_degradation is not None else None
            ),
            "runtime_warn_threshold_pct": float(warn_threshold_pct),
            "runtime_fail_threshold_pct": float(fail_threshold_pct),
            "issues": noise_issues,
            "status": _status_from_issues(noise_issues),
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
        preset_segment = _sanitize_preset_key(spec.key)
        if diagnostics_occurrence_total > 1:
            run_number = diagnostics_occurrence_index + DIAGNOSTICS_DUPLICATE_PRESET_SUFFIX_BASE
            preset_segment = f"{preset_segment}_run{run_number}"
        preset_diagnostics_dir = diagnostics_root_dir / "diagnostics" / preset_segment
        summary = diagnostics_aggregator.build_summary()
        json_path = write_coverage_summary_json(
            summary, preset_diagnostics_dir / "coverage_summary.json"
        )
        md_path = write_coverage_summary_markdown(
            summary, preset_diagnostics_dir / "coverage_summary.md"
        )
        result["diagnostics_artifacts"] = {
            "json": _artifact_pointer(json_path),
            "markdown": _artifact_pointer(md_path),
        }

    return result


def resolve_preset_run_specs(
    *,
    preset_keys: list[str] | None,
    config_path: str | None,
) -> list[PresetRunSpec]:
    """Resolve requested preset keys into concrete benchmark run specs."""

    keys = list(preset_keys or [])
    if "all" in keys:
        keys = ["cpu", "cuda_desktop", "cuda_h100"]

    if not keys:
        keys = ["custom"] if config_path else ["cpu"]

    resolved: list[PresetRunSpec] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)

        if key == "custom":
            if not config_path:
                raise ValueError("Preset 'custom' requires --config.")
            config = GeneratorConfig.from_yaml(config_path)
            preset_key = config.benchmark.preset_name or "custom"
            resolved.append(
                PresetRunSpec(key=preset_key, config=config, device=config.runtime.device)
            )
            continue

        config_file = DEFAULT_PRESET_CONFIGS.get(key)
        if not config_file:
            raise ValueError(f"Unknown benchmark preset key: {key}")

        config = GeneratorConfig.from_yaml(config_file)
        preset_device = str(
            config.benchmark.presets.get(key, {}).get("device", config.runtime.device)
        )
        resolved.append(PresetRunSpec(key=key, config=config, device=preset_device))

    return resolved


def run_benchmark_suite(
    preset_specs: list[PresetRunSpec],
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
    hardware_policy: str,
) -> dict[str, Any]:
    """Run a benchmark suite over one or more presets and attach regression diagnostics."""

    normalized_suite = suite.lower().strip()
    if normalized_suite not in {"smoke", "standard", "full"}:
        raise ValueError(f"Unsupported suite: {suite}")
    if collect_diagnostics and diagnostics_root_dir is None:
        raise ValueError("Benchmark diagnostics collection requires a diagnostics_root_dir.")

    include_micro = normalized_suite == "full"
    enable_repro = collect_reproducibility or normalized_suite == "full"
    key_totals: Counter[str] = Counter(spec.key for spec in preset_specs)
    key_seen: dict[str, int] = {}

    preset_results: list[dict[str, Any]] = []
    for spec in preset_specs:
        occurrence_index = key_seen.get(spec.key, 0)
        key_seen[spec.key] = occurrence_index + 1
        preset_results.append(
            run_preset_benchmark(
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
                hardware_policy=hardware_policy,
                diagnostics_occurrence_index=occurrence_index,
                diagnostics_occurrence_total=key_totals[spec.key],
            )
        )

    summary: dict[str, Any] = {
        "suite": normalized_suite,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "preset_results": preset_results,
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

    missingness_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="missingness_guardrails"
    )
    lineage_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="lineage_guardrails"
    )
    shift_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="shift_guardrails"
    )
    noise_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="noise_guardrails"
    )
    additional_issues = [*missingness_issues, *lineage_issues, *shift_issues, *noise_issues]
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
