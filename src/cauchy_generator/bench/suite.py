"""Benchmark suite orchestration across runtime profiles."""

from __future__ import annotations

import datetime as dt
import resource
import sys
import time
import torch
from dataclasses import dataclass
from typing import Any

from cauchy_generator.bench.baseline import compare_summary_to_baseline
from cauchy_generator.bench.micro import run_microbenchmarks
from cauchy_generator.bench.metrics import (
    reproducibility_signature,
    summarize_latencies,
)
from cauchy_generator.bench.throughput import run_throughput_benchmark
from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch_iter, generate_one
from cauchy_generator.hardware import (
    HardwareInfo,
    apply_hardware_profile,
    detect_hardware,
)
from cauchy_generator.rng import SeedManager


DEFAULT_PROFILE_CONFIGS: dict[str, str] = {
    "cpu": "configs/benchmark_cpu.yaml",
    "cuda_desktop": "configs/benchmark_cuda_desktop.yaml",
    "cuda_h100": "configs/benchmark_cuda_h100.yaml",
}


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

    result = run_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        device=requested_device,
    )
    result["profile_key"] = spec.key
    result["suite"] = suite
    result["device"] = requested_device
    result["hardware_backend"] = hw.backend
    result["hardware_device_name"] = hw.device_name
    result["hardware_memory_gb"] = hw.total_memory_gb
    result["hardware_peak_flops"] = hw.peak_flops
    result["hardware_profile"] = hw.profile

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
    fail_on_regression: bool,
    no_hardware_aware: bool,
) -> dict[str, Any]:
    """Run a benchmark suite over one or more profiles and attach regression diagnostics."""

    normalized_suite = suite.lower().strip()
    if normalized_suite not in {"smoke", "standard", "full"}:
        raise ValueError(f"Unsupported suite: {suite}")

    include_micro = normalized_suite == "full"
    enable_repro = collect_reproducibility or normalized_suite == "full"

    profile_results: list[dict[str, Any]] = []
    for spec in profile_specs:
        profile_results.append(
            run_profile_benchmark(
                spec,
                suite=normalized_suite,
                num_datasets_override=num_datasets_override,
                warmup_override=warmup_override,
                collect_memory=collect_memory,
                collect_reproducibility=enable_repro,
                include_micro=include_micro,
                no_hardware_aware=no_hardware_aware,
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

    regression["fail_on_regression"] = bool(fail_on_regression)
    regression["hard_fail"] = bool(fail_on_regression and regression.get("status") == "fail")
    summary["regression"] = regression
    return summary
