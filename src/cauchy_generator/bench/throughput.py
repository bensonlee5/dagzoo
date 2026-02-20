"""Throughput benchmark harness."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch


def run_throughput_benchmark(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int = 10,
    device: str | None = None,
) -> dict[str, float | int | str]:
    """Measure end-to-end generation throughput for a benchmark profile."""

    if warmup_datasets > 0:
        generate_batch(config, num_datasets=warmup_datasets, seed=config.seed + 1, device=device)

    start = time.perf_counter()
    generate_batch(config, num_datasets=num_datasets, seed=config.seed + 2, device=device)
    elapsed = time.perf_counter() - start
    dps = num_datasets / elapsed if elapsed > 0 else 0.0
    dpm = dps * 60.0
    return {
        "profile": config.benchmark.profile_name,
        "num_datasets": num_datasets,
        "warmup_datasets": warmup_datasets,
        "elapsed_seconds": elapsed,
        "datasets_per_second": dps,
        "datasets_per_minute": dpm,
        "slo_pass_100_datasets_per_min": dpm >= 100.0,
    }


def write_benchmark_json(results: dict[str, float | int | str], out_path: str | Path) -> Path:
    """Write benchmark results as JSON, replacing non-finite floats with null."""

    sanitized: dict[str, float | int | str | None] = {}
    for key, value in results.items():
        if isinstance(value, float) and not math.isfinite(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, sort_keys=True)
    return path
