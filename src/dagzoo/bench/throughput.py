"""Throughput benchmark harness."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from dagzoo.bench.constants import (
    SECONDS_PER_MINUTE,
    THROUGHPUT_MEASURE_SEED_OFFSET,
    THROUGHPUT_SLO_DATASETS_PER_MINUTE,
    THROUGHPUT_WARMUP_SEED_OFFSET,
)
from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.rng import offset_seed32
from dagzoo.types import DatasetBundle


def _consume_generation(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int,
    device: str | None,
    on_bundle: Callable[[DatasetBundle], object] | None = None,
) -> None:
    """Run generation for ``num_datasets`` items while discarding outputs."""

    for bundle in generate_batch_iter(
        config,
        num_datasets=num_datasets,
        seed=seed,
        device=device,
    ):
        if on_bundle is not None:
            on_bundle(bundle)


def run_throughput_benchmark(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int = 10,
    device: str | None = None,
    on_bundle: Callable[[DatasetBundle], object] | None = None,
) -> dict[str, Any]:
    """Measure end-to-end generation throughput for a benchmark preset."""

    if warmup_datasets > 0:
        _consume_generation(
            config,
            num_datasets=warmup_datasets,
            seed=offset_seed32(config.seed, THROUGHPUT_WARMUP_SEED_OFFSET),
            device=device,
        )

    start = time.perf_counter()
    _consume_generation(
        config,
        num_datasets=num_datasets,
        seed=offset_seed32(config.seed, THROUGHPUT_MEASURE_SEED_OFFSET),
        device=device,
        on_bundle=on_bundle,
    )
    elapsed = time.perf_counter() - start
    dps = num_datasets / elapsed if elapsed > 0 else 0.0
    dpm = dps * SECONDS_PER_MINUTE
    return {
        "preset": config.benchmark.preset_name,
        "num_datasets": num_datasets,
        "warmup_datasets": warmup_datasets,
        "elapsed_seconds": elapsed,
        "datasets_per_second": dps,
        "datasets_per_minute": dpm,
        "slo_pass_100_datasets_per_min": dpm >= THROUGHPUT_SLO_DATASETS_PER_MINUTE,
        "generation_mode": "fixed_batched",
    }
