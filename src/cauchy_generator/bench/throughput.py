"""Throughput benchmark harness."""

from __future__ import annotations

import time

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch_iter


def _consume_generation(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int,
    device: str | None,
) -> None:
    """Run generation for ``num_datasets`` items while discarding outputs."""

    for _ in generate_batch_iter(
        config,
        num_datasets=num_datasets,
        seed=seed,
        device=device,
    ):
        pass


def run_throughput_benchmark(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int = 10,
    device: str | None = None,
) -> dict[str, float | int | str | None]:
    """Measure end-to-end generation throughput for a benchmark profile."""

    if warmup_datasets > 0:
        _consume_generation(
            config,
            num_datasets=warmup_datasets,
            seed=config.seed + 1,
            device=device,
        )

    start = time.perf_counter()
    _consume_generation(
        config,
        num_datasets=num_datasets,
        seed=config.seed + 2,
        device=device,
    )
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
