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
from dagzoo.core.fixed_layout import FixedLayoutPlan, generate_batch_fixed_layout_iter
from dagzoo.rng import offset_seed32
from dagzoo.types import DatasetBundle


def _select_generation_iterator(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    fixed_layout_plan: FixedLayoutPlan | None = None,
) -> Any:
    """Choose one generator path for the full benchmark run."""

    _ = config
    _ = num_datasets
    if fixed_layout_plan is not None:
        return generate_batch_fixed_layout_iter
    return generate_batch_iter


def _consume_generation(
    generator: Any,
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int,
    device: str | None,
    fixed_layout_plan: FixedLayoutPlan | None = None,
    fixed_layout_batch_size: int | None = None,
    on_bundle: Callable[[DatasetBundle], object] | None = None,
) -> None:
    """Run generation for ``num_datasets`` items while discarding outputs."""

    generator_kwargs: dict[str, Any] = {
        "num_datasets": num_datasets,
        "seed": seed,
    }
    if fixed_layout_plan is not None:
        generator_kwargs["plan"] = fixed_layout_plan
        generator_kwargs["device"] = device
        if fixed_layout_batch_size is not None:
            generator_kwargs["batch_size"] = int(fixed_layout_batch_size)
    else:
        generator_kwargs["device"] = device

    for bundle in generator(config, **generator_kwargs):
        if on_bundle is not None:
            on_bundle(bundle)


def run_throughput_benchmark(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int = 10,
    device: str | None = None,
    fixed_layout_plan: FixedLayoutPlan | None = None,
    fixed_layout_batch_size: int | None = None,
    on_bundle: Callable[[DatasetBundle], object] | None = None,
) -> dict[str, Any]:
    """Measure end-to-end generation throughput for a benchmark preset."""

    generator = _select_generation_iterator(
        config,
        num_datasets=num_datasets,
        fixed_layout_plan=fixed_layout_plan,
    )
    if warmup_datasets > 0:
        _consume_generation(
            generator,
            config,
            num_datasets=warmup_datasets,
            seed=offset_seed32(config.seed, THROUGHPUT_WARMUP_SEED_OFFSET),
            device=device,
            fixed_layout_plan=fixed_layout_plan,
            fixed_layout_batch_size=fixed_layout_batch_size,
        )

    start = time.perf_counter()
    _consume_generation(
        generator,
        config,
        num_datasets=num_datasets,
        seed=offset_seed32(config.seed, THROUGHPUT_MEASURE_SEED_OFFSET),
        device=device,
        fixed_layout_plan=fixed_layout_plan,
        fixed_layout_batch_size=fixed_layout_batch_size,
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
