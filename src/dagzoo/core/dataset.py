"""Seeded synthetic dataset generation public entrypoints."""

from __future__ import annotations

from collections.abc import Iterator

from dagzoo.config import GeneratorConfig
from dagzoo.core import generation_context as _generation_context
from dagzoo.core import generation_engine as _generation_engine
from dagzoo.core.fixed_layout import (
    FixedLayoutPlan,
    generate_batch_fixed_layout,
    generate_batch_fixed_layout_iter,
    sample_fixed_layout,
)
from dagzoo.core.worker_partition import iter_worker_dataset_seeds
from dagzoo.rng import SeedManager
from dagzoo.types import DatasetBundle

__all__ = [
    "FixedLayoutPlan",
    "generate_batch",
    "generate_batch_fixed_layout",
    "generate_batch_fixed_layout_iter",
    "generate_batch_iter",
    "generate_one",
    "sample_fixed_layout",
]


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle with deterministic per-dataset randomness."""

    run_seed = _generation_context._resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _generation_context._resolve_device(config, device)
    return _generation_engine._generate_one_seeded(
        config,
        seed=run_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
    )


def generate_batch(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> list[DatasetBundle]:
    """Generate a batch of datasets using deterministic per-dataset child seeds."""

    return list(
        generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        )
    )


def generate_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets lazily using deterministic per-dataset child seeds."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _generation_context._resolve_device(config, device)
    run_seed = _generation_context._resolve_run_seed(config, seed)

    manager = SeedManager(run_seed)
    for dataset_index in range(num_datasets):
        dataset_seed = manager.child("dataset", dataset_index)
        yield _generation_engine._generate_one_seeded(
            config,
            seed=dataset_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )


def generate_worker_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield only this runtime worker's partition of the global dataset index space."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _generation_context._resolve_device(config, device)
    run_seed = _generation_context._resolve_run_seed(config, seed)

    for _dataset_index, dataset_seed in iter_worker_dataset_seeds(
        run_seed=run_seed,
        num_datasets=num_datasets,
        worker_count=int(config.runtime.worker_count),
        worker_index=int(config.runtime.worker_index),
    ):
        yield _generation_engine._generate_one_seeded(
            config,
            seed=dataset_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
