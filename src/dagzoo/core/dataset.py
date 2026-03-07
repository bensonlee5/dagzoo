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
    prepare_canonical_fixed_layout_run,
    sample_fixed_layout,
)
from dagzoo.core.worker_partition import iter_worker_dataset_seeds
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


def _validate_public_generation_config(config: GeneratorConfig) -> None:
    """Reject public generation configs that are intentionally unsupported."""

    if bool(config.filter.enabled):
        raise ValueError(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation."
        )
    if int(config.runtime.worker_count) > 1:
        raise ValueError(
            "runtime.worker_count > 1 is not supported for dagzoo generate. "
            "Multi-worker generation is temporarily disabled while canonical fixed-layout "
            "batch optimization is in progress."
        )


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle using the canonical fixed-layout run model."""

    return next(generate_batch_iter(config, num_datasets=1, seed=seed, device=device))


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
    """Yield datasets lazily using one fixed-layout plan sampled for the full run."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return
    _validate_public_generation_config(config)

    prepared = prepare_canonical_fixed_layout_run(
        config,
        num_datasets=num_datasets,
        seed=seed,
        device=device,
    )
    yield from generate_batch_fixed_layout_iter(
        prepared.config,
        plan=prepared.plan,
        num_datasets=num_datasets,
        seed=prepared.run_seed,
        batch_size=prepared.batch_size,
        device=prepared.requested_device,
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
