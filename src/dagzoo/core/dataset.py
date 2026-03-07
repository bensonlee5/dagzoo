"""Seeded synthetic dataset generation public entrypoints."""

from __future__ import annotations

from collections.abc import Iterator

from dagzoo.config import GeneratorConfig
from dagzoo.core.fixed_layout import (
    FixedLayoutPlan,
    generate_batch_fixed_layout,
    generate_batch_fixed_layout_iter,
    prepare_canonical_fixed_layout_run,
    sample_fixed_layout,
)
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


def _validate_public_generation_config(config: GeneratorConfig) -> None:
    """Reject public generation configs that are intentionally unsupported."""

    if bool(config.filter.enabled):
        raise ValueError(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation."
        )


def _annotate_canonical_batch_metadata(
    bundle: DatasetBundle,
    *,
    run_seed: int,
    dataset_index: int,
    run_num_datasets: int,
) -> DatasetBundle:
    """Rewrite canonical bundle metadata to preserve run-level replay information."""

    dataset_seed = bundle.metadata.get("seed")
    if not isinstance(dataset_seed, int) or isinstance(dataset_seed, bool):
        dataset_seed = SeedManager(int(run_seed)).child("dataset", int(dataset_index))
    bundle.metadata["seed"] = int(run_seed)
    bundle.metadata["dataset_seed"] = int(dataset_seed)
    bundle.metadata["dataset_index"] = int(dataset_index)
    bundle.metadata["run_num_datasets"] = int(run_num_datasets)
    return bundle


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
    for dataset_index, bundle in enumerate(
        generate_batch_fixed_layout_iter(
            prepared.config,
            plan=prepared.plan,
            num_datasets=num_datasets,
            seed=prepared.run_seed,
            batch_size=prepared.batch_size,
            device=prepared.requested_device,
        )
    ):
        yield _annotate_canonical_batch_metadata(
            bundle,
            run_seed=prepared.run_seed,
            dataset_index=dataset_index,
            run_num_datasets=num_datasets,
        )
