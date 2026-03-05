"""Deterministic worker partition helpers for dataset index/seed assignment."""

from __future__ import annotations

from collections.abc import Iterator

from dagzoo.rng import SeedManager


def _validate_worker_partition_inputs(
    *,
    num_datasets: int,
    worker_count: int,
    worker_index: int,
) -> None:
    """Validate worker-partition inputs for deterministic index assignment."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}.")
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}.")
    if worker_index < 0:
        raise ValueError(f"worker_index must be >= 0, got {worker_index}.")
    if worker_index >= worker_count:
        raise ValueError(
            f"worker_index must be < worker_count, got {worker_index} >= {worker_count}."
        )


def iter_worker_dataset_indices(
    *,
    num_datasets: int,
    worker_count: int,
    worker_index: int,
) -> Iterator[int]:
    """Yield global dataset indices for one worker using round-robin partitioning."""

    _validate_worker_partition_inputs(
        num_datasets=num_datasets,
        worker_count=worker_count,
        worker_index=worker_index,
    )
    yield from range(worker_index, num_datasets, worker_count)


def iter_worker_dataset_seeds(
    *,
    run_seed: int,
    num_datasets: int,
    worker_count: int,
    worker_index: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(dataset_index, dataset_seed)`` pairs for one worker partition."""

    manager = SeedManager(run_seed)
    for dataset_index in iter_worker_dataset_indices(
        num_datasets=num_datasets,
        worker_count=worker_count,
        worker_index=worker_index,
    ):
        yield dataset_index, manager.child("dataset", dataset_index)
