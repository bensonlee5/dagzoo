import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch_iter, generate_worker_batch_iter
from dagzoo.core.worker_partition import (
    iter_worker_dataset_indices,
    iter_worker_dataset_seeds,
)
from dagzoo.rng import SeedManager


def test_worker_partition_round_robin_indices_cover_global_space() -> None:
    num_datasets = 10
    worker_count = 3
    partitions = [
        set(
            iter_worker_dataset_indices(
                num_datasets=num_datasets,
                worker_count=worker_count,
                worker_index=worker_index,
            )
        )
        for worker_index in range(worker_count)
    ]

    assert partitions[0] == {0, 3, 6, 9}
    assert partitions[1] == {1, 4, 7}
    assert partitions[2] == {2, 5, 8}
    assert set().union(*partitions) == set(range(num_datasets))
    for i in range(worker_count):
        for j in range(i + 1, worker_count):
            assert partitions[i].isdisjoint(partitions[j])


def test_worker_partition_seed_derivation_is_global_index_stable() -> None:
    run_seed = 222
    num_datasets = 13

    single_worker_map = {
        dataset_index: dataset_seed
        for dataset_index, dataset_seed in iter_worker_dataset_seeds(
            run_seed=run_seed,
            num_datasets=num_datasets,
            worker_count=1,
            worker_index=0,
        )
    }

    multi_worker_map: dict[int, int] = {}
    for worker_index in range(4):
        for dataset_index, dataset_seed in iter_worker_dataset_seeds(
            run_seed=run_seed,
            num_datasets=num_datasets,
            worker_count=4,
            worker_index=worker_index,
        ):
            multi_worker_map[dataset_index] = dataset_seed

    assert multi_worker_map == single_worker_map


@pytest.mark.parametrize(
    ("num_datasets", "worker_count", "worker_index", "error_pattern"),
    [
        (-1, 1, 0, r"num_datasets must be >= 0"),
        (1, 0, 0, r"worker_count must be >= 1"),
        (1, 1, -1, r"worker_index must be >= 0"),
        (1, 2, 2, r"worker_index must be < worker_count"),
    ],
)
def test_worker_partition_rejects_invalid_inputs(
    num_datasets: int,
    worker_count: int,
    worker_index: int,
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        list(
            iter_worker_dataset_indices(
                num_datasets=num_datasets,
                worker_count=worker_count,
                worker_index=worker_index,
            )
        )


def test_generate_batch_iter_ignores_worker_partitioning(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 3
    cfg.runtime.worker_index = 1
    cfg.runtime.device = "cpu"

    observed_seeds: list[int] = []

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        observed_seeds.append(seed)
        return seed

    monkeypatch.setattr(
        "dagzoo.core.dataset._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    produced = list(generate_batch_iter(cfg, num_datasets=4, seed=777, device="cpu"))
    expected_indices = [0, 1, 2, 3]
    manager = SeedManager(777)
    expected_seeds = [manager.child("dataset", idx) for idx in expected_indices]

    assert observed_seeds == expected_seeds
    assert produced == expected_seeds


def test_generate_worker_batch_iter_respects_worker_partitioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 3
    cfg.runtime.worker_index = 1
    cfg.runtime.device = "cpu"

    observed_seeds: list[int] = []

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        observed_seeds.append(seed)
        return seed

    monkeypatch.setattr(
        "dagzoo.core.dataset._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    produced = list(generate_worker_batch_iter(cfg, num_datasets=10, seed=777, device="cpu"))
    expected_indices = [1, 4, 7]
    manager = SeedManager(777)
    expected_seeds = [manager.child("dataset", idx) for idx in expected_indices]

    assert observed_seeds == expected_seeds
    assert produced == expected_seeds
