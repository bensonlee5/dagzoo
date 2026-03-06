import queue
import threading
from types import SimpleNamespace

import dagzoo.core.parallel_generation as parallel_generation_mod
import numpy as np
import pytest

from dagzoo.bench.metrics import reproducibility_signature
from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core.parallel_generation import (
    ParallelGenerationWorkerError,
    generate_parallel_batch_iter,
)


def _tiny_parallel_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.benchmark.preset_name = "parallel_test"
    return cfg


def _ipc_bundle_payload(value: int) -> dict[str, object]:
    return {
        "X_train": np.full((2, 2), float(value), dtype=np.float32),
        "y_train": np.array([0, 1], dtype=np.int64),
        "X_test": np.full((1, 2), float(value), dtype=np.float32),
        "y_test": np.array([1], dtype=np.int64),
        "feature_types": ["num", "num"],
        "metadata": {"seed": value, "attempt_used": 0},
    }


class _FakeEvent:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


class _FakeSemaphore:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _FakeResultQueue:
    def __init__(self, semaphore: _FakeSemaphore) -> None:
        self._messages = [
            parallel_generation_mod._WorkerResultMessage(
                worker_index=1,
                dataset_index=1,
                bundle_payload=_ipc_bundle_payload(1),
            ),
            parallel_generation_mod._WorkerResultMessage(
                worker_index=0,
                dataset_index=0,
                bundle_payload=_ipc_bundle_payload(0),
            ),
        ]
        self._get_calls = 0
        self._semaphore = semaphore

    def get(self, timeout: float | None = None):
        _ = timeout
        if self._get_calls == 1 and self._semaphore.release_calls == 0:
            raise AssertionError("result slot should be released immediately after dequeue")
        self._get_calls += 1
        if not self._messages:
            raise queue.Empty
        return self._messages.pop(0)


class _FakeControlQueue:
    def get_nowait(self):
        raise queue.Empty


class _FakeSpawnContext:
    def __init__(
        self,
        *,
        result_queue: _FakeResultQueue,
        control_queue: _FakeControlQueue,
        semaphore: _FakeSemaphore,
        event: _FakeEvent,
    ) -> None:
        self._result_queue = result_queue
        self._control_queue = control_queue
        self._semaphore = semaphore
        self._event = event

    def Queue(self, maxsize: int | None = None):
        return self._result_queue if maxsize is not None else self._control_queue

    def BoundedSemaphore(self, value: int):
        _ = value
        return self._semaphore

    def Event(self):
        return self._event


def test_generate_parallel_batch_iter_matches_serial_output_and_seed_order() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 3

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))
    serial = list(generate_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))
    expected_seeds = [int(bundle.metadata["seed"]) for bundle in serial]

    assert [int(bundle.metadata["seed"]) for bundle in produced] == expected_seeds
    assert reproducibility_signature(produced) == reproducibility_signature(serial)


def test_generate_parallel_batch_iter_preserves_root_worker_index_in_bundle_metadata() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 3

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=4, seed=777, device="cpu"))

    assert len(produced) == 4
    assert all(bundle.metadata["config"]["runtime"]["worker_index"] == 0 for bundle in produced)


def test_generate_parallel_batch_iter_yields_nothing_for_zero_datasets() -> None:
    cfg = _tiny_parallel_config()

    assert list(generate_parallel_batch_iter(cfg, num_datasets=0, seed=777, device="cpu")) == []


def test_generate_parallel_batch_iter_propagates_worker_error() -> None:
    cfg = _tiny_parallel_config()
    cfg.filter.enabled = True

    with pytest.raises(ParallelGenerationWorkerError, match="filter.enabled"):
        list(generate_parallel_batch_iter(cfg, num_datasets=4, seed=777, device="cpu"))


def test_generate_parallel_batch_iter_rejects_single_worker_request() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 1

    with pytest.raises(ValueError, match=r"runtime\.worker_count > 1"):
        list(generate_parallel_batch_iter(cfg, num_datasets=1, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_nonzero_worker_index() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_index = 1

    with pytest.raises(ValueError, match=r"runtime\.worker_index == 0"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_non_cpu_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_context._resolve_device",
        lambda _config, _device: "cuda",
    )

    with pytest.raises(ValueError, match=r"supports resolved device 'cpu' only"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="auto"))


def test_generate_parallel_batch_iter_caps_local_worker_count_to_cpu_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 64
    spawned_worker_counts: list[int] = []
    original_spawn = parallel_generation_mod._spawn_parallel_workers

    def _recording_spawn_parallel_workers(**kwargs):
        spawned_worker_counts.append(len(kwargs["worker_specs"]))
        return original_spawn(**kwargs)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._local_parallel_worker_capacity",
        lambda: 2,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._spawn_parallel_workers",
        _recording_spawn_parallel_workers,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))

    assert len(produced) == 6
    assert spawned_worker_counts == [2]


def test_generate_parallel_batch_iter_handles_single_dataset_with_many_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 64
    spawned_worker_counts: list[int] = []
    original_spawn = parallel_generation_mod._spawn_parallel_workers

    def _recording_spawn_parallel_workers(**kwargs):
        spawned_worker_counts.append(len(kwargs["worker_specs"]))
        return original_spawn(**kwargs)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._spawn_parallel_workers",
        _recording_spawn_parallel_workers,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=1, seed=777, device="cpu"))
    serial = list(generate_batch_iter(cfg, num_datasets=1, seed=777, device="cpu"))

    assert len(produced) == 1
    assert int(produced[0].metadata["seed"]) == int(serial[0].metadata["seed"])
    assert spawned_worker_counts == [1]


def test_generate_parallel_batch_iter_releases_result_slots_before_in_order_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()
    semaphore = _FakeSemaphore()
    result_queue = _FakeResultQueue(semaphore)
    control_queue = _FakeControlQueue()
    event = _FakeEvent()
    ctx = _FakeSpawnContext(
        result_queue=result_queue,
        control_queue=control_queue,
        semaphore=semaphore,
        event=event,
    )

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._build_spawn_context",
        lambda: ctx,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._spawn_parallel_workers",
        lambda **_kwargs: [SimpleNamespace(exitcode=0), SimpleNamespace(exitcode=0)],
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._terminate_processes",
        lambda _processes: None,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._close_process_queue",
        lambda _queue: None,
    )

    produced = list(
        generate_parallel_batch_iter(
            cfg,
            num_datasets=2,
            seed=777,
            device="cpu",
            max_buffered_results=1,
        )
    )

    assert [int(bundle.metadata["seed"]) for bundle in produced] == [0, 1]
    assert semaphore.release_calls == 2


def test_generate_parallel_batch_iter_close_does_not_hang() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 2

    iterator = generate_parallel_batch_iter(
        cfg,
        num_datasets=12,
        seed=777,
        device="cpu",
        max_buffered_results=1,
    )
    _ = next(iterator)

    close_result: queue.Queue[BaseException | None] = queue.Queue()

    def _close_iterator() -> None:
        try:
            iterator.close()
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            close_result.put(exc)
            return
        close_result.put(None)

    close_thread = threading.Thread(target=_close_iterator, daemon=True)
    close_thread.start()
    close_thread.join(timeout=5.0)

    assert not close_thread.is_alive()
    assert close_result.get_nowait() is None
