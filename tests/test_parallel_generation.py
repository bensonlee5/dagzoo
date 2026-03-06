import queue
import threading
import time

import dagzoo.core.parallel_generation as parallel_generation_mod
import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.core.parallel_generation import (
    ParallelGenerationConfigError,
    generate_parallel_batch_iter,
)
from dagzoo.rng import SeedManager
from dagzoo.types import DatasetBundle


def _patch_torch_thread_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_threads: int,
) -> list[int]:
    set_calls: list[int] = []
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "get_num_threads",
        lambda: num_threads,
    )
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "set_num_threads",
        lambda value: set_calls.append(int(value)),
    )
    return set_calls


def test_generate_parallel_batch_iter_preserves_global_seed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 3
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    set_calls = _patch_torch_thread_settings(monkeypatch, num_threads=8)

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
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=7, seed=777, device="cpu"))
    manager = SeedManager(777)
    expected = [manager.child("dataset", idx) for idx in range(7)]

    assert produced == expected
    assert sorted(observed_seeds) == sorted(expected)
    assert set_calls == [2, 8]


def test_generate_parallel_batch_iter_preserves_root_worker_index_in_bundle_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 3
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    _ = _patch_torch_thread_settings(monkeypatch, num_threads=6)

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = seed
        _ = requested_device
        _ = resolved_device
        return DatasetBundle(
            X_train=None,
            y_train=None,
            X_test=None,
            y_test=None,
            feature_types=[],
            metadata={"config": {"runtime": {"worker_index": int(config.runtime.worker_index)}}},
        )

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=5, seed=777, device="cpu"))

    assert len(produced) == 5
    assert all(bundle.metadata["config"]["runtime"]["worker_index"] == 0 for bundle in produced)


def test_generate_parallel_batch_iter_yields_nothing_for_zero_datasets() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    assert list(generate_parallel_batch_iter(cfg, num_datasets=0, seed=777, device="cpu")) == []


def test_generate_parallel_batch_iter_propagates_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    set_calls = _patch_torch_thread_settings(monkeypatch, num_threads=8)

    failing_seed = SeedManager(777).child("dataset", 2)

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        if seed == failing_seed:
            raise RuntimeError("boom")
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    with pytest.raises(RuntimeError, match="boom"):
        list(generate_parallel_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))
    assert set_calls == [4, 8]


def test_generate_parallel_batch_iter_propagates_worker_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    deepcopy_calls = 0
    original_deepcopy = parallel_generation_mod.copy.deepcopy

    def _patched_deepcopy(value):
        nonlocal deepcopy_calls
        deepcopy_calls += 1
        if deepcopy_calls == 2:
            raise RuntimeError("copy boom")
        return original_deepcopy(value)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation.copy.deepcopy",
        _patched_deepcopy,
    )

    with pytest.raises(RuntimeError, match="copy boom"):
        list(generate_parallel_batch_iter(cfg, num_datasets=4, seed=777, device="cpu"))


def test_generate_parallel_batch_iter_rejects_single_worker_request() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 1
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    with pytest.raises(ValueError, match=r"runtime\.worker_count > 1"):
        list(generate_parallel_batch_iter(cfg, num_datasets=1, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_nonzero_worker_index() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 1
    cfg.runtime.device = "cpu"

    with pytest.raises(ValueError, match=r"runtime\.worker_index == 0"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_non_cpu_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_context._resolve_device",
        lambda _config, _device: "cuda",
    )

    with pytest.raises(ValueError, match=r"supports resolved device 'cpu' only"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="auto"))


def test_generate_parallel_batch_iter_rejects_thread_limited_active_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 4
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    set_calls = _patch_torch_thread_settings(monkeypatch, num_threads=2)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        lambda *_args, **_kwargs: pytest.fail("generation should not start when thread-limited"),
    )

    with pytest.raises(
        ParallelGenerationConfigError,
        match=r"active worker count",
    ):
        list(generate_parallel_batch_iter(cfg, num_datasets=3, seed=777, device="cpu"))
    assert set_calls == []


def test_generate_parallel_batch_iter_handles_single_dataset_with_many_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 4
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    set_calls = _patch_torch_thread_settings(monkeypatch, num_threads=1)

    observed_seeds: list[int] = []
    deepcopy_calls = 0
    original_deepcopy = parallel_generation_mod.copy.deepcopy

    def _patched_deepcopy(value):
        nonlocal deepcopy_calls
        deepcopy_calls += 1
        return original_deepcopy(value)

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        observed_seeds.append(seed)
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation.copy.deepcopy",
        _patched_deepcopy,
    )

    expected_seed = SeedManager(777).child("dataset", 0)

    assert list(generate_parallel_batch_iter(cfg, num_datasets=1, seed=777, device="cpu")) == [
        expected_seed
    ]
    assert observed_seeds == [expected_seed]
    assert deepcopy_calls == 1
    assert set_calls == []


def test_generate_parallel_batch_iter_close_does_not_hang_with_full_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    set_calls = _patch_torch_thread_settings(monkeypatch, num_threads=8)

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
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    iterator = generate_parallel_batch_iter(
        cfg,
        num_datasets=6,
        seed=777,
        device="cpu",
        max_buffered_results=1,
    )
    _ = next(iterator)
    deadline = time.monotonic() + 1.0
    while len(observed_seeds) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert len(observed_seeds) >= 2

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
    close_thread.join(timeout=1.0)

    assert not close_thread.is_alive()
    assert close_result.get_nowait() is None
    assert set_calls == [4, 8]


def test_generate_parallel_batch_iter_rechecks_queue_after_poll_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    _ = _patch_torch_thread_settings(monkeypatch, num_threads=2)

    forced_empty = False
    allow_seed0_emit = threading.Event()
    dataset0_buffered = threading.Event()
    original_put = parallel_generation_mod.queue.Queue.put
    original_get = parallel_generation_mod.queue.Queue.get

    def _patched_put(self, item, block: bool = True, timeout: float | None = None):
        result = original_put(self, item, block=block, timeout=timeout)
        if getattr(item, "dataset_index", None) == 0:
            dataset0_buffered.set()
        return result

    def _patched_get(self, block: bool = True, timeout: float | None = None):
        nonlocal forced_empty
        if timeout == 0.05 and self.maxsize == 1 and not forced_empty:
            allow_seed0_emit.set()
            assert dataset0_buffered.wait(timeout=1.0)
            forced_empty = True
            raise queue.Empty
        return original_get(self, block=block, timeout=timeout)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation.queue.Queue.put",
        _patched_put,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation.queue.Queue.get",
        _patched_get,
    )

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        assert allow_seed0_emit.wait(timeout=1.0)
        return int(seed)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    expected_seed = SeedManager(777).child("dataset", 0)

    assert list(
        generate_parallel_batch_iter(
            cfg,
            num_datasets=1,
            seed=777,
            device="cpu",
            max_buffered_results=1,
        )
    ) == [expected_seed]
    assert forced_empty


def test_generate_parallel_batch_iter_bounds_faster_worker_runahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    manager = SeedManager(777)
    seed0 = manager.child("dataset", 0)
    seed1 = manager.child("dataset", 1)
    seed3 = manager.child("dataset", 3)
    seed5 = manager.child("dataset", 5)

    observed_seeds: list[int] = []
    worker_zero_started = threading.Event()
    worker_one_second_started = threading.Event()
    release_worker_zero = threading.Event()

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        observed_seeds.append(seed)
        if seed == seed0:
            worker_zero_started.set()
            assert release_worker_zero.wait(timeout=1.0)
        if seed == seed3:
            worker_one_second_started.set()
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    iterator = generate_parallel_batch_iter(
        cfg,
        num_datasets=8,
        seed=777,
        device="cpu",
        max_buffered_results=1,
    )
    next_result: queue.Queue[int | BaseException] = queue.Queue()

    def _consume_first() -> None:
        try:
            next_result.put(next(iterator))
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            next_result.put(exc)

    consumer_thread = threading.Thread(target=_consume_first, daemon=True)
    consumer_thread.start()

    assert worker_zero_started.wait(timeout=1.0)
    assert worker_one_second_started.wait(timeout=1.0)
    assert seed1 in observed_seeds
    assert seed3 in observed_seeds
    assert seed5 not in observed_seeds
    assert consumer_thread.is_alive()

    release_worker_zero.set()
    consumer_thread.join(timeout=1.0)

    assert not consumer_thread.is_alive()
    assert next_result.get_nowait() == seed0
    iterator.close()


def test_generate_parallel_batch_iter_defers_later_worker_error_until_target_index_is_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    _ = _patch_torch_thread_settings(monkeypatch, num_threads=2)

    manager = SeedManager(777)
    seed0 = manager.child("dataset", 0)
    seed1 = manager.child("dataset", 1)
    seed2 = manager.child("dataset", 2)
    seed3 = manager.child("dataset", 3)

    dataset2_started = threading.Event()
    release_dataset2 = threading.Event()
    worker_one_failed = threading.Event()

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        if seed == seed2:
            dataset2_started.set()
            assert release_dataset2.wait(timeout=1.0)
            return seed
        if seed == seed3:
            assert dataset2_started.wait(timeout=1.0)
            worker_one_failed.set()
            raise RuntimeError("boom")
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    iterator = generate_parallel_batch_iter(
        cfg,
        num_datasets=4,
        seed=777,
        device="cpu",
        max_buffered_results=2,
    )
    assert next(iterator) == seed0
    assert next(iterator) == seed1

    next_result: queue.Queue[int | BaseException] = queue.Queue()

    def _consume_next() -> None:
        try:
            next_result.put(next(iterator))
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            next_result.put(exc)

    consumer_thread = threading.Thread(target=_consume_next, daemon=True)
    consumer_thread.start()

    assert dataset2_started.wait(timeout=1.0)
    assert worker_one_failed.wait(timeout=1.0)
    assert consumer_thread.is_alive()

    release_dataset2.set()
    consumer_thread.join(timeout=1.0)

    assert not consumer_thread.is_alive()
    assert next_result.get_nowait() == seed2
    with pytest.raises(RuntimeError, match="boom"):
        next(iterator)
    iterator.close()


def test_generate_parallel_batch_iter_preserves_original_worker_exception_on_timeout_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    manager = SeedManager(777)
    seed1 = manager.child("dataset", 1)
    bundle_get_calls = 0
    target_wait_started = threading.Event()
    original_queue_get = parallel_generation_mod.queue.Queue.get

    def _patched_queue_get(self, block: bool = True, timeout: float | None = None):
        nonlocal bundle_get_calls
        if timeout == 0.05 and self.maxsize == 1:
            bundle_get_calls += 1
            if bundle_get_calls == 2:
                target_wait_started.set()
        return original_queue_get(self, block=block, timeout=timeout)

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        if seed == seed1:
            assert target_wait_started.wait(timeout=1.0)
            raise RuntimeError("boom")
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation.queue.Queue.get",
        _patched_queue_get,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    with pytest.raises(RuntimeError, match="boom"):
        list(
            generate_parallel_batch_iter(
                cfg,
                num_datasets=4,
                seed=777,
                device="cpu",
                max_buffered_results=1,
            )
        )
