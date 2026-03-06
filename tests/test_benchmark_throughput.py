import queue
import threading
import typing

import dagzoo.core.parallel_generation as parallel_generation_mod
import pytest
from dagzoo.bench.throughput import run_throughput_benchmark
from dagzoo.config import GeneratorConfig
from dagzoo.core.parallel_generation import (
    ParallelGenerationConfigError,
    generate_parallel_batch_iter,
)
from dagzoo.rng import SeedManager, offset_seed32


def test_run_throughput_benchmark_uses_streaming_generation(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int, str | None]] = []

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        calls.append((num_datasets, int(seed or 0), device))
        for _ in range(num_datasets):
            yield None

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    cfg = GeneratorConfig()
    result = run_throughput_benchmark(
        cfg,
        num_datasets=3,
        warmup_datasets=2,
        device="cpu",
    )

    assert calls == [
        (2, offset_seed32(cfg.seed, 1), "cpu"),
        (3, offset_seed32(cfg.seed, 2), "cpu"),
    ]
    assert result["num_datasets"] == 3
    assert result["warmup_datasets"] == 2

    assert float(typing.cast(float, result["datasets_per_minute"])) >= 0.0


def test_run_throughput_benchmark_updates_callback_on_measured_generation(
    monkeypatch,
) -> None:
    observed: list[int] = []

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    cfg = GeneratorConfig()
    run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="cpu",
        on_bundle=lambda bundle: observed.append(int(bundle)),
    )
    assert observed == [0, 1, 2, 3]


def test_run_throughput_benchmark_uses_parallel_generation_for_multi_worker_cpu(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int, str | None]] = []
    observed: list[int] = []

    def _stub_generate_parallel_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        calls.append((num_datasets, int(seed or 0), device))
        yield from range(num_datasets)

    def _unexpected_generate_batch_iter(*args, **kwargs):
        raise AssertionError("sequential generator should not be used for multi-worker throughput")

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_parallel_batch_iter",
        _stub_generate_parallel_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _unexpected_generate_batch_iter,
    )

    cfg = GeneratorConfig()
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    result = run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="cpu",
        on_bundle=lambda bundle: observed.append(int(bundle)),
    )

    assert calls == [
        (2, offset_seed32(cfg.seed, 1), "cpu"),
        (4, offset_seed32(cfg.seed, 2), "cpu"),
    ]
    assert observed == [0, 1, 2, 3]
    assert result["num_datasets"] == 4


def test_run_throughput_benchmark_callback_exception_does_not_hang_parallel_path(
    monkeypatch,
) -> None:
    def _stub_generate_parallel_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        yield from generate_parallel_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
            max_buffered_results=1,
        )

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        return int(seed)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_parallel_batch_iter",
        _stub_generate_parallel_batch_iter,
    )
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "get_num_threads",
        lambda: 2,
    )
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "set_num_threads",
        lambda _value: None,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    cfg = GeneratorConfig()
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    result_queue: queue.Queue[BaseException | None] = queue.Queue()

    def _run_benchmark() -> None:
        try:
            run_throughput_benchmark(
                cfg,
                num_datasets=6,
                warmup_datasets=0,
                device="cpu",
                on_bundle=lambda _bundle: (_ for _ in ()).throw(RuntimeError("callback boom")),
            )
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            result_queue.put(exc)
            return
        result_queue.put(None)

    benchmark_thread = threading.Thread(target=_run_benchmark, daemon=True)
    benchmark_thread.start()
    benchmark_thread.join(timeout=1.0)

    assert not benchmark_thread.is_alive()
    error = result_queue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "callback boom"


def test_run_throughput_benchmark_preserves_ordered_prefix_before_later_worker_error(
    monkeypatch,
) -> None:
    def _stub_generate_parallel_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        yield from generate_parallel_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
            max_buffered_results=2,
        )

    cfg = GeneratorConfig()
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    manager = SeedManager(offset_seed32(cfg.seed, 2))
    seed0 = manager.child("dataset", 0)
    seed1 = manager.child("dataset", 1)
    seed2 = manager.child("dataset", 2)
    seed3 = manager.child("dataset", 3)

    release_second_callback = threading.Event()
    dataset2_started = threading.Event()
    worker_one_failed = threading.Event()
    observed: list[int] = []
    result_queue: queue.Queue[BaseException | None] = queue.Queue()

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        if seed == seed2:
            dataset2_started.set()
            return int(seed)
        if seed == seed3:
            assert dataset2_started.wait(timeout=1.0)
            worker_one_failed.set()
            raise RuntimeError("boom")
        return int(seed)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_parallel_batch_iter",
        _stub_generate_parallel_batch_iter,
    )
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "get_num_threads",
        lambda: 2,
    )
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "set_num_threads",
        lambda _value: None,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    def _on_bundle(bundle: int) -> None:
        observed.append(int(bundle))
        if int(bundle) == seed1:
            assert worker_one_failed.wait(timeout=1.0)
            assert release_second_callback.wait(timeout=1.0)

    def _run_benchmark() -> None:
        try:
            run_throughput_benchmark(
                cfg,
                num_datasets=4,
                warmup_datasets=0,
                device="cpu",
                on_bundle=_on_bundle,
            )
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            result_queue.put(exc)
            return
        result_queue.put(None)

    benchmark_thread = threading.Thread(target=_run_benchmark, daemon=True)
    benchmark_thread.start()

    assert dataset2_started.wait(timeout=1.0)
    assert worker_one_failed.wait(timeout=1.0)

    release_second_callback.set()
    benchmark_thread.join(timeout=1.0)

    assert not benchmark_thread.is_alive()
    error = result_queue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "boom"
    assert observed == [seed0, seed1, seed2]


def test_run_throughput_benchmark_rejects_thread_limited_multi_worker_cpu(
    monkeypatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "get_num_threads",
        lambda: 1,
    )
    set_calls: list[int] = []
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "set_num_threads",
        lambda value: set_calls.append(int(value)),
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        lambda *_args, **_kwargs: pytest.fail("generation should not start when thread-limited"),
    )

    with pytest.raises(
        ParallelGenerationConfigError,
        match=r"active worker count",
    ):
        run_throughput_benchmark(
            cfg,
            num_datasets=4,
            warmup_datasets=0,
            device="cpu",
        )
    assert set_calls == []


def test_run_throughput_benchmark_allows_small_multi_worker_batch_on_thread_limited_cpu(
    monkeypatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.runtime.worker_count = 4
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "get_num_threads",
        lambda: 1,
    )
    set_calls: list[int] = []
    monkeypatch.setattr(
        parallel_generation_mod.torch,
        "set_num_threads",
        lambda value: set_calls.append(int(value)),
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        lambda _config, *, seed, requested_device, resolved_device: int(seed),
    )

    observed: list[int] = []
    result = run_throughput_benchmark(
        cfg,
        num_datasets=1,
        warmup_datasets=1,
        device="cpu",
        on_bundle=lambda bundle: observed.append(int(bundle)),
    )

    assert observed
    assert result["num_datasets"] == 1
    assert result["warmup_datasets"] == 1
    assert set_calls == []
