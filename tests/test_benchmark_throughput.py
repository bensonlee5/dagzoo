import queue
import threading
import typing

import dagzoo.core.parallel_generation as parallel_generation_mod
from dagzoo.bench.throughput import run_throughput_benchmark
from dagzoo.config import GeneratorConfig
from dagzoo.core.parallel_generation import generate_parallel_batch_iter
from dagzoo.rng import offset_seed32


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


def test_run_throughput_benchmark_uses_sequential_generation_for_one_active_worker(
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
        yield from range(num_datasets)

    def _unexpected_generate_parallel_batch_iter(*args, **kwargs):
        raise AssertionError(
            "parallel generator should not be used when only one worker partition is active"
        )

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_parallel_batch_iter",
        _unexpected_generate_parallel_batch_iter,
    )

    cfg = GeneratorConfig()
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    result = run_throughput_benchmark(
        cfg,
        num_datasets=1,
        warmup_datasets=1,
        device="cpu",
    )

    assert calls == [
        (1, offset_seed32(cfg.seed, 1), "cpu"),
        (1, offset_seed32(cfg.seed, 2), "cpu"),
    ]
    assert result["num_datasets"] == 1


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
