import queue
import threading
import typing

from dagzoo.bench.throughput import run_throughput_benchmark
from dagzoo.config import GeneratorConfig
from dagzoo.rng import offset_seed32


def _tiny_parallel_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.benchmark.preset_name = "parallel_test"
    return cfg


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
        on_bundle=lambda bundle: observed.append(typing.cast(int, bundle)),
    )
    assert observed == [0, 1, 2, 3]


def test_run_throughput_benchmark_uses_sequential_generation(
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

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    cfg = GeneratorConfig()
    cfg.runtime.device = "cpu"

    result = run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="cpu",
    )

    assert calls == [
        (2, offset_seed32(cfg.seed, 1), "cpu"),
        (4, offset_seed32(cfg.seed, 2), "cpu"),
    ]
    assert result["num_datasets"] == 4


def test_run_throughput_benchmark_callback_exception_does_not_hang_parallel_path() -> None:
    cfg = _tiny_parallel_config()

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
    benchmark_thread.join(timeout=5.0)

    assert not benchmark_thread.is_alive()
    error = result_queue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "callback boom"
