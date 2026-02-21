import typing

from cauchy_generator.bench.throughput import run_throughput_benchmark
from cauchy_generator.config import GeneratorConfig


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
        "cauchy_generator.bench.throughput.generate_batch_iter",
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
        (2, cfg.seed + 1, "cpu"),
        (3, cfg.seed + 2, "cpu"),
    ]
    assert result["num_datasets"] == 3
    assert result["warmup_datasets"] == 2

    assert float(typing.cast(float, result["datasets_per_minute"])) >= 0.0
