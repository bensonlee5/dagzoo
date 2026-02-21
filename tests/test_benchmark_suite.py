import numpy as np

import cauchy_generator.bench.suite as suite_mod
from cauchy_generator.bench.micro import run_microbenchmarks
from cauchy_generator.bench.suite import ProfileRunSpec, run_benchmark_suite
from cauchy_generator.config import GeneratorConfig
from cauchy_generator.types import DatasetBundle


def _tiny_cpu_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.runtime.device = "cpu"
    cfg.runtime.prefer_torch = False
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    cfg.benchmark.num_datasets = 2
    cfg.benchmark.warmup_datasets = 0
    cfg.benchmark.latency_num_samples = 2
    cfg.benchmark.reproducibility_num_datasets = 1
    cfg.benchmark.profile_name = "cpu_test"
    cfg.benchmark.profiles["cpu_test"] = {
        "device": "cpu",
        "num_datasets": 2,
        "warmup_datasets": 0,
    }
    return cfg


def test_run_benchmark_suite_smoke_single_profile() -> None:
    cfg = _tiny_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    assert summary["suite"] == "smoke"
    assert summary["regression"]["status"] == "pass"
    assert len(summary["profile_results"]) == 1

    result = summary["profile_results"][0]
    assert result["profile_key"] == "cpu_test"
    assert result["datasets_per_minute"] > 0
    assert result["latency_p95_ms"] >= 0


def test_run_microbenchmarks_returns_expected_keys() -> None:
    cfg = _tiny_cpu_config()
    res = run_microbenchmarks(cfg, device="cpu", repeats=1)
    assert res["micro_repeats"] == 1
    assert "micro_random_function_linear_ms" in res
    assert "micro_node_pipeline_ms" in res
    assert "micro_generate_one_ms" in res


def test_collect_reproducibility_uses_streaming_generation(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int, str | None]] = []

    def _bundle(value: int) -> DatasetBundle:
        x_train = np.full((2, 2), float(value), dtype=np.float32)
        y_train = np.array([0, 1], dtype=np.int64)
        x_test = np.full((1, 2), float(value), dtype=np.float32)
        y_test = np.array([1], dtype=np.int64)
        return DatasetBundle(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            feature_types=["num", "num"],
            metadata={"seed": value, "attempt_used": 0},
        )

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        calls.append((num_datasets, int(seed or 0), device))
        for i in range(num_datasets):
            yield _bundle(int(seed or 0) + i)

    monkeypatch.setattr(
        "cauchy_generator.bench.suite.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    cfg = _tiny_cpu_config()
    out = suite_mod._collect_reproducibility(cfg, device="cpu", num_datasets=3)
    assert out["reproducibility_datasets"] == 3
    assert out["reproducibility_match"] is True
    assert len(calls) == 2
    assert calls[0] == calls[1]
