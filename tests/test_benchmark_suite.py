import numpy as np
from pathlib import Path

import pytest

import cauchy_generator.bench.suite as suite_mod
from cauchy_generator.bench.micro import run_microbenchmarks
from cauchy_generator.bench.suite import ProfileRunSpec, run_benchmark_suite
from cauchy_generator.config import GeneratorConfig
from cauchy_generator.types import DatasetBundle


def _tiny_cpu_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.runtime.device = "cpu"
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


def _tiny_missingness_cpu_config() -> GeneratorConfig:
    cfg = _tiny_cpu_config()
    cfg.dataset.missing_rate = 0.25
    cfg.dataset.missing_mechanism = "mcar"  # type: ignore[assignment]
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
        collect_diagnostics=False,
        diagnostics_root_dir=None,
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


def test_run_benchmark_suite_missingness_guardrails_emit_metrics() -> None:
    cfg = _tiny_missingness_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=100.0,
        fail_threshold_pct=200.0,
        baseline_payload=None,
        num_datasets_override=4,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    result = summary["profile_results"][0]
    guardrails = result["missingness_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "pass"
    assert guardrails["metadata_coverage_rate"] == pytest.approx(1.0)
    assert 0.0 <= guardrails["realized_rate_overall"] <= 1.0
    assert guardrails["runtime_baseline_datasets_per_minute"] > 0.0


def test_run_benchmark_suite_missingness_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_missingness_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")
    calls: list[dict[str, bool]] = []

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        missing_enabled = float(config.dataset.missing_rate) > 0.0
        calls.append(
            {
                "missing_enabled": missing_enabled,
                "has_callback": on_bundle is not None,
            }
        )
        dpm = 80.0 if missing_enabled else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None and missing_enabled:
            for i in range(num_datasets):
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata={
                            "seed": i,
                            "attempt_used": 0,
                            "missingness": {"missing_count_overall": 4},
                        },
                    )
                )
        return {
            "profile": config.benchmark.profile_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": dpm >= 100.0,
        }

    monkeypatch.setattr("cauchy_generator.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "cauchy_generator.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=1,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    result = summary["profile_results"][0]
    guardrails = result["missingness_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "missingness_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "missingness_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )
    baseline_calls = [call for call in calls if not call["missing_enabled"]]
    assert baseline_calls
    assert all(call["has_callback"] for call in baseline_calls)


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


def test_run_benchmark_suite_sanitizes_profile_key_for_diagnostics_paths(tmp_path) -> None:
    cfg = _tiny_cpu_config()
    spec = ProfileRunSpec(key="../../escape", config=cfg, device="cpu")
    diagnostics_root = tmp_path / "diag_root"

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
        collect_diagnostics=True,
        diagnostics_root_dir=diagnostics_root,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    result = summary["profile_results"][0]
    artifacts = result["diagnostics_artifacts"]
    assert isinstance(artifacts, dict)

    json_path = Path(artifacts["json"]).resolve()
    md_path = Path(artifacts["markdown"]).resolve()
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.is_relative_to(diagnostics_root.resolve())
    assert md_path.is_relative_to(diagnostics_root.resolve())
    assert not (tmp_path / "escape" / "coverage_summary.json").exists()


def test_run_benchmark_suite_uses_unique_diagnostics_dirs_for_duplicate_profile_keys(
    tmp_path,
) -> None:
    cfg_a = _tiny_cpu_config()
    cfg_b = _tiny_cpu_config()
    specs = [
        ProfileRunSpec(key="cpu_test", config=cfg_a, device="cpu"),
        ProfileRunSpec(key="cpu_test", config=cfg_b, device="cpu"),
    ]
    diagnostics_root = tmp_path / "diag_root"

    summary = run_benchmark_suite(
        specs,
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=True,
        diagnostics_root_dir=diagnostics_root,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    results = summary["profile_results"]
    assert len(results) == 2
    art_0 = results[0]["diagnostics_artifacts"]
    art_1 = results[1]["diagnostics_artifacts"]
    assert isinstance(art_0, dict)
    assert isinstance(art_1, dict)
    assert art_0["json"] != art_1["json"]
    assert art_0["markdown"] != art_1["markdown"]
    assert Path(art_0["json"]).exists()
    assert Path(art_1["json"]).exists()
    assert Path(art_0["markdown"]).exists()
    assert Path(art_1["markdown"]).exists()
