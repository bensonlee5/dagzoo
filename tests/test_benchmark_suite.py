import numpy as np
from pathlib import Path

import pytest

import cauchy_generator.bench.guardrails as guardrails_mod
import cauchy_generator.bench.suite as suite_mod
from cauchy_generator.bench.micro import run_microbenchmarks
from cauchy_generator.bench.report import write_suite_markdown
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


def _tiny_shift_cpu_config() -> GeneratorConfig:
    cfg = _tiny_cpu_config()
    cfg.shift.enabled = True
    cfg.shift.profile = "mixed"
    cfg.graph.n_nodes_min = 8
    cfg.graph.n_nodes_max = 12
    return cfg


def _tiny_noise_cpu_config() -> GeneratorConfig:
    cfg = _tiny_cpu_config()
    cfg.noise.family = "laplace"
    cfg.noise.scale = 1.0
    cfg.noise.student_t_df = 6.0
    cfg.noise.mixture_weights = None
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
    assert summary["regression"]["status"] in {"pass", "warn", "fail"}
    if summary["regression"]["status"] != "pass":
        assert any(
            issue["metric"] == "lineage_export_runtime_degradation_pct"
            for issue in summary["regression"]["issues"]
        )
    assert len(summary["profile_results"]) == 1

    result = summary["profile_results"][0]
    assert result["profile_key"] == "cpu_test"
    assert result["datasets_per_minute"] > 0
    assert result["latency_p95_ms"] >= 0
    lineage_guardrails = result["lineage_guardrails"]
    assert lineage_guardrails["enabled"] is True
    assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


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
    lineage_guardrails = result["lineage_guardrails"]
    assert lineage_guardrails["enabled"] is True
    assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


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


def test_run_benchmark_suite_shift_guardrails_emit_metrics() -> None:
    cfg = _tiny_shift_cpu_config()
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

    result = summary["profile_results"][0]
    guardrails = result["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["profile"] == "mixed"
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["directional_gating_enabled"] is False
    assert "directional_checks" in guardrails


def test_run_benchmark_suite_shift_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_shift_cpu_config()
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
        shift_enabled = bool(config.shift.enabled)
        calls.append(
            {
                "shift_enabled": shift_enabled,
                "has_callback": on_bundle is not None,
            }
        )
        dpm = 70.0 if shift_enabled else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {
                    "seed": i,
                    "attempt_used": 0,
                    "graph_edge_density": 0.35 if shift_enabled else 0.20,
                    "shift": {
                        "enabled": shift_enabled,
                        "edge_odds_multiplier": 1.3 if shift_enabled else 1.0,
                        "mechanism_nonlinear_mass": 0.72 if shift_enabled else 0.62,
                        "noise_variance_multiplier": 1.25 if shift_enabled else 1.0,
                    },
                }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
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
    monkeypatch.setattr(
        "cauchy_generator.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    result = summary["profile_results"][0]
    guardrails = result["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["runtime_gating_enabled"] is True
    assert guardrails["directional_gating_enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "shift_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "shift_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )
    baseline_calls = [call for call in calls if not call["shift_enabled"]]
    assert baseline_calls
    assert all(call["has_callback"] for call in baseline_calls)


def test_run_benchmark_suite_shift_directional_guardrail_failure_updates_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_shift_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")

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
        shift_enabled = bool(config.shift.enabled)
        dpm = 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {
                    "seed": i,
                    "attempt_used": 0,
                    "graph_edge_density": 0.20,
                    "shift": {
                        "enabled": shift_enabled,
                        "edge_odds_multiplier": 1.0,
                        "mechanism_nonlinear_mass": 0.60 if shift_enabled else 0.62,
                        "noise_variance_multiplier": 1.0,
                    },
                }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
                    )
                )
        return {
            "profile": config.benchmark.profile_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": True,
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
    monkeypatch.setattr(
        "cauchy_generator.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    guardrails = summary["profile_results"][0]["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    mechanism_issue = next(
        issue
        for issue in guardrails["issues"]
        if issue["metric"] == "shift_mechanism_nonlinear_mass_directionality"
    )
    expected_degradation = ((0.62 - 0.60) / 0.62) * 100.0
    assert mechanism_issue["degradation_pct"] == pytest.approx(expected_degradation)
    assert summary["regression"]["status"] == "fail"
    regression_issue = next(
        issue
        for issue in summary["regression"]["issues"]
        if issue["metric"] == "shift_mechanism_nonlinear_mass_directionality"
    )
    assert regression_issue["degradation_pct"] == pytest.approx(expected_degradation)


def test_run_benchmark_suite_noise_guardrails_emit_metrics() -> None:
    cfg = _tiny_noise_cpu_config()
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

    result = summary["profile_results"][0]
    guardrails = result["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert guardrails["family_requested"] == "laplace"
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["metadata_coverage_rate"] == pytest.approx(1.0)


def test_run_benchmark_suite_noise_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_noise_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")

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
        nonlegacy_noise = str(config.noise.family) != "legacy"
        dpm = 70.0 if nonlegacy_noise else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {
                    "seed": i,
                    "attempt_used": 0,
                    "noise": {
                        "family_requested": str(config.noise.family),
                        "family_sampled": str(config.noise.family),
                        "sampling_strategy": "dataset_level",
                        "scale": float(config.noise.scale),
                        "student_t_df": float(config.noise.student_t_df),
                        "mixture_weights": None,
                    },
                }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
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
    monkeypatch.setattr(
        "cauchy_generator.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    result = summary["profile_results"][0]
    guardrails = result["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["runtime_gating_enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "noise_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "noise_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )


def test_run_benchmark_suite_noise_metadata_coverage_failure_updates_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_noise_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")

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
        dpm = 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {"seed": i, "attempt_used": 0}
                if str(config.noise.family) == "legacy":
                    metadata["noise"] = {
                        "family_requested": "legacy",
                        "family_sampled": "legacy",
                        "sampling_strategy": "dataset_level",
                        "scale": 1.0,
                        "student_t_df": 5.0,
                        "mixture_weights": None,
                    }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
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
    monkeypatch.setattr(
        "cauchy_generator.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        no_hardware_aware=True,
    )

    result = summary["profile_results"][0]
    guardrails = result["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "noise_metadata_coverage" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "noise_metadata_coverage" for issue in summary["regression"]["issues"]
    )


def test_build_shift_directional_check_reports_none_degradation_when_baseline_zero() -> None:
    payload, issue = suite_mod._build_shift_directional_check(
        metric="graph_edge_density",
        enabled=True,
        gating_enabled=True,
        current=0.0,
        baseline=0.0,
        detail="graph axis should increase",
    )

    assert payload["status"] == "fail"
    assert issue is not None
    assert issue["metric"] == "shift_graph_edge_density_directionality"
    assert issue["degradation_pct"] is None


def test_run_benchmark_suite_lineage_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    spec = ProfileRunSpec(key="cpu_test", config=cfg, device="cpu")

    monkeypatch.setattr(
        "cauchy_generator.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "sample_datasets": 2,
            "lineage_metadata_coverage_rate": 1.0,
            "runtime_baseline_datasets_per_minute": 100.0,
            "runtime_with_lineage_datasets_per_minute": 70.0,
            "runtime_degradation_pct": 30.0,
            "runtime_warn_threshold_pct": 10.0,
            "runtime_fail_threshold_pct": 20.0,
            "issues": [
                {
                    "metric": "lineage_export_runtime_degradation_pct",
                    "severity": "fail",
                    "current": 70.0,
                    "baseline": 100.0,
                    "degradation_pct": 30.0,
                    "detail": "lineage overhead exceeded threshold",
                }
            ],
            "status": "fail",
        },
    )

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

    result = summary["profile_results"][0]
    guardrails = result["lineage_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "lineage_export_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )


def test_write_suite_markdown_profile_table_includes_shift_and_noise_columns(
    tmp_path: Path,
) -> None:
    summary = {
        "suite": "smoke",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "profile_results": [
            {
                "profile_key": "shift_smoke",
                "device": "cpu",
                "hardware_backend": "cpu",
                "datasets_per_minute": 120.0,
                "elapsed_seconds": 1.0,
                "latency_p95_ms": 4.0,
                "peak_rss_mb": 10.0,
                "diagnostics_enabled": False,
                "missingness_guardrails": {"enabled": False},
                "lineage_guardrails": {"enabled": False},
                "shift_guardrails": {"enabled": True, "status": "pass"},
                "noise_guardrails": {"enabled": True, "status": "warn"},
            }
        ],
        "regression": {"status": "pass", "issues": []},
    }
    path = write_suite_markdown(summary, tmp_path / "summary.md")
    text = path.read_text(encoding="utf-8")
    assert "| Shift |" in text
    assert "| Noise |" in text
    assert "| shift_smoke |" in text


def test_collect_lineage_guardrails_uses_median_of_three_trials_with_runtime_gating_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "cauchy_generator.dag_lineage"},
                },
            )

    trial_values = iter([100.0, 90.0, 1000.0, 100.0, 100.0, 90.0])

    def _stub_measure(
        _bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        assert not isinstance(_bundles, list)
        assert num_bundles > 0
        return float(next(trial_values))

    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter
    )
    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="smoke",
        num_datasets=2,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is True
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["runtime_gating_min_sample_datasets"] == 5
    assert guardrails["runtime_gating_suppressed_reason"] == "insufficient_sample_size"
    assert guardrails["runtime_trials"] == 3
    assert guardrails["runtime_baseline_trials_dpm"] == [100.0, 1000.0, 100.0]
    assert guardrails["runtime_with_lineage_trials_dpm"] == [90.0, 100.0, 90.0]
    assert guardrails["runtime_baseline_datasets_per_minute"] == pytest.approx(100.0)
    assert guardrails["runtime_with_lineage_datasets_per_minute"] == pytest.approx(90.0)
    assert guardrails["runtime_degradation_pct"] == pytest.approx(10.0)
    assert guardrails["status"] == "pass"
    assert not any(
        issue["metric"] == "lineage_export_runtime_degradation_pct"
        for issue in guardrails["issues"]
    )


def test_collect_lineage_guardrails_emits_runtime_issue_when_sample_is_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    cfg.benchmark.latency_num_samples = 6

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "cauchy_generator.dag_lineage"},
                },
            )

    trial_values = iter([100.0, 70.0, 100.0, 70.0, 100.0, 70.0])

    def _stub_measure(
        _bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        assert not isinstance(_bundles, list)
        assert num_bundles > 0
        return float(next(trial_values))

    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter
    )
    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="smoke",
        num_datasets=6,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is True
    assert guardrails["sample_datasets"] == 5
    assert guardrails["runtime_gating_enabled"] is True
    assert guardrails["runtime_gating_suppressed_reason"] is None
    assert guardrails["runtime_degradation_pct"] == pytest.approx(30.0)
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "lineage_export_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )


def test_collect_lineage_guardrails_reports_unavailable_for_non_runtime_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "cauchy_generator.dag_lineage"},
                },
            )

    def _stub_trials(
        *,
        baseline_stage_dir: Path,
        current_stage_dir: Path,
        num_bundles: int,
        config: GeneratorConfig,
        trials: int,
    ) -> tuple[list[float], list[float]]:
        _ = baseline_stage_dir
        _ = current_stage_dir
        _ = num_bundles
        _ = config
        _ = trials
        raise ValueError("codec unavailable")

    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter
    )
    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails._measure_lineage_persistence_trials",
        _stub_trials,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="smoke",
        num_datasets=2,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is False
    assert guardrails["reason"] == "unavailable"
    assert "codec unavailable" in guardrails["detail"]


def test_measure_lineage_persistence_trials_replays_staged_bundles_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    baseline_stage_dir = tmp_path / "baseline"
    current_stage_dir = tmp_path / "current"
    baseline_stage_dir.mkdir(parents=True, exist_ok=True)
    current_stage_dir.mkdir(parents=True, exist_ok=True)

    bundle = DatasetBundle(
        X_train=np.zeros((3, 4), dtype=np.float32),
        y_train=np.zeros(3, dtype=np.int64),
        X_test=np.zeros((1, 4), dtype=np.float32),
        y_test=np.zeros(1, dtype=np.int64),
        feature_types=["num", "num", "num", "num"],
        metadata={"seed": 0, "attempt_used": 0, "lineage": {"schema_name": "x"}},
    )
    for idx in range(2):
        file_name = f"bundle_{idx:06d}.pkl"
        guardrails_mod._stage_bundle(current_stage_dir / file_name, bundle, strip_lineage=False)
        guardrails_mod._stage_bundle(baseline_stage_dir / file_name, bundle, strip_lineage=True)

    def _unexpected_generate(*_args, **_kwargs):
        raise AssertionError("generate_batch_iter must not run during timed persistence trials")

    call_counts: list[int] = []

    def _stub_measure(
        bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        seen = sum(1 for _ in bundles)
        call_counts.append(seen)
        assert num_bundles == 2
        assert seen == 2
        return 100.0 if len(call_counts) % 2 == 1 else 90.0

    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails.generate_batch_iter", _unexpected_generate
    )
    monkeypatch.setattr(
        "cauchy_generator.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    baseline_trials, current_trials = guardrails_mod._measure_lineage_persistence_trials(
        baseline_stage_dir=baseline_stage_dir,
        current_stage_dir=current_stage_dir,
        num_bundles=2,
        config=cfg,
        trials=3,
    )

    assert baseline_trials == [100.0, 100.0, 100.0]
    assert current_trials == [90.0, 90.0, 90.0]
    assert call_counts == [2, 2, 2, 2, 2, 2]


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
