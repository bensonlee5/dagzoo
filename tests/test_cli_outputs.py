from pathlib import Path

import pytest

from dagzoo.cli import main


def test_generate_cli_prints_effective_config_and_resolution_trace(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("dagzoo.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--out",
            str(tmp_path / "generate"),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
            "--print-effective-config",
            "--print-resolution-trace",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "Effective config:" in captured.out
    assert "Resolution trace:" in captured.out


def test_filter_cli_prints_curated_output_summary(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Result:
        manifest_path = Path("manifest.ndjson")
        summary_path = Path("summary.json")
        total_datasets = 2
        accepted_datasets = 1
        rejected_datasets = 1
        datasets_per_minute = 42.0
        curated_out_dir = Path("curated")
        curated_accepted_datasets = 1

    monkeypatch.setattr("dagzoo.cli.run_deferred_filter", lambda **kwargs: _Result())

    code = main(
        [
            "filter",
            "--in",
            "input_shards",
            "--out",
            str(tmp_path / "filter_out"),
            "--curated-out",
            str(tmp_path / "curated_out"),
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "Wrote curated accepted-only shards:" in captured.out


def test_request_cli_prints_request_execution_summary(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FilterResult:
        manifest_path = Path("filter_manifest.ndjson")
        summary_path = Path("filter_summary.json")
        total_datasets = 2
        accepted_datasets = 1
        rejected_datasets = 1
        datasets_per_minute = 42.0
        curated_out_dir = Path("curated")
        curated_accepted_datasets = 1

    class _Result:
        effective_config_path = Path("effective_config.yaml")
        effective_config_trace_path = Path("effective_config_trace.yaml")
        generated_dir = Path("generated")
        generated_datasets = 2
        filter_result = _FilterResult()

    monkeypatch.setattr("dagzoo.cli.run_request_execution", lambda **kwargs: _Result())

    request_path = tmp_path / "request.yaml"
    request_path.write_text(
        "version: v1\n"
        "task: classification\n"
        "dataset_count: 2\n"
        "rows: 1024\n"
        "profile: default\n"
        "output_root: requests/out\n",
        encoding="utf-8",
    )

    code = main(
        [
            "request",
            "--request",
            str(request_path),
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "Wrote effective config:" in captured.out
    assert "Wrote filter manifest:" in captured.out
    assert "Deferred filter summary:" in captured.out
    assert "Wrote curated accepted-only shards:" in captured.out


def test_benchmark_cli_prints_configs_and_writes_baseline(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = {
        "preset_results": [
            {
                "preset_key": "custom",
                "effective_config": {"runtime": {"device": "cpu"}},
                "effective_config_trace": [
                    {
                        "path": "runtime.device",
                        "source": "config",
                        "old_value": "auto",
                        "new_value": "cpu",
                    }
                ],
                "datasets_per_minute": 1.0,
                "latency_p95_ms": 1.0,
            }
        ],
        "regression": {"status": "pass", "issues": [], "hard_fail": False},
    }

    monkeypatch.setattr("dagzoo.cli.run_benchmark_suite", lambda *args, **kwargs: summary)
    monkeypatch.setattr("dagzoo.cli.write_suite_json", lambda _summary, path: Path(path))
    monkeypatch.setattr(
        "dagzoo.cli._print_preset_result_line",
        lambda _result: None,
    )
    monkeypatch.setattr("dagzoo.cli.build_baseline_payload", lambda _summary: {"schema_version": 1})
    monkeypatch.setattr("dagzoo.cli.write_baseline", lambda _payload, path: Path(path))

    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--json-out",
            str(tmp_path / "summary.json"),
            "--save-baseline",
            str(tmp_path / "baseline.json"),
            "--print-effective-config",
            "--print-resolution-trace",
            "--no-memory",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "Effective config [custom]:" in captured.out
    assert "Resolution trace [custom]:" in captured.out
    assert "Wrote benchmark baseline:" in captured.out


def test_diversity_audit_cli_writes_summary_and_status(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "summary": {"overall_status": "warn", "num_variants": 1},
    }

    monkeypatch.setattr("dagzoo.cli.run_effective_diversity_audit", lambda **kwargs: report)
    monkeypatch.setattr(
        "dagzoo.cli.write_effective_diversity_artifacts",
        lambda _report, out_dir: {"summary_json": Path(out_dir) / "summary.json"},
    )

    code = main(
        [
            "diversity-audit",
            "--baseline-config",
            "configs/default.yaml",
            "--variant-config",
            "configs/preset_shift_benchmark_smoke.yaml",
            "--out-dir",
            str(tmp_path / "diversity"),
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "Wrote diversity artifact [summary_json]:" in captured.out
    assert "Diversity audit status=warn variants=1" in captured.out


def test_diversity_audit_cli_fail_on_regression_treats_insufficient_metrics_as_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.cli.run_effective_diversity_audit",
        lambda **_kwargs: {
            "summary": {"overall_status": "insufficient_metrics", "num_variants": 1}
        },
    )
    monkeypatch.setattr(
        "dagzoo.cli.write_effective_diversity_artifacts",
        lambda _report, out_dir: {"summary_json": Path(out_dir) / "summary.json"},
    )

    code = main(
        [
            "diversity-audit",
            "--baseline-config",
            "configs/default.yaml",
            "--variant-config",
            "configs/preset_shift_benchmark_smoke.yaml",
            "--fail-on-regression",
            "--out-dir",
            str(tmp_path / "diversity"),
        ]
    )

    assert code == 1


def test_filter_calibration_cli_writes_summary_and_status(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.cli.run_filter_calibration",
        lambda **_kwargs: {
            "summary": {
                "overall_status": "warn",
                "best_overall_threshold_requested": 0.804,
                "best_overall_diversity_status": "warn",
                "best_passing_threshold_requested": 0.801,
                "num_candidates": 5,
            }
        },
    )
    monkeypatch.setattr(
        "dagzoo.cli.write_filter_calibration_artifacts",
        lambda _report, out_dir: {"summary_json": Path(out_dir) / "summary.json"},
    )

    code = main(
        [
            "filter-calibration",
            "--config",
            "configs/preset_filter_benchmark_smoke.yaml",
            "--out-dir",
            str(tmp_path / "filter_calibration"),
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "Wrote filter calibration artifact [summary_json]:" in captured.out
    assert "Filter calibration status=warn best_overall=0.804 best_passing=0.801 candidates=5" in (
        captured.out
    )


def test_filter_calibration_cli_fail_on_regression_uses_best_overall_status(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.cli.run_filter_calibration",
        lambda **_kwargs: {
            "summary": {
                "overall_status": "warn",
                "best_overall_threshold_requested": 0.8,
                "best_overall_diversity_status": "fail",
                "best_passing_threshold_requested": None,
                "num_candidates": 5,
            }
        },
    )
    monkeypatch.setattr(
        "dagzoo.cli.write_filter_calibration_artifacts",
        lambda _report, out_dir: {"summary_json": Path(out_dir) / "summary.json"},
    )

    code = main(
        [
            "filter-calibration",
            "--config",
            "configs/preset_filter_benchmark_smoke.yaml",
            "--fail-on-regression",
            "--out-dir",
            str(tmp_path / "filter_calibration"),
        ]
    )

    assert code == 1


def test_hardware_cli_prints_detected_hardware(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Hardware:
        backend = "cpu"
        device_name = "cpu"
        tier = "cpu"
        total_memory_gb = 8.0
        peak_flops = 1.23e12

    monkeypatch.setattr("dagzoo.cli.detect_hardware", lambda _device: _Hardware())

    code = main(["hardware", "--device", "cpu"])

    assert code == 0
    captured = capsys.readouterr()
    assert "backend=cpu" in captured.out
    assert "tier=cpu" in captured.out
