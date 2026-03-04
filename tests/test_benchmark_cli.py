import json
from pathlib import Path

import yaml

from dagzoo.cli import _print_preset_result_line, main


def test_benchmark_cli_writes_json(tmp_path) -> None:
    out = tmp_path / "summary.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["suite"] == "smoke"
    assert len(payload["preset_results"]) == 1
    lineage_guardrails = payload["preset_results"][0]["lineage_guardrails"]
    assert isinstance(lineage_guardrails, dict)
    assert isinstance(lineage_guardrails["enabled"], bool)
    if lineage_guardrails["enabled"]:
        assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


def test_benchmark_cli_writes_effective_config_trace_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "bench_results"
    code = main(
        [
            "benchmark",
            "--preset",
            "cpu",
            "--suite",
            "smoke",
            "--num-datasets",
            "1",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    trace_files = sorted((out_dir / "effective_configs").glob("*_trace.yaml"))
    assert trace_files
    trace_payload = yaml.safe_load(trace_files[0].read_text(encoding="utf-8"))
    assert isinstance(trace_payload, list)
    assert any(
        isinstance(item, dict) and item.get("source") == "benchmark.suite_smoke_caps"
        for item in trace_payload
    )


def test_benchmark_cli_fail_on_regression(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_payload = {
        "version": 1,
        "suite": "standard",
        "metrics": ["datasets_per_minute"],
        "presets": {
            "medium_cuda": {"datasets_per_minute": 1.0e9},
        },
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--baseline",
            str(baseline_path),
            "--warn-threshold-pct",
            "1",
            "--fail-threshold-pct",
            "2",
            "--fail-on-regression",
        ]
    )
    assert code == 1


def test_benchmark_cli_diagnostics_emits_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "bench_results"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--diagnostics",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    assert profile["diagnostics_enabled"] is True

    artifacts = profile["diagnostics_artifacts"]
    assert isinstance(artifacts, dict)
    assert isinstance(artifacts["json"], str)
    assert isinstance(artifacts["markdown"], str)
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()


def test_benchmark_cli_json_out_diagnostics_does_not_emit_default_summary_artifacts(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "default.yaml"
    json_out = tmp_path / "summary.json"
    diagnostics_out = tmp_path / "diag_out"
    monkeypatch.chdir(tmp_path)

    code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--diagnostics",
            "--json-out",
            str(json_out),
            "--diagnostics-out-dir",
            str(diagnostics_out),
        ]
    )
    assert code == 0
    assert json_out.exists()
    assert not (tmp_path / "benchmarks").exists()


def test_benchmark_cli_diagnostics_pointers_resolve_when_roots_differ(tmp_path) -> None:
    out_dir = tmp_path / "bench_results"
    diagnostics_out = tmp_path / "diag_out"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--diagnostics",
            "--out-dir",
            str(out_dir),
            "--diagnostics-out-dir",
            str(diagnostics_out),
        ]
    )
    assert code == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    artifacts = profile["diagnostics_artifacts"]
    assert isinstance(artifacts, dict)
    json_path = Path(artifacts["json"])
    md_path = Path(artifacts["markdown"])
    assert json_path.is_absolute()
    assert md_path.is_absolute()
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.resolve().is_relative_to(diagnostics_out.resolve())
    assert md_path.resolve().is_relative_to(diagnostics_out.resolve())


def test_benchmark_cli_missingness_guardrails_are_emitted(tmp_path) -> None:
    out = tmp_path / "summary_missingness.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_missingness_mcar.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    guardrails = profile["missingness_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    lineage_guardrails = profile["lineage_guardrails"]
    assert isinstance(lineage_guardrails, dict)
    assert isinstance(lineage_guardrails["enabled"], bool)
    if lineage_guardrails["enabled"]:
        assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


def test_benchmark_cli_shift_guardrails_are_emitted(tmp_path) -> None:
    out = tmp_path / "summary_shift.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_shift_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    guardrails = profile["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert "directional_checks" in guardrails
    assert "runtime_gating_enabled" in guardrails


def test_benchmark_cli_noise_guardrails_are_emitted(tmp_path) -> None:
    out = tmp_path / "summary_noise.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_noise_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    guardrails = profile["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert "metadata_coverage_rate" in guardrails
    assert "runtime_gating_enabled" in guardrails


def test_benchmark_cli_many_class_smoke_preset_emits_runtime_metrics(tmp_path) -> None:
    out = tmp_path / "summary_many_class_smoke.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_many_class_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    assert float(profile["datasets_per_minute"]) > 0.0
    assert float(profile["latency_p95_ms"]) >= 0.0
    regression = payload["regression"]
    assert regression["status"] in {"pass", "warn", "fail"}
    guardrails = profile["lineage_guardrails"]
    assert isinstance(guardrails, dict)
    assert isinstance(guardrails["enabled"], bool)


def test_print_preset_result_line_includes_shift_status(capsys) -> None:
    _print_preset_result_line(
        {
            "preset_key": "shift_smoke",
            "device": "cpu",
            "hardware_backend": "cpu",
            "datasets_per_minute": 123.0,
            "latency_p95_ms": 4.2,
            "shift_guardrails": {"enabled": True, "status": "pass"},
        }
    )
    output = capsys.readouterr().out
    assert "shift=pass" in output


def test_print_preset_result_line_includes_noise_status(capsys) -> None:
    _print_preset_result_line(
        {
            "preset_key": "noise_smoke",
            "device": "cpu",
            "hardware_backend": "cpu",
            "datasets_per_minute": 123.0,
            "latency_p95_ms": 4.2,
            "noise_guardrails": {"enabled": True, "status": "warn"},
        }
    )
    output = capsys.readouterr().out
    assert "noise=warn" in output
