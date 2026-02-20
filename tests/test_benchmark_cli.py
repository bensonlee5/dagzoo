import json

from cauchy_generator.cli import main


def test_benchmark_cli_writes_json(tmp_path) -> None:
    out = tmp_path / "summary.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--profile",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--no-hardware-aware",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["suite"] == "smoke"
    assert len(payload["profile_results"]) == 1


def test_benchmark_cli_fail_on_regression(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_payload = {
        "version": 1,
        "suite": "standard",
        "metrics": ["datasets_per_minute"],
        "profiles": {
            "medium_cuda": {"datasets_per_minute": 1.0e9},
        },
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--profile",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--no-hardware-aware",
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
