from dagzoo.bench.baseline import (
    build_baseline_payload,
    compare_summary_to_baseline,
)
from dagzoo.bench.metrics import degradation_percent, percent_change


def test_percent_change_and_degradation_direction() -> None:
    assert percent_change(90.0, 100.0) == -10.0
    assert degradation_percent("datasets_per_minute", 90.0, 100.0) == 10.0
    assert degradation_percent("elapsed_seconds", 110.0, 100.0) == 10.0


def test_compare_summary_warn_and_fail() -> None:
    baseline_summary = {
        "suite": "standard",
        "preset_results": [{"preset_key": "cpu", "datasets_per_minute": 100.0}],
    }
    baseline = build_baseline_payload(baseline_summary)

    warn_summary = {
        "suite": "standard",
        "preset_results": [{"preset_key": "cpu", "datasets_per_minute": 85.0}],
    }
    warn_result = compare_summary_to_baseline(
        warn_summary,
        baseline,
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )
    assert warn_result["status"] == "warn"
    assert warn_result["issues"][0]["severity"] == "warn"

    fail_summary = {
        "suite": "standard",
        "preset_results": [{"preset_key": "cpu", "datasets_per_minute": 70.0}],
    }
    fail_result = compare_summary_to_baseline(
        fail_summary,
        baseline,
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )
    assert fail_result["status"] == "fail"
    assert fail_result["issues"][0]["severity"] == "fail"
