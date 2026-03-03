from cauchy_generator.telemetry import (
    PerfTelemetry,
    activate_perf_telemetry,
    get_active_perf_telemetry,
)


def test_perf_telemetry_collects_counters_and_timers() -> None:
    telemetry = PerfTelemetry(enabled=True)
    telemetry.increment("x", 2.0)
    telemetry.increment("x", 3.0)
    with telemetry.timer("t"):
        pass
    payload = telemetry.snapshot()
    assert payload["enabled"] is True
    assert payload["counters"]["x"] == 5.0
    assert payload["timers_ms"]["t"] >= 0.0


def test_activate_perf_telemetry_context_is_scoped() -> None:
    telemetry = PerfTelemetry(enabled=True)
    assert get_active_perf_telemetry() is None
    with activate_perf_telemetry(telemetry):
        assert get_active_perf_telemetry() is telemetry
    assert get_active_perf_telemetry() is None
