import numpy as np
import pytest

from dagzoo.bench.collectors import _ThroughputPressureCollector
from dagzoo.bench.stage_metrics import (
    StageSampleCollector,
    measure_filter_datasets_per_minute,
    measure_filter_stage_metrics,
    measure_write_datasets_per_minute,
)
from dagzoo.config import GeneratorConfig
from dagzoo.types import DatasetBundle


def _bundle(
    *,
    metadata: dict[str, object],
    runtime_metrics: dict[str, object] | None = None,
) -> DatasetBundle:
    return DatasetBundle(
        X_train=np.zeros((8, 4), dtype=np.float32),
        y_train=np.zeros(8, dtype=np.int64),
        X_test=np.zeros((4, 4), dtype=np.float32),
        y_test=np.zeros(4, dtype=np.int64),
        feature_types=["num", "num", "num", "num"],
        metadata=metadata,
        runtime_metrics={} if runtime_metrics is None else runtime_metrics,
    )


def test_throughput_pressure_collector_tracks_attempt_and_filter_rejections() -> None:
    collector = _ThroughputPressureCollector()
    collector.update(
        _bundle(
            metadata={
                "generation_attempts": {
                    "total_attempts": 1,
                    "filter_attempts": 1,
                    "filter_rejections": 0,
                }
            }
        )
    )
    collector.update(
        _bundle(
            metadata={
                "generation_attempts": {
                    "total_attempts": 3,
                    "filter_attempts": 3,
                    "filter_rejections": 2,
                }
            }
        )
    )

    summary = collector.build_summary()
    assert summary["datasets_seen"] == 2
    assert summary["attempts_total"] == 4
    assert summary["attempts_per_dataset_mean"] == 2.0
    assert summary["retry_dataset_count"] == 1
    assert summary["retry_dataset_rate"] == 0.5
    assert summary["filter_attempts_total"] == 4
    assert summary["filter_rejections_total"] == 2
    assert summary["filter_rejection_rate_attempt_level"] == 0.5
    assert summary["filter_retry_dataset_count"] == 1
    assert summary["filter_retry_dataset_rate"] == 0.5


def test_throughput_pressure_collector_falls_back_to_legacy_metadata() -> None:
    collector = _ThroughputPressureCollector()
    collector.update(
        _bundle(
            metadata={
                "attempt_used": 2,
                "filter": {"enabled": True, "accepted": True},
            }
        )
    )
    collector.update(_bundle(metadata={"attempt_used": 0, "filter": {"enabled": False}}))

    summary = collector.build_summary()
    assert summary["datasets_seen"] == 2
    assert summary["attempts_total"] == 4
    assert summary["retry_dataset_count"] == 1
    assert summary["filter_attempts_total"] == 1
    assert summary["filter_rejections_total"] == 0
    assert summary["filter_rejection_rate_attempt_level"] == 0.0
    assert summary["filter_retry_dataset_count"] == 0
    assert summary["filter_retry_dataset_rate"] == 0.0


def test_throughput_pressure_collector_filter_rates_are_none_when_filter_not_attempted() -> None:
    collector = _ThroughputPressureCollector()
    collector.update(_bundle(metadata={"attempt_used": 0, "filter": {"enabled": False}}))
    summary = collector.build_summary()
    assert summary["filter_attempts_total"] == 0
    assert summary["filter_rejection_rate_attempt_level"] is None
    assert summary["filter_retry_dataset_rate"] is None


def test_stage_sample_collector_caps_samples() -> None:
    collector = StageSampleCollector(max_samples=2)
    collector.update(_bundle(metadata={"seed": 1}))
    collector.update(_bundle(metadata={"seed": 2}))
    collector.update(_bundle(metadata={"seed": 3}))
    assert len(collector.bundles) == 2


def test_stage_metric_helpers_return_zero_for_empty_samples() -> None:
    cfg = GeneratorConfig()
    assert measure_filter_datasets_per_minute([], config=cfg) == 0.0
    assert measure_write_datasets_per_minute([], config=cfg) == 0.0


def test_filter_stage_metric_replays_filter_and_reports_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return bool(int(_kwargs["seed"]) % 2), {"n_valid_oob": 128}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    bundles = [
        _bundle(metadata={"seed": 11, "dataset_seed": 21}),
        _bundle(metadata={"seed": 12, "dataset_seed": 22}),
    ]
    measurement = measure_filter_stage_metrics(bundles, config=cfg)
    assert measurement.filter_attempts_total == 2
    assert measurement.filter_accepted_datasets == 1
    assert measurement.filter_rejections_total == 1
    assert measurement.filter_rejected_datasets == 1
    assert measurement.datasets_per_minute > 0.0
    assert replay_seeds == [21, 22]
    assert measure_filter_datasets_per_minute(bundles, config=cfg) > 0.0


def test_filter_stage_metric_returns_zero_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = False
    calls: dict[str, int] = {"count": 0}

    def _stub_filter(*_args, **_kwargs):
        calls["count"] += 1
        return True, {}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    measurement = measure_filter_stage_metrics([_bundle(metadata={})], config=cfg)
    assert measurement.datasets_per_minute == 0.0
    assert measurement.filter_attempts_total == 0
    assert measurement.filter_accepted_datasets == 0
    assert measurement.filter_rejections_total == 0
    assert measurement.filter_rejected_datasets == 0
    assert calls["count"] == 0


def test_filter_stage_metric_uses_fallback_seed_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    cfg.seed = 77
    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return True, {}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    _ = measure_filter_stage_metrics(
        [
            _bundle(metadata={}),
            _bundle(metadata={"seed": 100}),
        ],
        config=cfg,
    )
    assert replay_seeds == [77, 100]


def test_filter_stage_metric_falls_back_to_legacy_seed_when_dataset_seed_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return True, {}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    _ = measure_filter_stage_metrics(
        [
            _bundle(metadata={"seed": 41}),
            _bundle(metadata={"seed": 42}),
        ],
        config=cfg,
    )
    assert replay_seeds == [41, 42]
