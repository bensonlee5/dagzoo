import numpy as np
import pytest

from dagzoo.bench.collectors import _ThroughputPressureCollector
from dagzoo.bench.stage_metrics import (
    StageSampleCollector,
    measure_filter_datasets_per_minute,
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


def test_filter_stage_metric_uses_recorded_elapsed_seconds() -> None:
    cfg = GeneratorConfig()
    bundles = [
        _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": 0.25}),
        _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": 0.50}),
    ]
    dpm = measure_filter_datasets_per_minute(
        bundles,
        config=cfg,
    )
    assert dpm == pytest.approx(160.0)


def test_filter_stage_metric_ignores_missing_or_invalid_elapsed_seconds() -> None:
    cfg = GeneratorConfig()
    dpm = measure_filter_datasets_per_minute(
        [
            _bundle(metadata={}),
            _bundle(metadata={}, runtime_metrics={}),
            _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": "bad"}),
            _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": -0.1}),
            _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": 0.0}),
        ],
        config=cfg,
    )
    assert dpm == 0.0


def test_filter_stage_metric_uses_only_valid_timed_bundles() -> None:
    cfg = GeneratorConfig()
    dpm = measure_filter_datasets_per_minute(
        [
            _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": 0.5}),
            _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": -1.0}),
            _bundle(metadata={}, runtime_metrics={"filter_elapsed_seconds": 0.5}),
        ],
        config=cfg,
    )
    assert dpm == pytest.approx(120.0)
