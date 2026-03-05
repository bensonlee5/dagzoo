"""Helpers for benchmark stage-level throughput measurement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import tempfile
import time

from dagzoo.bench.constants import SECONDS_PER_MINUTE
from dagzoo.config import GeneratorConfig
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.types import DatasetBundle


@dataclass(slots=True)
class StageSampleCollector:
    """Collect a bounded sample of emitted bundles for stage timing probes."""

    max_samples: int
    bundles: list[DatasetBundle] = field(default_factory=list)

    def update(self, bundle: DatasetBundle) -> None:
        """Store one emitted bundle if the sample has remaining capacity."""

        if len(self.bundles) >= max(0, int(self.max_samples)):
            return
        self.bundles.append(bundle)


def measure_write_datasets_per_minute(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
) -> float:
    """Measure parquet write throughput on sampled bundles."""

    num_bundles = len(bundles)
    if num_bundles <= 0:
        return 0.0

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dagzoo_stage_write_") as tmp_dir:
        _ = write_packed_parquet_shards_stream(
            bundles,
            out_dir=tmp_dir,
            shard_size=max(1, int(config.output.shard_size)),
            compression=str(config.output.compression),
        )
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return 0.0
    return (float(num_bundles) / elapsed) * SECONDS_PER_MINUTE


def measure_filter_datasets_per_minute(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
) -> float:
    """Measure filter-stage throughput from generation-time filter timings."""

    _ = config
    if not bundles:
        return 0.0

    total_seconds = 0.0
    timed_bundles = 0
    for bundle in bundles:
        runtime_metrics = bundle.runtime_metrics
        if not isinstance(runtime_metrics, dict):
            continue
        elapsed = runtime_metrics.get("filter_elapsed_seconds")
        if not isinstance(elapsed, (int, float)):
            continue
        elapsed_seconds = float(elapsed)
        if not elapsed_seconds > 0.0:
            continue
        total_seconds += elapsed_seconds
        timed_bundles += 1

    if timed_bundles <= 0 or total_seconds <= 0.0:
        return 0.0
    return (float(timed_bundles) / total_seconds) * SECONDS_PER_MINUTE
