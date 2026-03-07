"""Helpers for benchmark stage-level throughput measurement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import tempfile
import time
from typing import Any

import numpy as np
import torch

from dagzoo.bench.constants import SECONDS_PER_MINUTE
from dagzoo.config import GeneratorConfig
from dagzoo.filtering import apply_extra_trees_filter
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.math_utils import to_numpy as _to_numpy
from dagzoo.rng import SEED32_MAX
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


@dataclass(slots=True)
class FilterStageMeasurement:
    """Deferred filter stage replay metrics over sampled bundles."""

    datasets_per_minute: float
    filter_attempts_total: int
    filter_rejections_total: int


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


def _coerce_bundle_seed(bundle: DatasetBundle, *, fallback_seed: int) -> int:
    """Resolve per-bundle seed for deterministic deferred filter replay."""

    metadata = bundle.metadata
    for key in ("dataset_seed", "seed"):
        raw_seed: Any = metadata.get(key)
        if isinstance(raw_seed, bool):
            raw_seed = None
        if isinstance(raw_seed, float):
            if raw_seed.is_integer():
                raw_seed = int(raw_seed)
            else:
                raw_seed = None
        if isinstance(raw_seed, int):
            return int(raw_seed % (SEED32_MAX + 1))
    return int(fallback_seed % (SEED32_MAX + 1))


def measure_filter_stage_metrics(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
) -> FilterStageMeasurement:
    """Replay deferred filter stage over sampled bundles and return throughput + outcomes."""

    if not bundles or not bool(config.filter.enabled):
        return FilterStageMeasurement(
            datasets_per_minute=0.0,
            filter_attempts_total=0,
            filter_rejections_total=0,
        )

    attempts_total = 0
    rejections_total = 0
    start = time.perf_counter()
    for idx, bundle in enumerate(bundles):
        x_train = _to_numpy(bundle.X_train).astype("float32", copy=False)
        x_test = _to_numpy(bundle.X_test).astype("float32", copy=False)
        y_train = _to_numpy(bundle.y_train)
        y_test = _to_numpy(bundle.y_test)
        x_all = x_train
        if x_test.size > 0:
            x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = y_train
        if y_test.size > 0:
            y_all = np.concatenate([y_train, y_test], axis=0)
        if str(config.dataset.task) == "classification":
            y_all = y_all.astype("int64", copy=False)
        else:
            y_all = y_all.astype("float32", copy=False)

        attempts_total += 1
        accepted, _details = apply_extra_trees_filter(
            torch.from_numpy(x_all),
            torch.from_numpy(y_all),
            task=str(config.dataset.task),
            seed=_coerce_bundle_seed(bundle, fallback_seed=config.seed + idx),
            n_estimators=int(config.filter.n_estimators),
            max_depth=int(config.filter.max_depth),
            min_samples_leaf=int(config.filter.min_samples_leaf),
            max_leaf_nodes=config.filter.max_leaf_nodes,
            max_features=config.filter.max_features,
            n_bootstrap=int(config.filter.n_bootstrap),
            threshold=float(config.filter.threshold),
            n_jobs=int(config.filter.n_jobs),
        )
        if not bool(accepted):
            rejections_total += 1

    elapsed = max(0.0, time.perf_counter() - start)
    dpm = ((float(attempts_total) / elapsed) * SECONDS_PER_MINUTE) if elapsed > 0.0 else 0.0
    return FilterStageMeasurement(
        datasets_per_minute=float(dpm),
        filter_attempts_total=int(attempts_total),
        filter_rejections_total=int(rejections_total),
    )


def measure_filter_datasets_per_minute(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
) -> float:
    """Measure deferred filter-stage throughput on sampled bundles."""

    measurement = measure_filter_stage_metrics(bundles, config=config)
    return float(measurement.datasets_per_minute)
