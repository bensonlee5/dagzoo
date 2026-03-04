"""Helpers for benchmark stage-level throughput measurement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import tempfile
import time
from typing import Any

import torch

from dagzoo.bench.constants import SECONDS_PER_MINUTE, STAGE_FILTER_SEED_OFFSET
from dagzoo.config import GeneratorConfig
from dagzoo.filtering import apply_extra_trees_filter
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.rng import offset_seed32, validate_seed32
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
    """Measure filter-only throughput on sampled bundles."""

    num_bundles = len(bundles)
    if num_bundles <= 0:
        return 0.0

    start = time.perf_counter()
    for idx, bundle in enumerate(bundles):
        x, y = _bundle_to_filter_inputs(bundle, task=str(config.dataset.task))
        run_seed = _resolve_filter_seed(
            bundle,
            fallback_seed=offset_seed32(config.seed, STAGE_FILTER_SEED_OFFSET + idx),
        )
        _ = apply_extra_trees_filter(
            x,
            y,
            task=str(config.dataset.task),
            seed=run_seed,
            n_estimators=int(config.filter.n_estimators),
            max_depth=int(config.filter.max_depth),
            min_samples_leaf=int(config.filter.min_samples_leaf),
            max_leaf_nodes=(
                int(config.filter.max_leaf_nodes)
                if config.filter.max_leaf_nodes is not None
                else None
            ),
            max_features=config.filter.max_features,
            n_bootstrap=int(config.filter.n_bootstrap),
            threshold=float(config.filter.threshold),
            n_jobs=int(config.filter.n_jobs),
        )
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return 0.0
    return (float(num_bundles) / elapsed) * SECONDS_PER_MINUTE


def _bundle_to_filter_inputs(
    bundle: DatasetBundle,
    *,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full dataset tensors from train/test splits for filter replay."""

    x_train = torch.as_tensor(bundle.X_train, dtype=torch.float32, device="cpu")
    x_test = torch.as_tensor(bundle.X_test, dtype=torch.float32, device="cpu")
    x = torch.cat([x_train, x_test], dim=0)

    if task == "classification":
        y_train = torch.as_tensor(bundle.y_train, dtype=torch.int64, device="cpu").view(-1)
        y_test = torch.as_tensor(bundle.y_test, dtype=torch.int64, device="cpu").view(-1)
    else:
        y_train = torch.as_tensor(bundle.y_train, dtype=torch.float32, device="cpu")
        y_test = torch.as_tensor(bundle.y_test, dtype=torch.float32, device="cpu")
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
    y = torch.cat([y_train, y_test], dim=0)
    return x, y


def _resolve_filter_seed(bundle: DatasetBundle, *, fallback_seed: int) -> int:
    """Resolve deterministic filter seed from bundle metadata with fallback."""

    metadata: Any = bundle.metadata
    if isinstance(metadata, dict):
        candidate = metadata.get("seed")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return validate_seed32(candidate, field_name="seed")
    return validate_seed32(fallback_seed, field_name="seed")
