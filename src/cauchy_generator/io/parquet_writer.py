"""Parquet writer utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from cauchy_generator.math_utils import sanitize_json as _sanitize_json, to_numpy as _to_numpy
from cauchy_generator.types import DatasetBundle

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None


def _array_to_columns(x: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a 2D feature matrix into Parquet column mapping."""

    return {f"f_{i:04d}": x[:, i] for i in range(x.shape[1])}


def _write_split(path: Path, x: np.ndarray, y: np.ndarray, compression: str) -> None:
    """Write one train/test split to a Parquet file."""

    if pa is None or pq is None:
        raise RuntimeError(
            "pyarrow is required for Parquet output. Install project dependencies with uv."
        )
    data = _array_to_columns(x)
    data["y"] = y
    table = pa.table(data)
    pq.write_table(table, path, compression=compression)


def _ensure_output_dir_safe(out_dir: Path) -> None:
    """Fail fast when output directory already contains prior shard outputs."""

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return
    stale = next(out_dir.glob("shard_*"), None)
    if stale is not None:
        raise RuntimeError(
            f"Output directory already contains shard data: {out_dir}. "
            "Choose a new --out directory or remove existing shard_* folders first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_bundle_dataset(
    bundle: DatasetBundle,
    *,
    out_dir: Path,
    dataset_index: int,
    shard_size: int,
    compression: str,
) -> Path:
    """Write a single dataset bundle under its shard/dataset directory."""

    shard_id = dataset_index // max(1, shard_size)
    shard_dir = out_dir / f"shard_{shard_id:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = shard_dir / f"dataset_{dataset_index:06d}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    x_train = _to_numpy(bundle.X_train)
    y_train = _to_numpy(bundle.y_train)
    x_test = _to_numpy(bundle.X_test)
    y_test = _to_numpy(bundle.y_test)

    _write_split(dataset_dir / "train.parquet", x_train, y_train, compression)
    _write_split(dataset_dir / "test.parquet", x_test, y_test, compression)

    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json(bundle.metadata),
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return dataset_dir


def write_parquet_shards_stream(
    bundles: Iterable[DatasetBundle],
    out_dir: str | Path,
    *,
    shard_size: int = 128,
    compression: str = "zstd",
) -> int:
    """Write bundles from an iterable stream, returning number of datasets written."""

    out = Path(out_dir)
    _ensure_output_dir_safe(out)

    written = 0
    for idx, bundle in enumerate(bundles):
        _write_bundle_dataset(
            bundle,
            out_dir=out,
            dataset_index=idx,
            shard_size=shard_size,
            compression=compression,
        )
        written = idx + 1
    return written


def write_parquet_shards(
    bundles: Sequence[DatasetBundle],
    out_dir: str | Path,
    *,
    shard_size: int = 128,
    compression: str = "zstd",
) -> list[Path]:
    """Write dataset bundles as partitioned Parquet shards with metadata."""

    out = Path(out_dir)
    _ensure_output_dir_safe(out)
    shard_paths: list[Path] = []

    for idx, bundle in enumerate(bundles):
        dataset_dir = _write_bundle_dataset(
            bundle,
            out_dir=out,
            dataset_index=idx,
            shard_size=shard_size,
            compression=compression,
        )
        shard_paths.append(dataset_dir)

    return shard_paths
