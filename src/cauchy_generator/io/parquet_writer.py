"""Parquet writer utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cauchy_generator.types import DatasetBundle

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors or array-like inputs to NumPy arrays."""

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_to_columns(x: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a 2D feature matrix into Parquet column mapping."""

    return {f"f_{i:04d}": x[:, i] for i in range(x.shape[1])}


def _write_split(path: Path, x: np.ndarray, y: np.ndarray, compression: str) -> None:
    """Write one train/test split to a Parquet file."""

    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for Parquet output. Install project dependencies with uv.")
    data = _array_to_columns(x)
    data["y"] = y
    table = pa.table(data)
    pq.write_table(table, path, compression=compression)


def write_parquet_shards(
    bundles: Sequence[DatasetBundle],
    out_dir: str | Path,
    *,
    shard_size: int = 128,
    compression: str = "zstd",
) -> list[Path]:
    """Write dataset bundles as partitioned Parquet shards with metadata."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []

    for idx, bundle in enumerate(bundles):
        shard_id = idx // max(1, shard_size)
        shard_dir = out / f"shard_{shard_id:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = shard_dir / f"dataset_{idx:06d}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        x_train = _to_numpy(bundle.X_train)
        y_train = _to_numpy(bundle.y_train)
        x_test = _to_numpy(bundle.X_test)
        y_test = _to_numpy(bundle.y_test)

        _write_split(dataset_dir / "train.parquet", x_train, y_train, compression)
        _write_split(dataset_dir / "test.parquet", x_test, y_test, compression)

        with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(bundle.metadata, f, indent=2, sort_keys=True)
        shard_paths.append(dataset_dir)

    return shard_paths
