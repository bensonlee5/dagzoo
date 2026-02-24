"""Parquet writer utilities."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence, cast

import numpy as np

from cauchy_generator.io.lineage_artifact import (
    pack_upper_triangle_adjacency,
    sha256_hex,
    upper_triangle_bit_length,
)
from cauchy_generator.io.lineage_schema import (
    LINEAGE_ADJACENCY_ENCODING,
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION_COMPACT,
    LINEAGE_SCHEMA_VERSION_DENSE,
    validate_lineage_payload,
)
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


def _shard_id_for_dataset(*, dataset_index: int, shard_size: int) -> int:
    """Return shard id for a dataset index under configured shard size."""

    return dataset_index // max(1, shard_size)


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


@dataclass(slots=True)
class _ShardLineageState:
    """Mutable lineage artifact state for one output shard."""

    blob_path: Path
    index_path: Path
    blob_file: BinaryIO | None = None
    byte_offset: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)


def _lineage_state_for_shard(
    *,
    shard_id: int,
    shard_dir: Path,
    lineage_states: dict[int, _ShardLineageState],
) -> _ShardLineageState:
    """Get or initialize lineage artifact state for one shard."""

    state = lineage_states.get(shard_id)
    if state is not None:
        return state

    lineage_dir = shard_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    state = _ShardLineageState(
        blob_path=lineage_dir / "adjacency.bitpack.bin",
        index_path=lineage_dir / "adjacency.index.json",
    )
    lineage_states[shard_id] = state
    return state


def _build_compact_lineage_payload(
    lineage: Mapping[str, Any],
    *,
    dataset_index: int,
    bit_offset: int,
    bit_length: int,
    edge_count: int,
    sha256: str,
    blob_path: str,
    index_path: str,
    n_nodes: int,
) -> dict[str, Any]:
    """Build compact lineage payload for persisted metadata."""

    assignments = cast(Mapping[str, Any], lineage["assignments"])
    feature_to_node = list(cast(list[int], assignments["feature_to_node"]))
    target_to_node = cast(int, assignments["target_to_node"])
    return {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION_COMPACT,
        "graph": {
            "n_nodes": int(n_nodes),
            "edge_count": int(edge_count),
            "adjacency_ref": {
                "encoding": LINEAGE_ADJACENCY_ENCODING,
                "blob_path": blob_path,
                "index_path": index_path,
                "dataset_index": int(dataset_index),
                "bit_offset": int(bit_offset),
                "bit_length": int(bit_length),
                "sha256": sha256,
            },
        },
        "assignments": {
            "feature_to_node": feature_to_node,
            "target_to_node": target_to_node,
        },
    }


def _ensure_shard_blob_open(state: _ShardLineageState) -> BinaryIO:
    """Return an append-ready shard blob handle, opening it once per shard."""

    blob_file = state.blob_file
    if blob_file is not None and not blob_file.closed:
        return blob_file

    opened = state.blob_path.open("ab")
    state.blob_file = opened
    return opened


def _close_shard_blob(state: _ShardLineageState) -> None:
    """Close a single shard blob handle if open."""

    blob_file = state.blob_file
    if blob_file is None:
        return
    try:
        blob_file.close()
    finally:
        state.blob_file = None


def _close_shard_lineage_files(lineage_states: Mapping[int, _ShardLineageState]) -> None:
    """Close open shard blob handles."""

    for state in lineage_states.values():
        _close_shard_blob(state)


def _persist_lineage_artifact_for_dataset(
    metadata: dict[str, Any],
    *,
    dataset_index: int,
    shard_id: int,
    shard_dir: Path,
    lineage_states: dict[int, _ShardLineageState],
) -> dict[str, Any]:
    """Convert dense lineage payload to compact shard artifact pointers for one dataset."""

    lineage_raw = metadata.get("lineage")
    if not isinstance(lineage_raw, Mapping):
        return metadata
    lineage = cast(Mapping[str, Any], lineage_raw)
    validate_lineage_payload(lineage)

    schema_version = cast(str, lineage["schema_version"])
    if schema_version != LINEAGE_SCHEMA_VERSION_DENSE:
        return metadata

    graph = cast(Mapping[str, Any], lineage["graph"])
    adjacency_raw = graph["adjacency"]
    n_nodes, edge_count, packed = pack_upper_triangle_adjacency(adjacency_raw)
    bit_length = upper_triangle_bit_length(n_nodes)

    state = _lineage_state_for_shard(
        shard_id=shard_id,
        shard_dir=shard_dir,
        lineage_states=lineage_states,
    )
    bit_offset = state.byte_offset * 8
    state.byte_offset += len(packed)

    blob_file = _ensure_shard_blob_open(state)
    blob_file.write(packed)
    checksum = sha256_hex(packed)

    blob_path = str(Path("..") / "lineage" / state.blob_path.name)
    index_path = str(Path("..") / "lineage" / state.index_path.name)
    compact_lineage = _build_compact_lineage_payload(
        lineage,
        dataset_index=dataset_index,
        bit_offset=bit_offset,
        bit_length=bit_length,
        edge_count=edge_count,
        sha256=checksum,
        blob_path=blob_path,
        index_path=index_path,
        n_nodes=n_nodes,
    )
    validate_lineage_payload(compact_lineage)
    metadata["lineage"] = compact_lineage

    state.records.append(
        {
            "dataset_index": int(dataset_index),
            "n_nodes": int(n_nodes),
            "edge_count": int(edge_count),
            "bit_offset": int(bit_offset),
            "bit_length": int(bit_length),
            "sha256": checksum,
        }
    )
    return metadata


def _write_shard_lineage_indexes(lineage_states: Mapping[int, _ShardLineageState]) -> None:
    """Write shard-level lineage index files after all datasets are persisted."""

    for state in lineage_states.values():
        if not state.records:
            continue
        payload = {
            "schema_name": LINEAGE_SCHEMA_NAME,
            "schema_version": LINEAGE_SCHEMA_VERSION_COMPACT,
            "encoding": LINEAGE_ADJACENCY_ENCODING,
            "records": state.records,
        }
        with state.index_path.open("w", encoding="utf-8") as f:
            json.dump(
                _sanitize_json(payload),
                f,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )


def _finalize_lineage_states(
    lineage_states: Mapping[int, _ShardLineageState],
    *,
    strict_index_write: bool,
) -> None:
    """Close handles and flush lineage indexes.

    When `strict_index_write` is False, index write failures are swallowed so the
    original generation exception remains the primary error.
    """

    _close_shard_lineage_files(lineage_states)
    if strict_index_write:
        _write_shard_lineage_indexes(lineage_states)
        return
    try:
        _write_shard_lineage_indexes(lineage_states)
    except Exception:
        return


def _write_bundle_dataset(
    bundle: DatasetBundle,
    *,
    out_dir: Path,
    dataset_index: int,
    shard_size: int,
    compression: str,
    lineage_states: dict[int, _ShardLineageState],
) -> Path:
    """Write a single dataset bundle under its shard/dataset directory."""

    shard_id = _shard_id_for_dataset(dataset_index=dataset_index, shard_size=shard_size)
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

    metadata = deepcopy(bundle.metadata)
    metadata = _persist_lineage_artifact_for_dataset(
        metadata,
        dataset_index=dataset_index,
        shard_id=shard_id,
        shard_dir=shard_dir,
        lineage_states=lineage_states,
    )
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json(metadata),
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

    lineage_states: dict[int, _ShardLineageState] = {}
    active_shard_id: int | None = None
    written = 0
    success = False
    try:
        for idx, bundle in enumerate(bundles):
            shard_id = _shard_id_for_dataset(dataset_index=idx, shard_size=shard_size)
            if active_shard_id is not None and shard_id != active_shard_id:
                previous_state = lineage_states.get(active_shard_id)
                if previous_state is not None:
                    _close_shard_blob(previous_state)
            active_shard_id = shard_id
            _write_bundle_dataset(
                bundle,
                out_dir=out,
                dataset_index=idx,
                shard_size=shard_size,
                compression=compression,
                lineage_states=lineage_states,
            )
            written = idx + 1
        success = True
        return written
    finally:
        _finalize_lineage_states(lineage_states, strict_index_write=success)


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
    lineage_states: dict[int, _ShardLineageState] = {}
    active_shard_id: int | None = None
    shard_paths: list[Path] = []
    success = False

    try:
        for idx, bundle in enumerate(bundles):
            shard_id = _shard_id_for_dataset(dataset_index=idx, shard_size=shard_size)
            if active_shard_id is not None and shard_id != active_shard_id:
                previous_state = lineage_states.get(active_shard_id)
                if previous_state is not None:
                    _close_shard_blob(previous_state)
            active_shard_id = shard_id
            dataset_dir = _write_bundle_dataset(
                bundle,
                out_dir=out,
                dataset_index=idx,
                shard_size=shard_size,
                compression=compression,
                lineage_states=lineage_states,
            )
            shard_paths.append(dataset_dir)
        success = True
        return shard_paths
    finally:
        _finalize_lineage_states(lineage_states, strict_index_write=success)
