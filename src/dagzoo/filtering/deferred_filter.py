"""Deferred CPU filtering over persisted shard outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, TextIO

from dagzoo.config import FilterConfig
from dagzoo.filtering.extra_trees_filter import _apply_extra_trees_filter_numpy
from dagzoo.io.parquet_writer import (
    _PackedShardState,
    _close_packed_shard_handles,
    _ensure_metadata_file_open,
    _require_pyarrow,
    _write_packed_split,
    pq,
)
from dagzoo.math_utils import sanitize_json as _sanitize_json
from dagzoo.rng import SEED32_MAX, SEED32_MIN

import numpy as np


MANIFEST_FILENAME = "filter_manifest.ndjson"
SUMMARY_FILENAME = "filter_summary.json"


@dataclass(slots=True)
class DeferredFilterRunResult:
    """Result payload for one deferred filter command execution."""

    manifest_path: Path
    summary_path: Path
    total_datasets: int
    accepted_datasets: int
    rejected_datasets: int
    elapsed_seconds: float
    datasets_per_minute: float
    curated_out_dir: Path | None = None
    curated_accepted_datasets: int = 0


@dataclass(slots=True)
class _PackedSplitDataset:
    """One dataset worth of packed parquet rows for a single split."""

    dataset_index: int
    x: np.ndarray
    y: np.ndarray


@dataclass(slots=True)
class _CuratedShardWriter:
    """Incremental writer state for one curated accepted-only shard."""

    shard_state: _PackedShardState
    final_shard_dir: Path

    @property
    def shard_dir(self) -> Path:
        return self.shard_state.shard_dir


def _discover_shard_dirs(input_path: Path) -> list[Path]:
    """Resolve shard directories from a root output dir or a direct shard path."""

    if input_path.is_dir() and (input_path / "metadata.ndjson").exists():
        return [input_path]

    shards = sorted(p for p in input_path.glob("shard_*") if p.is_dir())
    shards = [p for p in shards if (p / "metadata.ndjson").exists()]
    if shards:
        return shards

    raise FileNotFoundError(
        "No shard directories found under input path. "
        f"Expected either <dir>/metadata.ndjson or <dir>/shard_*/metadata.ndjson: {input_path}"
    )


def _ensure_filter_output_dir_safe(out_dir: Path) -> None:
    """Fail fast when deferred-filter output already contains prior artifacts."""

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return

    stale_paths = [out_dir / MANIFEST_FILENAME, out_dir / SUMMARY_FILENAME]
    stale = next((path for path in stale_paths if path.exists()), None)
    if stale is not None:
        raise RuntimeError(
            f"Deferred filter output directory already contains prior artifacts: {out_dir}. "
            f"Remove {stale.name} or choose a new --out directory."
        )

    out_dir.mkdir(parents=True, exist_ok=True)


def _iter_metadata_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield metadata.ndjson records for one shard."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    "Invalid metadata record in "
                    f"{path}:{line_number}: expected object, got {type(payload)}"
                )
            yield payload


def _build_packed_split_dataset(
    *,
    dataset_index: int,
    x_chunks: list[np.ndarray],
    y_chunks: list[np.ndarray],
    split_path: Path,
) -> _PackedSplitDataset:
    """Convert one accumulated packed split group into NumPy arrays."""

    if not x_chunks:
        x_np = np.empty((0, 0), dtype=np.float32)
    elif len(x_chunks) == 1:
        x_np = np.asarray(x_chunks[0], dtype=np.float32, copy=False)
    else:
        x_np = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    if not y_chunks:
        y_np = np.empty((0,), dtype=np.float32)
    elif len(y_chunks) == 1:
        y_np = np.asarray(y_chunks[0])
    else:
        y_np = np.concatenate(y_chunks, axis=0)
    if x_np.ndim != 2:
        raise ValueError(
            "Invalid packed feature shape while replaying deferred filter: "
            f"split={split_path} dataset_index={dataset_index} shape={x_np.shape}"
        )
    if y_np.ndim != 1:
        y_np = np.asarray(y_np).reshape(-1)
    return _PackedSplitDataset(dataset_index=dataset_index, x=x_np, y=y_np)


def _packed_feature_column_to_numpy_matrix(
    *,
    feature_column: Any,
    split_path: Path,
    dataset_index: int,
) -> np.ndarray:
    """Convert one packed list column batch into a dense 2D NumPy matrix."""

    offsets = np.asarray(feature_column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    n_rows = max(0, int(offsets.shape[0] - 1))
    if n_rows == 0:
        return np.empty((0, 0), dtype=np.float32)

    base_offset = int(offsets[0])
    normalized_offsets = offsets - base_offset
    row_widths = np.diff(normalized_offsets)
    if row_widths.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    expected_width = int(row_widths[0])
    if np.any(row_widths != expected_width):
        raise ValueError(
            "Invalid packed feature shape while replaying deferred filter: "
            f"split={split_path} dataset_index={dataset_index} shape=ragged"
        )

    total_values = int(normalized_offsets[-1])
    values = np.asarray(
        feature_column.values.slice(base_offset, total_values).to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    if expected_width == 0:
        return np.empty((n_rows, 0), dtype=np.float32)
    return values.reshape(n_rows, expected_width)


def _iter_packed_split_datasets(split_path: Path) -> Iterator[_PackedSplitDataset]:
    """Yield packed split rows one dataset at a time, validating shard ordering."""

    _require_pyarrow()

    parquet_file = pq.ParquetFile(split_path)
    required_columns = {"dataset_index", "row_index", "x", "y"}
    missing_columns = sorted(required_columns.difference(parquet_file.schema_arrow.names))
    if missing_columns:
        raise ValueError(
            f"Packed split is missing required columns in {split_path}: {missing_columns}"
        )

    current_dataset_index: int | None = None
    expected_row_index = 0
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    for batch in parquet_file.iter_batches(columns=["dataset_index", "row_index", "x", "y"]):
        dataset_indices = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        if dataset_indices.size == 0:
            continue
        row_indices = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
        feature_column = batch.column(2)
        batch_y_rows = np.asarray(batch.column(3).to_numpy(zero_copy_only=False))

        group_starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(dataset_indices) != 0) + 1)
        )
        group_ends = np.concatenate(
            (group_starts[1:], np.array([dataset_indices.size], dtype=np.int64))
        )

        for start_raw, end_raw in zip(group_starts, group_ends, strict=True):
            start = int(start_raw)
            end = int(end_raw)
            dataset_index = int(dataset_indices[start])
            group_row_indices = row_indices[start:end]

            if current_dataset_index is None:
                current_dataset_index = dataset_index
                expected_row_index = 0
            elif dataset_index < current_dataset_index:
                raise ValueError(
                    "Packed split rows must be grouped by monotonically increasing dataset_index: "
                    f"split={split_path} saw dataset_index={dataset_index} after "
                    f"{current_dataset_index}"
                )
            elif dataset_index != current_dataset_index:
                yield _build_packed_split_dataset(
                    dataset_index=current_dataset_index,
                    x_chunks=x_chunks,
                    y_chunks=y_chunks,
                    split_path=split_path,
                )
                current_dataset_index = dataset_index
                expected_row_index = 0
                x_chunks = []
                y_chunks = []

            expected_last_row_index = expected_row_index + int(group_row_indices.size) - 1
            group_is_contiguous = bool(
                group_row_indices.size == 0
                or (
                    int(group_row_indices[0]) == expected_row_index
                    and int(group_row_indices[-1]) == expected_last_row_index
                    and (
                        group_row_indices.size == 1 or bool(np.all(np.diff(group_row_indices) == 1))
                    )
                )
            )
            if not group_is_contiguous:
                actual_row_index = int(group_row_indices[0]) if group_row_indices.size > 0 else -1
                raise ValueError(
                    "Packed split rows must have contiguous row_index values starting at 0: "
                    f"split={split_path} dataset_index={dataset_index} "
                    f"expected_row_index={expected_row_index} got={actual_row_index}"
                )

            group_x_rows = _packed_feature_column_to_numpy_matrix(
                feature_column=feature_column.slice(start, end - start),
                split_path=split_path,
                dataset_index=dataset_index,
            )

            x_chunks.append(group_x_rows)
            y_chunks.append(batch_y_rows[start:end])
            expected_row_index += int(group_row_indices.size)

    if current_dataset_index is not None:
        yield _build_packed_split_dataset(
            dataset_index=current_dataset_index,
            x_chunks=x_chunks,
            y_chunks=y_chunks,
            split_path=split_path,
        )


def _coerce_seed(raw_seed: object, *, dataset_index: int) -> int:
    """Resolve a valid seed32 for filter replay."""

    if isinstance(raw_seed, bool):
        raw_seed = None

    if isinstance(raw_seed, float):
        if math.isfinite(raw_seed) and float(raw_seed).is_integer():
            raw_seed = int(raw_seed)
        else:
            raw_seed = None

    if isinstance(raw_seed, int):
        if SEED32_MIN <= raw_seed <= SEED32_MAX:
            return int(raw_seed)

    return int(dataset_index % (SEED32_MAX + 1))


def _resolve_filter_seed(metadata_payload: Mapping[str, Any], *, dataset_index: int) -> int:
    """Resolve filter replay seed from persisted metadata with child-seed preference."""

    return _coerce_seed(
        metadata_payload.get("dataset_seed", metadata_payload.get("seed")),
        dataset_index=dataset_index,
    )


def _resolve_task_and_filter_config(
    *,
    metadata_payload: Mapping[str, Any],
    n_jobs_override: int | None,
) -> tuple[str, FilterConfig]:
    """Resolve task + filter config for one dataset record."""

    task: str | None = None
    embedded_filter: Mapping[str, Any] | None = None

    config_payload = metadata_payload.get("config")
    if isinstance(config_payload, Mapping):
        dataset_payload = config_payload.get("dataset")
        if isinstance(dataset_payload, Mapping):
            dataset_task = dataset_payload.get("task")
            if isinstance(dataset_task, str) and dataset_task.strip():
                task = dataset_task.strip().lower()

        filter_payload = config_payload.get("filter")
        if isinstance(filter_payload, Mapping):
            embedded_filter = filter_payload

    if task not in {"classification", "regression"}:
        raise ValueError(
            "Deferred filter requires embedded metadata.config.dataset.task in shard metadata."
        )

    if embedded_filter is not None:
        filter_cfg = FilterConfig(**dict(embedded_filter))
    else:
        raise ValueError(
            "Deferred filter requires embedded metadata.config.filter in shard metadata."
        )

    filter_cfg.enabled = True
    if n_jobs_override is not None:
        filter_cfg.n_jobs = int(n_jobs_override)
        filter_cfg.__post_init__()

    return task, filter_cfg


def _build_filter_metadata(
    *,
    existing_filter: object,
    accepted: bool,
    filter_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build normalized filter metadata payload for deferred status."""

    payload: dict[str, Any] = dict(existing_filter) if isinstance(existing_filter, Mapping) else {}
    payload["mode"] = "deferred"
    payload["status"] = "accepted" if accepted else "rejected"
    payload["enabled"] = True
    payload["accepted"] = bool(accepted)
    payload.update(dict(filter_details))
    return payload


def _filter_dataset(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int,
    filter_cfg: FilterConfig,
) -> tuple[bool, dict[str, Any], float]:
    """Replay ExtraTrees filter on persisted train/test rows for one dataset."""

    x_all = np.concatenate([x_train, x_test], axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate([y_train, y_test], axis=0)
    y_dtype = np.int64 if task == "classification" else np.float32
    y_all = y_all.astype(y_dtype, copy=False)

    start = time.perf_counter()
    accepted, details = _apply_extra_trees_filter_numpy(
        x_all,
        y_all,
        task=task,
        seed=seed,
        n_estimators=filter_cfg.n_estimators,
        max_depth=filter_cfg.max_depth,
        min_samples_leaf=filter_cfg.min_samples_leaf,
        max_leaf_nodes=filter_cfg.max_leaf_nodes,
        max_features=filter_cfg.max_features,
        n_bootstrap=filter_cfg.n_bootstrap,
        threshold=filter_cfg.threshold,
        n_jobs=filter_cfg.n_jobs,
    )
    elapsed_seconds = max(0.0, time.perf_counter() - start)
    return bool(accepted), dict(details), float(elapsed_seconds)


def _ensure_curated_output_dir_safe(out_dir: Path) -> None:
    """Fail fast when curated output already contains shard artifacts."""

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return

    stale = next(out_dir.glob("shard_*"), None)
    if stale is not None:
        raise RuntimeError(
            f"Curated output directory already contains shard data: {out_dir}. "
            "Choose a new --curated-out directory or remove existing shard_* folders first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)


def _write_ndjson_record(handle: TextIO, record: Mapping[str, Any]) -> None:
    """Append one JSON-safe NDJSON record to an already-open handle."""

    handle.write(
        json.dumps(
            _sanitize_json(dict(record)),
            sort_keys=True,
            allow_nan=False,
        )
    )
    handle.write("\n")


def _staged_output_path(*, parent_dir: Path, final_name: str, staging_token: str) -> Path:
    """Return one hidden temp path used for deferred-filter staging."""

    return parent_dir / f".{final_name}.{staging_token}.tmp"


def _cleanup_path(path: Path | None) -> None:
    """Best-effort cleanup for one staged or promoted artifact path."""

    if path is None or not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def _promote_staged_path(*, staged_path: Path, final_path: Path) -> None:
    """Promote one staged file or directory into its final visible location."""

    if final_path.exists():
        raise RuntimeError(
            "Deferred filter promotion target already exists: "
            f"{final_path}. Remove the existing artifact and retry."
        )
    staged_path.replace(final_path)


def _create_curated_shard_writer(
    *,
    curated_out_dir: Path,
    shard_name: str,
    staging_token: str,
) -> _CuratedShardWriter:
    """Initialize incremental writer state for one curated shard."""

    shard_dir = _staged_output_path(
        parent_dir=curated_out_dir,
        final_name=shard_name,
        staging_token=staging_token,
    )
    shard_dir.mkdir(parents=True, exist_ok=False)
    final_shard_dir = curated_out_dir / shard_name
    return _CuratedShardWriter(
        shard_state=_PackedShardState(
            shard_dir=shard_dir,
            train_path=shard_dir / "train.parquet",
            test_path=shard_dir / "test.parquet",
            metadata_path=shard_dir / "metadata.ndjson",
        ),
        final_shard_dir=final_shard_dir,
    )


def _ensure_curated_metadata_file_open(state: _CuratedShardWriter) -> TextIO:
    """Return an append-ready metadata handle for a curated shard."""

    return _ensure_metadata_file_open(state.shard_state)


def _write_curated_split(
    *,
    state: _CuratedShardWriter,
    split: str,
    dataset_index: int,
    x: np.ndarray,
    y: np.ndarray,
    compression: str,
) -> None:
    """Append one accepted dataset split into a curated shard parquet file."""

    _write_packed_split(
        state=state.shard_state,
        split=split,
        dataset_index=dataset_index,
        x=x,
        y=y,
        compression=compression,
    )


def _write_curated_dataset(
    *,
    state: _CuratedShardWriter,
    dataset_index: int,
    train_split: _PackedSplitDataset,
    test_split: _PackedSplitDataset,
    record: Mapping[str, Any],
) -> None:
    """Append one accepted dataset to curated shard outputs."""

    _write_curated_split(
        state=state,
        split="train",
        dataset_index=dataset_index,
        x=train_split.x,
        y=train_split.y,
        compression="zstd",
    )
    _write_curated_split(
        state=state,
        split="test",
        dataset_index=dataset_index,
        x=test_split.x,
        y=test_split.y,
        compression="zstd",
    )
    metadata_file = _ensure_curated_metadata_file_open(state)
    _write_ndjson_record(metadata_file, record)


def _close_curated_shard_writer(state: _CuratedShardWriter | None) -> None:
    """Close open parquet and metadata handles for one curated shard."""

    if state is None:
        return
    _close_packed_shard_handles(state.shard_state)


def _copy_lineage_tree_safe(*, source_dir: Path, dest_dir: Path) -> None:
    """Copy lineage artifacts without following symlinks."""

    if source_dir.is_symlink():
        raise RuntimeError(f"Lineage directory must not be a symlink: {source_dir}")

    for source_path in sorted(source_dir.rglob("*")):
        rel_path = source_path.relative_to(source_dir)
        dest_path = dest_dir / rel_path
        if source_path.is_symlink():
            raise RuntimeError(f"Lineage artifact must not be a symlink: {source_path}")
        if source_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if source_path.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            continue
        raise RuntimeError(f"Unsupported lineage artifact entry: {source_path}")


def _consume_expected_split(
    split_iter: Iterator[_PackedSplitDataset],
    *,
    expected_dataset_index: int,
    split_path: Path,
) -> _PackedSplitDataset:
    """Consume the next packed split group and validate dataset alignment."""

    try:
        split_dataset = next(split_iter)
    except StopIteration as exc:
        raise ValueError(
            "Missing packed split rows for deferred filtering: "
            f"split={split_path} dataset_index={expected_dataset_index}"
        ) from exc

    if split_dataset.dataset_index != expected_dataset_index:
        raise ValueError(
            "Packed split coverage mismatch for deferred filtering: "
            f"split={split_path} expected dataset_index={expected_dataset_index} "
            f"got={split_dataset.dataset_index}"
        )
    return split_dataset


def _ensure_split_iter_exhausted(
    split_iter: Iterator[_PackedSplitDataset],
    *,
    split_path: Path,
) -> None:
    """Ensure a packed split iterator has no extra dataset groups beyond metadata."""

    try:
        extra_split = next(split_iter)
    except StopIteration:
        return
    raise ValueError(
        "Packed split contains extra dataset rows beyond metadata coverage: "
        f"split={split_path} dataset_index={extra_split.dataset_index}"
    )


def run_deferred_filter(
    *,
    in_dir: str | Path,
    out_dir: str | Path,
    curated_out_dir: str | Path | None = None,
    n_jobs_override: int | None = None,
) -> DeferredFilterRunResult:
    """Replay ExtraTrees filter over persisted shard outputs."""

    _require_pyarrow()

    input_path = Path(in_dir)
    output_path = Path(out_dir)
    _ensure_filter_output_dir_safe(output_path)

    curated_path: Path | None = None
    if curated_out_dir is not None:
        curated_path = Path(curated_out_dir)
        _ensure_curated_output_dir_safe(curated_path)

    shard_dirs = _discover_shard_dirs(input_path)

    rejected_reason_counts: Counter[str] = Counter()

    accepted_total = 0
    rejected_total = 0
    total_elapsed_seconds = 0.0
    curated_accepted_total = 0

    manifest_path = output_path / MANIFEST_FILENAME
    summary_path = output_path / SUMMARY_FILENAME
    staging_token = str(time.time_ns())
    staged_manifest_path = _staged_output_path(
        parent_dir=output_path,
        final_name=MANIFEST_FILENAME,
        staging_token=staging_token,
    )
    staged_summary_path = _staged_output_path(
        parent_dir=output_path,
        final_name=SUMMARY_FILENAME,
        staging_token=staging_token,
    )
    staged_curated_dirs: list[Path] = []
    promotable_curated_dirs: list[tuple[Path, Path]] = []
    promoted_final_paths: list[Path] = []

    try:
        with staged_manifest_path.open("w", encoding="utf-8") as manifest_file:
            for shard_dir in shard_dirs:
                metadata_path = shard_dir / "metadata.ndjson"
                train_path = shard_dir / "train.parquet"
                test_path = shard_dir / "test.parquet"
                if not metadata_path.exists() or not train_path.exists() or not test_path.exists():
                    raise FileNotFoundError(
                        "Shard directory is missing required artifacts "
                        "(metadata.ndjson/train.parquet/test.parquet): "
                        f"{shard_dir}"
                    )

                train_iter = _iter_packed_split_datasets(train_path)
                test_iter = _iter_packed_split_datasets(test_path)
                last_dataset_index = -1
                curated_writer: _CuratedShardWriter | None = None
                curated_written = 0

                try:
                    for record in _iter_metadata_records(metadata_path):
                        dataset_index_raw = record.get("dataset_index")
                        if dataset_index_raw is None or isinstance(dataset_index_raw, bool):
                            raise ValueError(
                                "Invalid dataset_index in metadata record: "
                                f"shard={shard_dir} dataset_index={dataset_index_raw!r}"
                            )
                        try:
                            dataset_index = int(dataset_index_raw)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                "Invalid dataset_index in metadata record: "
                                f"shard={shard_dir} dataset_index={dataset_index_raw!r}"
                            ) from exc

                        if dataset_index <= last_dataset_index:
                            raise ValueError(
                                "Metadata records must use strictly increasing dataset_index values: "
                                f"shard={shard_dir} dataset_index={dataset_index}"
                            )
                        last_dataset_index = dataset_index

                        metadata_payload = record.get("metadata")
                        if not isinstance(metadata_payload, Mapping):
                            raise ValueError(
                                "Invalid metadata payload for deferred filtering: "
                                f"shard={shard_dir} dataset_index={dataset_index}"
                            )

                        train_split = _consume_expected_split(
                            train_iter,
                            expected_dataset_index=dataset_index,
                            split_path=train_path,
                        )
                        test_split = _consume_expected_split(
                            test_iter,
                            expected_dataset_index=dataset_index,
                            split_path=test_path,
                        )

                        task, filter_cfg = _resolve_task_and_filter_config(
                            metadata_payload=metadata_payload,
                            n_jobs_override=n_jobs_override,
                        )
                        seed = _resolve_filter_seed(metadata_payload, dataset_index=dataset_index)

                        accepted, filter_details, elapsed_seconds = _filter_dataset(
                            x_train=train_split.x,
                            y_train=train_split.y,
                            x_test=test_split.x,
                            y_test=test_split.y,
                            task=task,
                            seed=seed,
                            filter_cfg=filter_cfg,
                        )
                        total_elapsed_seconds += elapsed_seconds

                        filter_metadata = _build_filter_metadata(
                            existing_filter=metadata_payload.get("filter"),
                            accepted=accepted,
                            filter_details=filter_details,
                        )

                        normalized_record = dict(record)
                        normalized_metadata = dict(metadata_payload)
                        normalized_metadata["filter"] = filter_metadata
                        normalized_record["metadata"] = normalized_metadata

                        reason_value = filter_details.get("reason")
                        reason = (
                            str(reason_value)
                            if isinstance(reason_value, str) and reason_value
                            else None
                        )
                        if not accepted:
                            rejected_total += 1
                            rejected_reason_counts[reason or "below_threshold"] += 1
                        else:
                            accepted_total += 1
                            if curated_path is not None:
                                if curated_writer is None:
                                    curated_writer = _create_curated_shard_writer(
                                        curated_out_dir=curated_path,
                                        shard_name=shard_dir.name,
                                        staging_token=staging_token,
                                    )
                                    staged_curated_dirs.append(curated_writer.shard_dir)
                                _write_curated_dataset(
                                    state=curated_writer,
                                    dataset_index=dataset_index,
                                    train_split=train_split,
                                    test_split=test_split,
                                    record=normalized_record,
                                )
                                curated_written += 1

                        _write_ndjson_record(
                            manifest_file,
                            {
                                "dataset_index": dataset_index,
                                "seed": seed,
                                "source_shard": shard_dir.name,
                                "accepted": bool(accepted),
                                "status": "accepted" if accepted else "rejected",
                                "reason": reason,
                                "elapsed_seconds": float(elapsed_seconds),
                                "filter": filter_metadata,
                            },
                        )

                    _ensure_split_iter_exhausted(train_iter, split_path=train_path)
                    _ensure_split_iter_exhausted(test_iter, split_path=test_path)
                finally:
                    _close_curated_shard_writer(curated_writer)

                if curated_writer is not None and curated_written > 0:
                    source_lineage_dir = shard_dir / "lineage"
                    if source_lineage_dir.exists():
                        _copy_lineage_tree_safe(
                            source_dir=source_lineage_dir,
                            dest_dir=curated_writer.shard_dir / "lineage",
                        )
                    promotable_curated_dirs.append(
                        (curated_writer.shard_dir, curated_writer.final_shard_dir)
                    )
                    curated_accepted_total += curated_written

        total_datasets = accepted_total + rejected_total
        datasets_per_minute = (
            (float(total_datasets) / float(total_elapsed_seconds)) * 60.0
            if total_elapsed_seconds > 0.0
            else 0.0
        )

        summary_payload: dict[str, Any] = {
            "input_dir": str(input_path.resolve()),
            "out_dir": str(output_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "total_datasets": int(total_datasets),
            "accepted_datasets": int(accepted_total),
            "rejected_datasets": int(rejected_total),
            "acceptance_rate": (
                float(accepted_total) / float(total_datasets) if total_datasets > 0 else None
            ),
            "rejected_reason_counts": {
                key: int(rejected_reason_counts[key]) for key in sorted(rejected_reason_counts)
            },
            "elapsed_seconds": float(total_elapsed_seconds),
            "datasets_per_minute": float(datasets_per_minute),
            "curated_out_dir": str(curated_path.resolve()) if curated_path is not None else None,
            "curated_accepted_datasets": int(curated_accepted_total),
        }
        staged_summary_path.write_text(
            json.dumps(_sanitize_json(summary_payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

        for staged_dir, final_dir in promotable_curated_dirs:
            _promote_staged_path(staged_path=staged_dir, final_path=final_dir)
            promoted_final_paths.append(final_dir)
        _promote_staged_path(staged_path=staged_manifest_path, final_path=manifest_path)
        promoted_final_paths.append(manifest_path)
        _promote_staged_path(staged_path=staged_summary_path, final_path=summary_path)
        promoted_final_paths.append(summary_path)
    except Exception:
        for path in reversed(promoted_final_paths):
            _cleanup_path(path)
        raise
    finally:
        _cleanup_path(staged_manifest_path)
        _cleanup_path(staged_summary_path)
        for staged_dir in staged_curated_dirs:
            _cleanup_path(staged_dir)

    return DeferredFilterRunResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        total_datasets=int(total_datasets),
        accepted_datasets=int(accepted_total),
        rejected_datasets=int(rejected_total),
        elapsed_seconds=float(total_elapsed_seconds),
        datasets_per_minute=float(datasets_per_minute),
        curated_out_dir=curated_path,
        curated_accepted_datasets=int(curated_accepted_total),
    )
