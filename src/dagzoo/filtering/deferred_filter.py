"""Deferred CPU filtering over persisted shard outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from dagzoo.config import FilterConfig, GeneratorConfig
from dagzoo.filtering.extra_trees_filter import apply_extra_trees_filter
from dagzoo.math_utils import sanitize_json as _sanitize_json
from dagzoo.rng import SEED32_MAX, SEED32_MIN

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pc = None
    pq = None

import numpy as np
import torch


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
class _ShardState:
    """Loaded shard artifacts required for deferred filter replay."""

    shard_dir: Path
    metadata_records: list[dict[str, Any]]
    train_table: Any
    test_table: Any
    train_rows_by_dataset: dict[int, tuple[np.ndarray, np.ndarray]]
    test_rows_by_dataset: dict[int, tuple[np.ndarray, np.ndarray]]


def _require_pyarrow() -> None:
    if pa is None or pc is None or pq is None:
        raise RuntimeError(
            "pyarrow is required for deferred filtering. Install project dependencies with uv."
        )


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


def _load_metadata_records(path: Path) -> list[dict[str, Any]]:
    """Read metadata.ndjson records for one shard."""

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid metadata record in {path}: expected object, got {type(payload)}"
            )
        records.append(payload)
    return records


def _rows_by_dataset(table: Any, *, split_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct per-dataset split arrays from packed parquet rows."""

    dataset_indices = table.column("dataset_index").to_pylist()
    row_indices = table.column("row_index").to_pylist()
    x_rows = table.column("x").to_pylist()
    y_rows = table.column("y").to_pylist()

    grouped: dict[int, list[tuple[int, Any, Any]]] = defaultdict(list)
    for dataset_index, row_index, x_row, y_row in zip(
        dataset_indices,
        row_indices,
        x_rows,
        y_rows,
        strict=True,
    ):
        grouped[int(dataset_index)].append((int(row_index), x_row, y_row))

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dataset_index, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        x_np = np.asarray([item[1] for item in rows], dtype=np.float32)
        y_np = np.asarray([item[2] for item in rows])
        if x_np.ndim != 2:
            raise ValueError(
                "Invalid packed feature shape while replaying deferred filter: "
                f"split={split_path} dataset_index={dataset_index} shape={x_np.shape}"
            )
        if y_np.ndim != 1:
            y_np = np.asarray(y_np).reshape(-1)
        result[dataset_index] = (x_np, y_np)

    return result


def _load_shard_state(shard_dir: Path) -> _ShardState:
    """Load one shard's parquet + metadata artifacts for deferred replay."""

    _require_pyarrow()

    metadata_path = shard_dir / "metadata.ndjson"
    train_path = shard_dir / "train.parquet"
    test_path = shard_dir / "test.parquet"

    if not metadata_path.exists() or not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Shard directory is missing required artifacts "
            "(metadata.ndjson/train.parquet/test.parquet): "
            f"{shard_dir}"
        )

    metadata_records = _load_metadata_records(metadata_path)
    train_table = pq.read_table(train_path, columns=["dataset_index", "row_index", "x", "y"])
    test_table = pq.read_table(test_path, columns=["dataset_index", "row_index", "x", "y"])

    return _ShardState(
        shard_dir=shard_dir,
        metadata_records=metadata_records,
        train_table=train_table,
        test_table=test_table,
        train_rows_by_dataset=_rows_by_dataset(train_table, split_path=train_path),
        test_rows_by_dataset=_rows_by_dataset(test_table, split_path=test_path),
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


def _resolve_task_and_filter_config(
    *,
    metadata_payload: Mapping[str, Any],
    fallback_config: GeneratorConfig | None,
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
        if fallback_config is None:
            raise ValueError(
                "Missing dataset task in shard metadata. Provide --config so deferred filter can "
                "resolve dataset.task."
            )
        task = str(fallback_config.dataset.task)

    if embedded_filter is not None:
        filter_cfg = FilterConfig(**dict(embedded_filter))
    elif fallback_config is not None:
        filter_cfg = FilterConfig(**fallback_config.to_dict().get("filter", {}))
    else:
        raise ValueError(
            "Missing filter config in shard metadata. Provide --config so deferred filter can "
            "resolve filter parameters."
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

    x_t = torch.from_numpy(x_all)
    y_t = torch.from_numpy(y_all)

    start = time.perf_counter()
    accepted, details = apply_extra_trees_filter(
        x_t,
        y_t,
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


def _write_metadata_ndjson(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write metadata records to NDJSON with JSON-safe sanitization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _sanitize_json(record),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")


def _filter_table_by_dataset_index(table: Any, dataset_indices: set[int]) -> Any:
    """Return table rows whose dataset_index is in ``dataset_indices``."""

    if not dataset_indices:
        return table.slice(0, 0)

    value_set = pa.array(sorted(dataset_indices), type=pa.int64())
    mask = pc.is_in(table.column("dataset_index"), value_set=value_set)
    return table.filter(mask)


def _write_curated_shard(
    *,
    shard_state: _ShardState,
    curated_out_dir: Path,
    accepted_dataset_indices: set[int],
    accepted_records: list[dict[str, Any]],
) -> int:
    """Write accepted-only shard artifacts for one source shard."""

    if not accepted_dataset_indices:
        return 0

    shard_out_dir = curated_out_dir / shard_state.shard_dir.name
    shard_out_dir.mkdir(parents=True, exist_ok=True)

    train_table = _filter_table_by_dataset_index(shard_state.train_table, accepted_dataset_indices)
    test_table = _filter_table_by_dataset_index(shard_state.test_table, accepted_dataset_indices)

    pq.write_table(train_table, shard_out_dir / "train.parquet", compression="zstd")
    pq.write_table(test_table, shard_out_dir / "test.parquet", compression="zstd")
    _write_metadata_ndjson(shard_out_dir / "metadata.ndjson", accepted_records)

    source_lineage_dir = shard_state.shard_dir / "lineage"
    if source_lineage_dir.exists():
        shutil.copytree(source_lineage_dir, shard_out_dir / "lineage", dirs_exist_ok=True)

    return len(accepted_records)


def run_deferred_filter(
    *,
    in_dir: str | Path,
    out_dir: str | Path,
    config: GeneratorConfig | None = None,
    curated_out_dir: str | Path | None = None,
    n_jobs_override: int | None = None,
) -> DeferredFilterRunResult:
    """Replay ExtraTrees filter over persisted shard outputs."""

    _require_pyarrow()

    input_path = Path(in_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    curated_path: Path | None = None
    if curated_out_dir is not None:
        curated_path = Path(curated_out_dir)
        _ensure_curated_output_dir_safe(curated_path)

    shard_dirs = _discover_shard_dirs(input_path)

    manifest_records: list[dict[str, Any]] = []
    rejected_reason_counts: Counter[str] = Counter()

    accepted_total = 0
    rejected_total = 0
    total_elapsed_seconds = 0.0
    curated_accepted_total = 0

    for shard_dir in shard_dirs:
        shard_state = _load_shard_state(shard_dir)

        accepted_dataset_indices: set[int] = set()
        accepted_records: list[dict[str, Any]] = []

        for record in shard_state.metadata_records:
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

            metadata_payload = record.get("metadata")
            if not isinstance(metadata_payload, Mapping):
                raise ValueError(
                    "Invalid metadata payload for deferred filtering: "
                    f"shard={shard_dir} dataset_index={dataset_index}"
                )

            train_split = shard_state.train_rows_by_dataset.get(dataset_index)
            test_split = shard_state.test_rows_by_dataset.get(dataset_index)
            if train_split is None or test_split is None:
                raise ValueError(
                    "Missing packed split rows for deferred filtering: "
                    f"shard={shard_dir} dataset_index={dataset_index}"
                )

            task, filter_cfg = _resolve_task_and_filter_config(
                metadata_payload=metadata_payload,
                fallback_config=config,
                n_jobs_override=n_jobs_override,
            )
            seed = _coerce_seed(metadata_payload.get("seed"), dataset_index=dataset_index)

            accepted, filter_details, elapsed_seconds = _filter_dataset(
                x_train=train_split[0],
                y_train=train_split[1],
                x_test=test_split[0],
                y_test=test_split[1],
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
            reason = str(reason_value) if isinstance(reason_value, str) and reason_value else None
            if not accepted:
                rejected_total += 1
                rejected_reason_counts[reason or "below_threshold"] += 1
            else:
                accepted_total += 1
                accepted_dataset_indices.add(dataset_index)
                accepted_records.append(normalized_record)

            manifest_records.append(
                {
                    "dataset_index": dataset_index,
                    "seed": seed,
                    "source_shard": shard_dir.name,
                    "accepted": bool(accepted),
                    "status": "accepted" if accepted else "rejected",
                    "reason": reason,
                    "elapsed_seconds": float(elapsed_seconds),
                    "filter": filter_metadata,
                }
            )

        if curated_path is not None:
            curated_accepted_total += _write_curated_shard(
                shard_state=shard_state,
                curated_out_dir=curated_path,
                accepted_dataset_indices=accepted_dataset_indices,
                accepted_records=accepted_records,
            )

    total_datasets = accepted_total + rejected_total
    datasets_per_minute = (
        (float(total_datasets) / float(total_elapsed_seconds)) * 60.0
        if total_elapsed_seconds > 0.0
        else 0.0
    )

    manifest_path = output_path / MANIFEST_FILENAME
    summary_path = output_path / SUMMARY_FILENAME

    _write_metadata_ndjson(manifest_path, manifest_records)

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
    summary_path.write_text(
        json.dumps(_sanitize_json(summary_payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

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
