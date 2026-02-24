import io
import json
from pathlib import Path

import numpy as np
import pytest

from cauchy_generator.io import resolve_lineage_path
from cauchy_generator.io.lineage_artifact import unpack_upper_triangle_adjacency
from cauchy_generator.io.lineage_schema import (
    LINEAGE_ADJACENCY_ENCODING,
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    LINEAGE_SCHEMA_VERSION_COMPACT,
)
from cauchy_generator.io.parquet_writer import write_parquet_shards, write_parquet_shards_stream
from cauchy_generator.io.parquet_writer import _sanitize_json
from cauchy_generator.types import DatasetBundle


def test_sanitize_json_replaces_non_finite_floats() -> None:
    payload = {
        "ok": 1.0,
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "nan": float("nan"),
        "nested": {"vals": [0.0, float("inf")]},
    }
    sanitized = _sanitize_json(payload)
    assert sanitized["ok"] == 1.0
    assert sanitized["inf"] is None
    assert sanitized["neg_inf"] is None
    assert sanitized["nan"] is None
    assert sanitized["nested"]["vals"][1] is None
    json.dumps(sanitized, allow_nan=False)


def _bundle(seed: int) -> DatasetBundle:
    x_train = np.full((2, 2), float(seed), dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    x_test = np.full((1, 2), float(seed), dtype=np.float32)
    y_test = np.array([1], dtype=np.int64)
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=["num", "num"],
        metadata={"seed": seed, "peak_flops": float("inf")},
    )


def _bundle_with_dense_lineage(seed: int) -> DatasetBundle:
    bundle = _bundle(seed)
    bundle.metadata["lineage"] = {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": 2,
            "adjacency": [[0, 1], [0, 0]],
        },
        "assignments": {
            "feature_to_node": [0, 1],
            "target_to_node": 1,
        },
    }
    return bundle


def test_io_exports_resolve_lineage_path(tmp_path) -> None:
    resolved = resolve_lineage_path(tmp_path, "../lineage/adjacency.bitpack.bin")
    assert resolved == tmp_path / "../lineage/adjacency.bitpack.bin"


def test_write_parquet_shards_stream_writes_iterable(tmp_path, monkeypatch) -> None:
    def _stub_write_split(path, _x, _y, _compression):
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _stub_write_split)
    bundles = (_bundle(i) for i in range(3))
    written = write_parquet_shards_stream(
        bundles,
        tmp_path,
        shard_size=2,
        compression="zstd",
    )

    assert written == 3
    assert (tmp_path / "shard_00000" / "dataset_000000" / "train.parquet").exists()
    assert (tmp_path / "shard_00000" / "dataset_000001" / "test.parquet").exists()
    assert (tmp_path / "shard_00001" / "dataset_000002" / "metadata.json").exists()
    metadata = json.loads(
        (tmp_path / "shard_00001" / "dataset_000002" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["peak_flops"] is None


def test_write_parquet_shards_returns_paths(tmp_path, monkeypatch) -> None:
    def _stub_write_split(path, _x, _y, _compression):
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _stub_write_split)
    paths = write_parquet_shards([_bundle(1), _bundle(2)], tmp_path, shard_size=1)
    assert len(paths) == 2
    assert paths[0].name == "dataset_000000"
    assert paths[1].name == "dataset_000001"


def test_write_parquet_shards_stream_rejects_stale_output(tmp_path) -> None:
    stale_dir = tmp_path / "shard_00000"
    stale_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="already contains shard data"):
        write_parquet_shards_stream([_bundle(1)], tmp_path, shard_size=1)


def test_write_parquet_shards_stream_writes_lineage_metadata(tmp_path, monkeypatch) -> None:
    def _stub_write_split(path, _x, _y, _compression):
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _stub_write_split)
    bundle = _bundle_with_dense_lineage(7)

    written = write_parquet_shards_stream([bundle], tmp_path, shard_size=1, compression="zstd")
    assert written == 1
    metadata = json.loads(
        (tmp_path / "shard_00000" / "dataset_000000" / "metadata.json").read_text(encoding="utf-8")
    )
    lineage = metadata["lineage"]
    assert lineage["schema_name"] == LINEAGE_SCHEMA_NAME
    assert lineage["schema_version"] == LINEAGE_SCHEMA_VERSION_COMPACT
    graph = lineage["graph"]
    assert graph["n_nodes"] == 2
    assert graph["edge_count"] == 1
    adjacency_ref = graph["adjacency_ref"]
    assert adjacency_ref["encoding"] == LINEAGE_ADJACENCY_ENCODING
    assert adjacency_ref["dataset_index"] == 0
    assert adjacency_ref["bit_offset"] == 0
    assert adjacency_ref["bit_length"] == 1
    assert isinstance(adjacency_ref["sha256"], str)
    assert len(adjacency_ref["sha256"]) == 64

    dataset_dir = tmp_path / "shard_00000" / "dataset_000000"
    blob_path = resolve_lineage_path(dataset_dir, adjacency_ref["blob_path"])
    index_path = resolve_lineage_path(dataset_dir, adjacency_ref["index_path"])
    assert blob_path.exists()
    assert index_path.exists()

    byte_offset = int(adjacency_ref["bit_offset"]) // 8
    byte_length = (int(adjacency_ref["bit_length"]) + 7) // 8
    with blob_path.open("rb") as f:
        f.seek(byte_offset)
        payload = f.read(byte_length)
    dense = unpack_upper_triangle_adjacency(
        payload,
        n_nodes=int(graph["n_nodes"]),
        bit_length=int(adjacency_ref["bit_length"]),
    )
    assert dense.tolist() == [[0, 1], [0, 0]]

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["schema_name"] == LINEAGE_SCHEMA_NAME
    assert index_payload["schema_version"] == LINEAGE_SCHEMA_VERSION_COMPACT
    assert index_payload["encoding"] == LINEAGE_ADJACENCY_ENCODING
    assert isinstance(index_payload["records"], list)
    assert index_payload["records"][0]["dataset_index"] == 0


def test_write_parquet_shards_stream_opens_lineage_blob_once_per_shard(
    tmp_path, monkeypatch
) -> None:
    def _stub_write_split(path, _x, _y, _compression):
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _stub_write_split)

    original_open = Path.open
    counters = {"open": 0, "close": 0}

    class _CountingBlob(io.BytesIO):
        def close(self) -> None:
            counters["close"] += 1
            super().close()

    open_blobs: dict[str, _CountingBlob] = {}

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self.name == "adjacency.bitpack.bin" and mode == "ab":
            counters["open"] += 1
            key = str(self)
            blob = open_blobs.get(key)
            if blob is None or blob.closed:
                blob = _CountingBlob()
                open_blobs[key] = blob
            blob.seek(0, io.SEEK_END)
            return blob
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    written = write_parquet_shards_stream(bundles, tmp_path, shard_size=8, compression="zstd")

    assert written == 2
    assert counters["open"] == 1
    assert counters["close"] == 1


def test_write_parquet_shards_stream_limits_open_lineage_blob_descriptors(
    tmp_path, monkeypatch
) -> None:
    def _stub_write_split(path, _x, _y, _compression):
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _stub_write_split)

    original_open = Path.open
    counters = {"max_open": 0}
    open_count = {"value": 0}

    class _CountingBlob(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            open_count["value"] += 1
            counters["max_open"] = max(counters["max_open"], open_count["value"])

        def close(self) -> None:
            if not self.closed:
                open_count["value"] -= 1
            super().close()

    open_blobs: dict[str, _CountingBlob] = {}

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self.name == "adjacency.bitpack.bin" and mode == "ab":
            key = str(self)
            blob = open_blobs.get(key)
            if blob is None or blob.closed:
                blob = _CountingBlob()
                open_blobs[key] = blob
            blob.seek(0, io.SEEK_END)
            return blob
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    bundles = [_bundle_with_dense_lineage(i) for i in range(12)]
    written = write_parquet_shards_stream(bundles, tmp_path, shard_size=1, compression="zstd")

    assert written == 12
    assert counters["max_open"] <= 1
    assert open_count["value"] == 0


def test_write_parquet_shards_stream_closes_lineage_blob_on_failure(tmp_path, monkeypatch) -> None:
    split_calls = {"count": 0}

    def _failing_write_split(path, _x, _y, _compression):
        split_calls["count"] += 1
        if split_calls["count"] >= 3:
            raise RuntimeError("forced split failure")
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _failing_write_split)

    original_open = Path.open
    counters = {"open": 0, "close": 0}

    class _CountingBlob(io.BytesIO):
        def close(self) -> None:
            counters["close"] += 1
            super().close()

    open_blobs: dict[str, _CountingBlob] = {}

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self.name == "adjacency.bitpack.bin" and mode == "ab":
            counters["open"] += 1
            key = str(self)
            blob = open_blobs.get(key)
            if blob is None or blob.closed:
                blob = _CountingBlob()
                open_blobs[key] = blob
            blob.seek(0, io.SEEK_END)
            return blob
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    with pytest.raises(RuntimeError, match="forced split failure"):
        write_parquet_shards_stream(bundles, tmp_path, shard_size=8, compression="zstd")

    assert counters["open"] == 1
    assert counters["close"] == 1


def test_write_parquet_shards_stream_writes_lineage_index_on_failure(tmp_path, monkeypatch) -> None:
    split_calls = {"count": 0}

    def _failing_write_split(path, _x, _y, _compression):
        split_calls["count"] += 1
        if split_calls["count"] >= 3:
            raise RuntimeError("forced split failure")
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _failing_write_split)

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    with pytest.raises(RuntimeError, match="forced split failure"):
        write_parquet_shards_stream(bundles, tmp_path, shard_size=8, compression="zstd")

    dataset_dir = tmp_path / "shard_00000" / "dataset_000000"
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    adjacency_ref = metadata["lineage"]["graph"]["adjacency_ref"]
    index_path = resolve_lineage_path(dataset_dir, adjacency_ref["index_path"])
    assert index_path.exists()

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = index_payload["records"]
    assert any(int(record["dataset_index"]) == 0 for record in records)


def test_write_parquet_shards_writes_lineage_index_on_failure(tmp_path, monkeypatch) -> None:
    split_calls = {"count": 0}

    def _failing_write_split(path, _x, _y, _compression):
        split_calls["count"] += 1
        if split_calls["count"] >= 3:
            raise RuntimeError("forced split failure")
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _failing_write_split)

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    with pytest.raises(RuntimeError, match="forced split failure"):
        write_parquet_shards(bundles, tmp_path, shard_size=8, compression="zstd")

    dataset_dir = tmp_path / "shard_00000" / "dataset_000000"
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    adjacency_ref = metadata["lineage"]["graph"]["adjacency_ref"]
    index_path = resolve_lineage_path(dataset_dir, adjacency_ref["index_path"])
    assert index_path.exists()

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = index_payload["records"]
    assert any(int(record["dataset_index"]) == 0 for record in records)
