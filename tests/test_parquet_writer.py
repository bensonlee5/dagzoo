import json

import numpy as np

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


def test_write_parquet_shards_backcompat_returns_paths(tmp_path, monkeypatch) -> None:
    def _stub_write_split(path, _x, _y, _compression):
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cauchy_generator.io.parquet_writer._write_split", _stub_write_split)
    paths = write_parquet_shards([_bundle(1), _bundle(2)], tmp_path, shard_size=1)
    assert len(paths) == 2
    assert paths[0].name == "dataset_000000"
    assert paths[1].name == "dataset_000001"
