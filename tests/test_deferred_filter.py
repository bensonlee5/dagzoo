import json

import numpy as np
import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.filtering.deferred_filter import run_deferred_filter
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.types import DatasetBundle


def _bundle_with_embedded_config(
    seed: int,
    *,
    dataset_seed: int | None = None,
    dataset_index: int | None = None,
) -> DatasetBundle:
    metadata = {
        "seed": seed,
        "filter": {"mode": "deferred", "status": "not_run"},
        "config": {
            "dataset": {"task": "classification"},
            "filter": {"enabled": True},
        },
    }
    if dataset_seed is not None:
        metadata["dataset_seed"] = int(dataset_seed)
    if dataset_index is not None:
        metadata["dataset_index"] = int(dataset_index)

    return DatasetBundle(
        X_train=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        y_train=np.array([0, 1, 0], dtype=np.int64),
        X_test=np.array([[1.5, 0.5], [0.5, 1.5]], dtype=np.float32),
        y_test=np.array([1, 0], dtype=np.int64),
        feature_types=["num", "num"],
        metadata=metadata,
    )


def _bundle_without_config(seed: int) -> DatasetBundle:
    bundle = _bundle_with_embedded_config(seed)
    bundle.metadata["config"] = {"dataset": {"task": "classification"}}
    return bundle


def _load_ndjson(path) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload.append(json.loads(line))
    return payload


def test_run_deferred_filter_writes_manifest_and_summary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(101), _bundle_with_embedded_config(102)]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=2, compression="zstd")

    def _stub_filter(*_args, **_kwargs):
        seed = int(_kwargs["seed"])
        accepted = bool(seed % 2)
        details = {"wins_ratio": 1.0 if accepted else 0.0, "n_valid_oob": 128}
        if not accepted:
            details["reason"] = "below_threshold"
        return accepted, details

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_extra_trees_filter", _stub_filter)

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)
    assert result.total_datasets == 2
    assert result.accepted_datasets == 1
    assert result.rejected_datasets == 1

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["accepted_datasets"] == 1
    assert summary["rejected_datasets"] == 1
    assert summary["rejected_reason_counts"]["below_threshold"] == 1

    manifest_records = _load_ndjson(result.manifest_path)
    assert len(manifest_records) == 2
    statuses = {str(record["status"]) for record in manifest_records}
    assert statuses == {"accepted", "rejected"}
    for record in manifest_records:
        filter_payload = record["filter"]
        assert isinstance(filter_payload, dict)
        assert filter_payload["mode"] == "deferred"
        assert filter_payload["status"] in {"accepted", "rejected"}


def test_run_deferred_filter_writes_curated_output_for_accepted_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    curated_out = tmp_path / "curated"
    bundles = [
        _bundle_with_embedded_config(201),
        _bundle_with_embedded_config(202),
        _bundle_with_embedded_config(203),
    ]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=3, compression="zstd")

    def _stub_filter(*_args, **_kwargs):
        seed = int(_kwargs["seed"])
        accepted = bool(seed % 2)
        return accepted, {"wins_ratio": 1.0 if accepted else 0.0, "n_valid_oob": 128}

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_extra_trees_filter", _stub_filter)

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir, curated_out_dir=curated_out)
    assert result.curated_accepted_datasets == 2

    shard_dir = curated_out / "shard_00000"
    assert shard_dir.exists()
    metadata_records = _load_ndjson(shard_dir / "metadata.ndjson")
    assert [int(record["dataset_index"]) for record in metadata_records] == [0, 2]

    train_table = pyarrow_parquet.read_table(shard_dir / "train.parquet")
    dataset_indices = {int(value) for value in train_table.column("dataset_index").to_pylist()}
    assert dataset_indices == {0, 2}


def test_run_deferred_filter_requires_fallback_config_when_metadata_lacks_filter_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_without_config(7)]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_extra_trees_filter",
        lambda *_args, **_kwargs: (True, {"wins_ratio": 1.0, "n_valid_oob": 128}),
    )

    with pytest.raises(ValueError, match="Missing filter config in shard metadata"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)

    cfg = GeneratorConfig()
    cfg.dataset.task = "classification"
    cfg.filter.enabled = True

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir, config=cfg)
    assert result.total_datasets == 1
    assert result.accepted_datasets == 1


def test_run_deferred_filter_prefers_dataset_seed_when_present(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [
        _bundle_with_embedded_config(101, dataset_seed=501, dataset_index=0),
        _bundle_with_embedded_config(102, dataset_seed=502, dataset_index=1),
    ]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=2, compression="zstd")

    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return True, {"wins_ratio": 1.0, "n_valid_oob": 128}

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_extra_trees_filter", _stub_filter)

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)

    assert result.total_datasets == 2
    assert result.accepted_datasets == 2
    assert replay_seeds == [501, 502]
