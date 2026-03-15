from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from dagzoo.cli import main
from dagzoo.core.generate_handoff import (
    GENERATE_HANDOFF_SCHEMA_NAME,
    GENERATE_HANDOFF_SCHEMA_VERSION,
    build_generate_handoff_manifest,
    validate_generate_handoff_manifest,
    write_generate_handoff_manifest,
)


def _generate_overrides(handoff_root: str) -> dict[str, object]:
    return {
        "num_datasets": 2,
        "seed": 7,
        "rows": "1024..4096",
        "device": "cpu",
        "hardware_policy": "none",
        "missing_rate": None,
        "missing_mechanism": None,
        "missing_mar_observed_fraction": None,
        "missing_mar_logit_scale": None,
        "missing_mnar_logit_scale": None,
        "diagnostics": False,
        "diagnostics_out_dir": None,
        "handoff_root": handoff_root,
    }


def _write_generate_run_artifacts(run_root: Path) -> None:
    generated_dir = run_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "effective_config.yaml").write_text("seed: 7\n", encoding="utf-8")
    (generated_dir / "effective_config_trace.yaml").write_text("- source: test\n", encoding="utf-8")


def _load_ndjson(path: Path) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload.append(json.loads(line))
    return payload


def test_build_generate_handoff_manifest_is_versioned_and_valid(tmp_path) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)
    payload = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    validate_generate_handoff_manifest(payload)

    assert payload["schema_name"] == GENERATE_HANDOFF_SCHEMA_NAME
    assert payload["schema_version"] == GENERATE_HANDOFF_SCHEMA_VERSION
    assert payload["identity"]["source_family"] == "dagzoo.fixed_layout_scm"
    assert len(payload["identity"]["generate_run_id"]) == 32
    assert len(payload["identity"]["generated_corpus_id"]) == 32
    assert payload["generate_invocation"]["config_path"] == str(
        Path("configs/default.yaml").resolve()
    )
    assert payload["generate_invocation"]["overrides"]["handoff_root"] == str(run_root.resolve())
    assert payload["artifacts"]["generated_dir"] == str((run_root / "generated").resolve())
    assert payload["artifacts_relative"] == {
        "run_root": ".",
        "generated_dir": "generated",
        "effective_config_path": "generated/effective_config.yaml",
        "effective_config_trace_path": "generated/effective_config_trace.yaml",
    }
    assert len(payload["checksums"]["effective_config_sha256"]) == 64
    assert payload["defaults"] == {
        "recommended_training_corpus": "generated",
        "recommended_training_artifact_key": "generated_dir",
        "curation_policy": "none",
    }
    assert payload["summary"]["generated_datasets"] == 2
    assert payload["throughput"]["generation_stage"]["datasets_per_minute"] == pytest.approx(10.0)
    assert payload["diversity_artifacts"] == {
        "summary_json_path": None,
        "summary_md_path": None,
    }


def test_write_generate_handoff_manifest_writes_json_and_rejects_invalid_payload(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)
    manifest_path = write_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
        out_path=run_root / "handoff_manifest.json",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(payload)
    assert payload["summary"]["generated_datasets"] == 2
    assert payload["defaults"]["recommended_training_corpus"] == "generated"
    assert list(run_root.glob(".handoff_manifest.json.*.tmp")) == []

    payload["generate_invocation"]["overrides"]["num_datasets"] = "two"
    with pytest.raises(
        ValueError,
        match=r"handoff_manifest.generate_invocation.overrides.num_datasets: must be an integer",
    ):
        validate_generate_handoff_manifest(payload)


def test_write_generate_handoff_manifest_does_not_overwrite_existing_file(tmp_path) -> None:
    run_root = tmp_path / "run"
    manifest_path = run_root / "handoff_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"sentinel": true}\n', encoding="utf-8")
    _write_generate_run_artifacts(run_root)

    with pytest.raises(RuntimeError, match="promotion target already exists"):
        _ = write_generate_handoff_manifest(
            config_path="configs/default.yaml",
            generate_invocation_overrides=_generate_overrides(str(run_root)),
            run_root=run_root,
            generated_dir=run_root / "generated",
            effective_config_path=run_root / "generated" / "effective_config.yaml",
            effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
            generated_datasets=2,
            generation_elapsed_seconds=12.0,
            requested_device="cpu",
            resolved_device="cpu",
            hardware_backend="cpu",
            hardware_device_name="CPU",
            hardware_tier="cpu",
            hardware_policy="none",
            out_path=manifest_path,
        )

    assert manifest_path.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    assert list(manifest_path.parent.glob(".handoff_manifest.json.*.tmp")) == []


def test_generate_cli_handoff_root_writes_generated_outputs_and_manifest(tmp_path) -> None:
    handoff_root = tmp_path / "handoff_run"

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (handoff_root / "generated" / "effective_config.yaml").read_text(encoding="utf-8")
    )
    trace_payload = yaml.safe_load(
        (handoff_root / "generated" / "effective_config_trace.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["output"]["out_dir"] == str((handoff_root / "generated").resolve())
    assert any(
        isinstance(item, dict)
        and item.get("source") == "generate.handoff_root"
        and item.get("path") == "output.out_dir"
        for item in trace_payload
    )

    generated_metadata_path = handoff_root / "generated" / "shard_00000" / "metadata.ndjson"
    assert generated_metadata_path.exists()
    generated_metadata = _load_ndjson(generated_metadata_path)
    assert len(generated_metadata) == 1

    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["schema_name"] == GENERATE_HANDOFF_SCHEMA_NAME
    assert handoff["artifacts"]["generated_dir"] == str((handoff_root / "generated").resolve())
    assert handoff["artifacts_relative"]["generated_dir"] == "generated"
    assert handoff["defaults"]["recommended_training_corpus"] == "generated"
    assert handoff["defaults"]["recommended_training_artifact_key"] == "generated_dir"
    assert handoff["defaults"]["curation_policy"] == "none"
    assert handoff["identity"]["source_family"] == "dagzoo.fixed_layout_scm"
    assert handoff["generate_invocation"]["config_path"] == str(
        Path("configs/default.yaml").resolve()
    )
    assert handoff["generate_invocation"]["overrides"]["num_datasets"] == 1
    assert handoff["generate_invocation"]["overrides"]["handoff_root"] == str(
        handoff_root.resolve()
    )
    assert len(handoff["checksums"]["effective_config_sha256"]) == 64
    assert handoff["summary"]["generated_datasets"] == 1
    assert handoff["diversity_artifacts"]["summary_json_path"] is None


def test_generate_handoff_manifest_uses_wall_clock_generation_timing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = tmp_path / "handoff_run"
    perf_counter_values = iter((10.0, 16.0))
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.perf_counter", lambda: next(perf_counter_values)
    )

    def _stub_generate_batch_iter(*_args, **_kwargs):
        return iter(())

    def _stub_write_packed_parquet_shards_stream(
        _stream,
        *,
        out_dir: Path,
        shard_size: int,
        compression: str,
    ) -> int:
        assert shard_size > 0
        assert compression
        out_dir.mkdir(parents=True, exist_ok=True)
        return 2

    monkeypatch.setattr("dagzoo.cli.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.write_packed_parquet_shards_stream",
        _stub_write_packed_parquet_shards_stream,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["throughput"]["generation_stage"]["elapsed_seconds"] == pytest.approx(6.0)
    assert handoff["throughput"]["generation_stage"]["datasets_per_minute"] == pytest.approx(20.0)


def test_generate_cli_handoff_root_preserves_rows_under_cuda_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    patch_detect_hardware,
) -> None:
    handoff_root = tmp_path / "handoff_cuda"
    patch_detect_hardware("cuda_h100", "dagzoo.core.config_resolution.detect_hardware")
    monkeypatch.setattr("dagzoo.core.generation_context.torch.cuda.is_available", lambda: True)

    def _stub_generate_batch_iter(*_args, **_kwargs):
        return iter(())

    def _stub_write_packed_parquet_shards_stream(
        _stream,
        *,
        out_dir: Path,
        shard_size: int,
        compression: str,
    ) -> int:
        assert shard_size > 0
        assert compression
        out_dir.mkdir(parents=True, exist_ok=True)
        return 1

    monkeypatch.setattr("dagzoo.cli.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.write_packed_parquet_shards_stream",
        _stub_write_packed_parquet_shards_stream,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "1",
            "--rows",
            "2000..60000",
            "--device",
            "cuda",
            "--hardware-policy",
            "cuda_tiered_v1",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (handoff_root / "generated" / "effective_config.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["dataset"]["rows"]["mode"] == "fixed"
    assert int(effective_config["runtime"]["fixed_layout_target_cells"]) == 160_000_000

    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["hardware"]["tier"] == "cuda_h100"
    assert handoff["hardware"]["resolved_device"] == "cuda"
    assert handoff["generate_invocation"]["overrides"]["rows"] == "2000..60000"


def test_generate_handoff_identity_is_stable_after_handoff_root_move(tmp_path) -> None:
    handoff_root = tmp_path / "handoff_run"

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "1",
            "--rows",
            "1024",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    original_manifest_path = handoff_root / "handoff_manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(original_manifest)

    moved_root = tmp_path / "handoff_run_copy"
    shutil.copytree(handoff_root, moved_root)
    moved_manifest_path = moved_root / "handoff_manifest.json"
    moved_manifest = json.loads(moved_manifest_path.read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(moved_manifest)
    original_generated_records = _load_ndjson(
        handoff_root / "generated" / "shard_00000" / "metadata.ndjson"
    )
    moved_generated_records = _load_ndjson(
        moved_root / "generated" / "shard_00000" / "metadata.ndjson"
    )

    assert moved_manifest["identity"] == original_manifest["identity"]
    assert moved_manifest["checksums"] == original_manifest["checksums"]
    assert moved_generated_records == original_generated_records
    for key, relative_path in moved_manifest["artifacts_relative"].items():
        resolved = (moved_root / relative_path).resolve()
        if key == "run_root":
            assert resolved == moved_root.resolve()
        else:
            assert resolved.exists()
