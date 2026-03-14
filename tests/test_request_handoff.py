from __future__ import annotations

import json
from pathlib import Path

import pytest

from dagzoo.config import REQUEST_FILE_VERSION_V1, REQUEST_TASK_CLASSIFICATION, RequestFileConfig
from dagzoo.core.request_handoff import (
    REQUEST_HANDOFF_SCHEMA_NAME,
    REQUEST_HANDOFF_SCHEMA_VERSION,
    build_request_handoff_manifest,
    validate_request_handoff_manifest,
    write_request_handoff_manifest,
)


def _request_config(output_root: str) -> RequestFileConfig:
    return RequestFileConfig.from_dict(
        {
            "version": REQUEST_FILE_VERSION_V1,
            "task": REQUEST_TASK_CLASSIFICATION,
            "dataset_count": 2,
            "rows": 1024,
            "profile": "default",
            "output_root": output_root,
        }
    )


def _write_request_run_artifacts(run_root: Path) -> None:
    generated_dir = run_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "effective_config.yaml").write_text("seed: 7\n", encoding="utf-8")
    (generated_dir / "effective_config_trace.yaml").write_text("- source: test\n", encoding="utf-8")


def test_build_request_handoff_manifest_is_versioned_and_valid(tmp_path) -> None:
    request = _request_config(output_root="requests/demo")
    _write_request_run_artifacts(tmp_path / "run")
    payload = build_request_handoff_manifest(
        request_path=tmp_path / "request.yaml",
        request=request,
        run_root=tmp_path / "run",
        generated_dir=tmp_path / "run" / "generated",
        effective_config_path=tmp_path / "run" / "generated" / "effective_config.yaml",
        effective_config_trace_path=tmp_path / "run" / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    validate_request_handoff_manifest(payload)

    assert payload["schema_name"] == REQUEST_HANDOFF_SCHEMA_NAME
    assert payload["schema_version"] == REQUEST_HANDOFF_SCHEMA_VERSION
    assert payload["identity"]["source_family"] == "dagzoo.fixed_layout_scm"
    assert len(payload["identity"]["request_run_id"]) == 32
    assert len(payload["identity"]["generated_corpus_id"]) == 32
    assert payload["request"]["payload"] == request.to_dict()
    assert payload["artifacts"]["generated_dir"] == str((tmp_path / "run" / "generated").resolve())
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


def test_write_request_handoff_manifest_writes_json_and_rejects_invalid_payload(tmp_path) -> None:
    request = _request_config(output_root="requests/demo")
    _write_request_run_artifacts(tmp_path / "run")
    manifest_path = write_request_handoff_manifest(
        request_path=tmp_path / "request.yaml",
        request=request,
        run_root=tmp_path / "run",
        generated_dir=tmp_path / "run" / "generated",
        effective_config_path=tmp_path / "run" / "generated" / "effective_config.yaml",
        effective_config_trace_path=tmp_path / "run" / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
        out_path=tmp_path / "run" / "handoff_manifest.json",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_request_handoff_manifest(payload)
    assert payload["summary"]["generated_datasets"] == 2
    assert payload["defaults"]["recommended_training_corpus"] == "generated"
    assert list((tmp_path / "run").glob(".handoff_manifest.json.*.tmp")) == []

    payload["request"]["payload"]["version"] = "v2"
    with pytest.raises(
        ValueError, match=r"handoff_manifest.request.payload: version must be one of"
    ):
        validate_request_handoff_manifest(payload)


def test_write_request_handoff_manifest_does_not_overwrite_existing_file(tmp_path) -> None:
    request = _request_config(output_root="requests/demo")
    manifest_path = tmp_path / "run" / "handoff_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"sentinel": true}\n', encoding="utf-8")
    _write_request_run_artifacts(tmp_path / "run")

    with pytest.raises(RuntimeError, match="promotion target already exists"):
        _ = write_request_handoff_manifest(
            request_path=tmp_path / "request.yaml",
            request=request,
            run_root=tmp_path / "run",
            generated_dir=tmp_path / "run" / "generated",
            effective_config_path=tmp_path / "run" / "generated" / "effective_config.yaml",
            effective_config_trace_path=tmp_path
            / "run"
            / "generated"
            / "effective_config_trace.yaml",
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
