from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from conftest import load_repo_config, write_yaml

from dagzoo.cli import main
from dagzoo.cli.request_execution import run_request_execution
from dagzoo.config import (
    MISSINGNESS_MECHANISM_MCAR,
    REQUEST_FILE_VERSION_V1,
    REQUEST_PROFILE_DEFAULT,
    REQUEST_PROFILE_SMOKE,
    REQUEST_TASK_CLASSIFICATION,
    REQUEST_TASK_REGRESSION,
    GeneratorConfig,
    RequestFileConfig,
)
from dagzoo.config.io import load_packaged_generator_config
from dagzoo.core.config_resolution import resolve_request_config, serialize_resolution_events
from dagzoo.core.request_handoff import (
    REQUEST_HANDOFF_SCHEMA_NAME,
    validate_request_handoff_manifest,
)
from dagzoo.filtering import DeferredFilterRunResult
from dagzoo.hardware import HardwareInfo


def _mock_cuda_h100(_requested_device: str) -> HardwareInfo:
    return HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA H100 SXM",
        total_memory_gb=80.0,
        peak_flops=989e12,
        tier="cuda_h100",
    )


def _request_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": REQUEST_FILE_VERSION_V1,
        "task": REQUEST_TASK_CLASSIFICATION,
        "dataset_count": 2,
        "rows": 1024,
        "profile": REQUEST_PROFILE_DEFAULT,
        "output_root": "requests/out",
    }
    payload.update(overrides)
    return payload


def _load_ndjson(path: Path) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload.append(json.loads(line))
    return payload


@pytest.mark.parametrize(
    "resource_name",
    [
        "default.yaml",
        "preset_missingness_mcar.yaml",
        "preset_missingness_mar.yaml",
        "preset_missingness_mnar.yaml",
    ],
)
def test_load_packaged_request_config_matches_repo_copy(resource_name: str) -> None:
    packaged = load_packaged_generator_config(resource_name)
    repo = load_repo_config(resource_name)

    assert packaged.to_dict() == repo.to_dict()


def test_resolve_request_config_loads_packaged_request_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_resources: list[str] = []

    def _stub_load_packaged_generator_config(resource_name: str) -> GeneratorConfig:
        loaded_resources.append(resource_name)
        return load_repo_config(resource_name)

    monkeypatch.setattr(
        "dagzoo.core.config_resolution.load_packaged_generator_config",
        _stub_load_packaged_generator_config,
    )

    request = RequestFileConfig.from_dict(
        _request_payload(missingness_profile=MISSINGNESS_MECHANISM_MCAR)
    )

    resolved = resolve_request_config(
        request=request,
        device_override="cpu",
        hardware_policy="none",
    )

    assert loaded_resources == ["default.yaml", "preset_missingness_mcar.yaml"]
    assert resolved.config.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MCAR


def test_resolve_request_config_maps_output_root_seed_and_rows() -> None:
    request = RequestFileConfig.from_dict(
        _request_payload(
            output_root="requests/demo_run",
            seed=123,
        )
    )

    resolved = resolve_request_config(
        request=request,
        device_override="cpu",
        hardware_policy="none",
    )

    assert resolved.config.seed == 123
    assert resolved.requested_device == "cpu"
    assert resolved.config.runtime.device == "cpu"
    assert resolved.config.output.out_dir == "requests/demo_run/generated"
    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "fixed"
    assert resolved.config.dataset.rows.value == 1024

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "seed" and event["source"] == "request.seed" and event["new_value"] == 123
        for event in trace
    )
    assert any(
        event["path"] == "output.out_dir"
        and event["source"] == "request.output_root"
        and event["new_value"] == "requests/demo_run/generated"
        for event in trace
    )


def test_resolve_request_config_applies_smoke_profile_without_overriding_rows() -> None:
    request = RequestFileConfig.from_dict(
        _request_payload(
            task=REQUEST_TASK_REGRESSION,
            profile=REQUEST_PROFILE_SMOKE,
            rows=1024,
        )
    )

    resolved = resolve_request_config(
        request=request,
        device_override="cpu",
        hardware_policy="none",
    )

    assert resolved.config.dataset.task == REQUEST_TASK_REGRESSION
    assert resolved.config.dataset.n_train == 128
    assert resolved.config.dataset.n_test == 32
    assert resolved.config.dataset.n_features_min == 8
    assert resolved.config.dataset.n_features_max == 12
    assert resolved.config.graph.n_nodes_min == 2
    assert resolved.config.graph.n_nodes_max == 12
    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "fixed"
    assert resolved.config.dataset.rows.value == 1024

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(event["source"] == "request.profile_smoke" for event in trace)
    assert any(
        event["path"] == "dataset.rows" and event["source"] == "request.rows" for event in trace
    )


def test_resolve_request_config_preserves_smoke_caps_under_cuda_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    request = RequestFileConfig.from_dict(
        _request_payload(
            task=REQUEST_TASK_REGRESSION,
            profile=REQUEST_PROFILE_SMOKE,
            rows=1024,
        )
    )

    resolved = resolve_request_config(
        request=request,
        device_override="cuda",
        hardware_policy="cuda_tiered_v1",
    )

    assert resolved.requested_device == "cuda"
    assert resolved.config.dataset.n_train == 128
    assert resolved.config.dataset.n_test == 32
    assert resolved.config.dataset.n_features_max == 12
    assert resolved.config.graph.n_nodes_max == 12
    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "fixed"
    assert resolved.config.dataset.rows.value == 1024
    assert resolved.config.runtime.fixed_layout_target_cells == 160_000_000


def test_resolve_request_config_reapplies_non_smoke_rows_after_cuda_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    request = RequestFileConfig.from_dict(
        _request_payload(
            profile=REQUEST_PROFILE_DEFAULT,
            rows="2000..60000",
        )
    )

    resolved = resolve_request_config(
        request=request,
        device_override="cuda",
        hardware_policy="cuda_tiered_v1",
    )

    assert resolved.requested_device == "cuda"
    assert resolved.config.dataset.n_test == 1024
    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "range"
    assert resolved.config.dataset.rows.start == 2000
    assert resolved.config.dataset.rows.stop == 60000

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "dataset.rows"
        and event["source"] == "request.rows"
        and event["new_value"]
        == {
            "mode": "range",
            "value": None,
            "start": 2000,
            "stop": 60000,
            "choices": [],
        }
        for event in trace
    )


def test_resolve_request_config_applies_missingness_profile_without_task_leakage() -> None:
    request = RequestFileConfig.from_dict(
        _request_payload(
            task=REQUEST_TASK_REGRESSION,
            missingness_profile=MISSINGNESS_MECHANISM_MCAR,
        )
    )

    resolved = resolve_request_config(
        request=request,
        device_override="cpu",
        hardware_policy="none",
    )

    assert resolved.config.dataset.task == REQUEST_TASK_REGRESSION
    assert resolved.config.dataset.missing_rate == pytest.approx(0.2)
    assert resolved.config.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MCAR
    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "dataset.missing_mechanism"
        and event["source"] == "request.missingness_profile"
        and event["new_value"] == MISSINGNESS_MECHANISM_MCAR
        for event in trace
    )


def test_request_cli_end_to_end_writes_generated_filter_and_curated_outputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "request_run"
    request_path = write_yaml(
        tmp_path,
        "request.yaml",
        _request_payload(
            task=REQUEST_TASK_REGRESSION,
            dataset_count=1,
            rows=1024,
            profile=REQUEST_PROFILE_SMOKE,
            output_root=str(output_root),
        ),
    )

    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 1.0, "n_valid_oob": 128}

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter._apply_extra_trees_filter_numpy", _stub_filter
    )

    code = main(
        [
            "request",
            "--request",
            str(request_path),
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (output_root / "generated" / "effective_config.yaml").read_text(encoding="utf-8")
    )
    trace_payload = yaml.safe_load(
        (output_root / "generated" / "effective_config_trace.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["output"]["out_dir"] == str(output_root / "generated")
    assert effective_config["dataset"]["rows"] is None
    assert int(effective_config["dataset"]["n_train"]) == 128
    assert int(effective_config["dataset"]["n_test"]) == 32
    assert any(
        isinstance(item, dict) and item.get("source") == "request.smoke_rows_cap"
        for item in trace_payload
    )
    generated_metadata_path = output_root / "generated" / "shard_00000" / "metadata.ndjson"
    assert generated_metadata_path.exists()
    assert (output_root / "filter" / "filter_manifest.ndjson").exists()
    summary = json.loads((output_root / "filter" / "filter_summary.json").read_text("utf-8"))
    assert summary["total_datasets"] == 1
    assert summary["accepted_datasets"] == 1
    assert summary["rejected_datasets"] == 0
    curated_metadata_path = output_root / "curated" / "shard_00000" / "metadata.ndjson"
    assert curated_metadata_path.exists()
    generated_metadata = _load_ndjson(generated_metadata_path)
    curated_metadata = _load_ndjson(curated_metadata_path)
    assert len(generated_metadata) == 1
    assert len(curated_metadata) == 1
    generated_record = generated_metadata[0]["metadata"]
    curated_record = curated_metadata[0]["metadata"]
    assert isinstance(generated_record, dict)
    assert isinstance(curated_record, dict)
    assert isinstance(generated_record.get("dataset_id"), str)
    assert generated_record["split_groups"] == curated_record["split_groups"]
    assert generated_record["dataset_id"] == curated_record["dataset_id"]
    handoff = json.loads((output_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_request_handoff_manifest(handoff)
    assert handoff["schema_name"] == REQUEST_HANDOFF_SCHEMA_NAME
    assert handoff["artifacts"]["filtered_corpus_dir"] == str((output_root / "curated").resolve())
    assert handoff["artifacts"]["filter_summary_path"] == str(
        (output_root / "filter" / "filter_summary.json").resolve()
    )
    assert handoff["artifacts_relative"]["filtered_corpus_dir"] == "curated"
    assert handoff["defaults"]["recommended_training_corpus"] == "curated"
    assert handoff["defaults"]["curation_policy"] == "accepted_only"
    assert handoff["identity"]["source_family"] == "dagzoo.fixed_layout_scm"
    assert len(handoff["checksums"]["filter_summary_sha256"]) == 64
    assert handoff["summary"]["accepted_datasets"] == 1
    assert handoff["diversity_artifacts"]["summary_json_path"] is None


def test_request_execution_manifest_uses_wall_clock_filter_timing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "request_run"
    request_path = write_yaml(
        tmp_path,
        "request.yaml",
        _request_payload(
            task=REQUEST_TASK_REGRESSION,
            dataset_count=2,
            rows=1024,
            profile=REQUEST_PROFILE_DEFAULT,
            output_root=str(output_root),
        ),
    )

    perf_counter_values = iter((10.0, 16.0, 20.0, 50.0))
    monkeypatch.setattr(
        "dagzoo.cli.request_execution.perf_counter", lambda: next(perf_counter_values)
    )

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

    def _stub_run_deferred_filter(**kwargs) -> DeferredFilterRunResult:
        assert kwargs["in_dir"] == output_root / "generated"
        assert kwargs["out_dir"] == output_root / "filter"
        assert kwargs["curated_out_dir"] == output_root / "curated"
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        kwargs["curated_out_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["out_dir"] / "filter_manifest.ndjson").write_text(
            '{"dataset_index": 0, "accepted": true}\n',
            encoding="utf-8",
        )
        (kwargs["out_dir"] / "filter_summary.json").write_text(
            '{"accepted_datasets": 1, "rejected_datasets": 1}\n',
            encoding="utf-8",
        )
        return DeferredFilterRunResult(
            manifest_path=output_root / "filter" / "filter_manifest.ndjson",
            summary_path=output_root / "filter" / "filter_summary.json",
            total_datasets=2,
            accepted_datasets=1,
            rejected_datasets=1,
            elapsed_seconds=0.5,
            datasets_per_minute=999.0,
            curated_out_dir=output_root / "curated",
            curated_accepted_datasets=1,
        )

    monkeypatch.setattr(
        "dagzoo.cli.request_execution.write_packed_parquet_shards_stream",
        _stub_write_packed_parquet_shards_stream,
    )
    monkeypatch.setattr(
        "dagzoo.cli.request_execution.get_cli_public_api",
        lambda: SimpleNamespace(
            generate_batch_iter=lambda *_args, **_kwargs: iter(()),
            run_deferred_filter=_stub_run_deferred_filter,
        ),
    )

    _ = run_request_execution(
        request_path=request_path,
        device_override="cpu",
        hardware_policy="none",
        n_jobs_override=None,
        print_effective_config_flag=False,
        print_resolution_trace_flag=False,
    )

    handoff = json.loads((output_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_request_handoff_manifest(handoff)
    assert handoff["throughput"]["generation_stage"]["elapsed_seconds"] == pytest.approx(6.0)
    assert handoff["throughput"]["generation_stage"]["datasets_per_minute"] == pytest.approx(20.0)
    assert handoff["throughput"]["filter_stage"]["elapsed_seconds"] == pytest.approx(30.0)
    assert handoff["throughput"]["filter_stage"]["datasets_per_minute"] == pytest.approx(4.0)


def test_request_handoff_identity_is_stable_after_request_run_directory_move(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "request_run"
    request_path = write_yaml(
        tmp_path,
        "request.yaml",
        _request_payload(
            task=REQUEST_TASK_REGRESSION,
            dataset_count=1,
            rows=1024,
            profile=REQUEST_PROFILE_SMOKE,
            output_root=str(output_root),
            seed=7,
        ),
    )

    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 1.0, "n_valid_oob": 128}

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter._apply_extra_trees_filter_numpy", _stub_filter
    )

    code = main(
        [
            "request",
            "--request",
            str(request_path),
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    original_manifest_path = output_root / "handoff_manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    validate_request_handoff_manifest(original_manifest)

    moved_root = tmp_path / "request_run_copy"
    shutil.copytree(output_root, moved_root)
    moved_manifest_path = moved_root / "handoff_manifest.json"
    moved_manifest = json.loads(moved_manifest_path.read_text(encoding="utf-8"))
    validate_request_handoff_manifest(moved_manifest)
    original_generated_records = _load_ndjson(
        output_root / "generated" / "shard_00000" / "metadata.ndjson"
    )
    moved_generated_records = _load_ndjson(
        moved_root / "generated" / "shard_00000" / "metadata.ndjson"
    )
    original_curated_records = _load_ndjson(
        output_root / "curated" / "shard_00000" / "metadata.ndjson"
    )
    moved_curated_records = _load_ndjson(moved_root / "curated" / "shard_00000" / "metadata.ndjson")

    assert moved_manifest["identity"] == original_manifest["identity"]
    assert moved_manifest["checksums"] == original_manifest["checksums"]
    assert moved_generated_records == original_generated_records
    assert moved_curated_records == original_curated_records
    for key, relative_path in moved_manifest["artifacts_relative"].items():
        resolved = (moved_root / relative_path).resolve()
        if key == "run_root":
            assert resolved == moved_root.resolve()
        else:
            assert resolved.exists()
