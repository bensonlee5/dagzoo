from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dagzoo.cli import main
from dagzoo.config import (
    GeneratorConfig,
    MISSINGNESS_MECHANISM_MCAR,
    REQUEST_FILE_VERSION_V1,
    REQUEST_PROFILE_DEFAULT,
    REQUEST_PROFILE_SMOKE,
    REQUEST_TASK_CLASSIFICATION,
    REQUEST_TASK_REGRESSION,
    RequestFileConfig,
)
from dagzoo.config.io import load_packaged_generator_config
from dagzoo.core.config_resolution import resolve_request_config, serialize_resolution_events
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
    repo = GeneratorConfig.from_yaml(Path("configs") / resource_name)

    assert packaged.to_dict() == repo.to_dict()


def test_resolve_request_config_loads_packaged_request_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_resources: list[str] = []

    def _stub_load_packaged_generator_config(resource_name: str) -> GeneratorConfig:
        loaded_resources.append(resource_name)
        return GeneratorConfig.from_yaml(Path("configs") / resource_name)

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
    request_path = tmp_path / "request.yaml"
    output_root = tmp_path / "request_run"
    request_path.write_text(
        yaml.safe_dump(
            _request_payload(
                task=REQUEST_TASK_REGRESSION,
                dataset_count=1,
                rows=1024,
                profile=REQUEST_PROFILE_SMOKE,
                output_root=str(output_root),
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
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
    assert (output_root / "generated" / "shard_00000" / "metadata.ndjson").exists()
    assert (output_root / "filter" / "filter_manifest.ndjson").exists()
    summary = json.loads((output_root / "filter" / "filter_summary.json").read_text("utf-8"))
    assert summary["total_datasets"] == 1
    assert summary["accepted_datasets"] == 1
    assert summary["rejected_datasets"] == 0
    assert (output_root / "curated" / "shard_00000" / "metadata.ndjson").exists()
