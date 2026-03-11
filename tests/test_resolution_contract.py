from __future__ import annotations

import pytest

from dagzoo.config import (
    REQUEST_FILE_VERSION_V1,
    REQUEST_PROFILE_DEFAULT,
    REQUEST_PROFILE_SMOKE,
    REQUEST_TASK_CLASSIFICATION,
    GeneratorConfig,
    RequestFileConfig,
)
from dagzoo.core.config_resolution import (
    resolve_generate_config,
    resolve_request_config,
    serialize_resolution_events,
)


def _request_payload(*, rows: object, profile: str = REQUEST_PROFILE_DEFAULT) -> dict[str, object]:
    return {
        "version": REQUEST_FILE_VERSION_V1,
        "task": REQUEST_TASK_CLASSIFICATION,
        "dataset_count": 2,
        "rows": rows,
        "profile": profile,
        "output_root": "requests/out",
    }


@pytest.mark.parametrize(
    ("tier", "device_override", "hardware_policy", "rows"),
    [
        ("cpu", "cpu", "none", 1024),
        ("cpu", "cpu", "none", "400..60000"),
        ("cpu", "cpu", "none", "1024,2048,4096"),
        ("cuda_desktop", "cuda", "cuda_tiered_v1", 1024),
        ("cuda_desktop", "cuda", "cuda_tiered_v1", "400..60000"),
        ("cuda_datacenter", "cuda", "cuda_tiered_v1", 1024),
        ("cuda_datacenter", "cuda", "cuda_tiered_v1", "1024..4096"),
        ("cuda_datacenter", "cuda", "cuda_tiered_v1", "1024,2048,4096"),
        ("cuda_h100", "cuda", "cuda_tiered_v1", 2048),
        ("cuda_h100", "cuda", "cuda_tiered_v1", "2000..60000"),
        ("cuda_h100", "cuda", "cuda_tiered_v1", "2000,4000,8000"),
    ],
)
def test_non_smoke_request_resolution_matches_generate_rows_contract(
    patch_detect_hardware,
    tier: str,
    device_override: str,
    hardware_policy: str,
    rows: object,
) -> None:
    patch_detect_hardware(tier, "dagzoo.core.config_resolution.detect_hardware")
    config = GeneratorConfig.from_yaml("configs/default.yaml")
    request = RequestFileConfig.from_dict(_request_payload(rows=rows))

    resolved_generate = resolve_generate_config(
        config,
        device_override=device_override,
        rows=rows,
        hardware_policy=hardware_policy,
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )
    resolved_request = resolve_request_config(
        request=request,
        device_override=device_override,
        hardware_policy=hardware_policy,
    )

    assert resolved_request.requested_device == resolved_generate.requested_device
    assert resolved_request.config.dataset.rows == resolved_generate.config.dataset.rows
    assert resolved_request.config.dataset.n_test == resolved_generate.config.dataset.n_test
    assert (
        resolved_request.config.runtime.fixed_layout_target_cells
        == resolved_generate.config.runtime.fixed_layout_target_cells
    )

    request_trace = serialize_resolution_events(resolved_request.trace_events)
    assert any(
        event["path"] == "dataset.rows" and event["source"] == "request.rows"
        for event in request_trace
    )


@pytest.mark.parametrize("rows", [1024, "1024..4096"])
def test_non_smoke_request_resolution_matches_generate_rows_failures(
    patch_detect_hardware,
    rows: object,
) -> None:
    patch_detect_hardware("cuda_h100", "dagzoo.core.config_resolution.detect_hardware")
    config = GeneratorConfig.from_yaml("configs/default.yaml")
    request = RequestFileConfig.from_dict(_request_payload(rows=rows))
    expected_error = "dataset.rows minimum total rows must be > dataset.n_test"

    with pytest.raises(ValueError, match=expected_error) as generate_exc_info:
        resolve_generate_config(
            config,
            device_override="cuda",
            rows=rows,
            hardware_policy="cuda_tiered_v1",
            missing_rate=None,
            missing_mechanism=None,
            missing_mar_observed_fraction=None,
            missing_mar_logit_scale=None,
            missing_mnar_logit_scale=None,
            diagnostics_enabled=False,
        )

    with pytest.raises(ValueError, match=expected_error) as request_exc_info:
        resolve_request_config(
            request=request,
            device_override="cuda",
            hardware_policy="cuda_tiered_v1",
        )

    assert str(request_exc_info.value) == str(generate_exc_info.value)


@pytest.mark.parametrize(
    ("tier", "device_override", "hardware_policy", "expected_target_cells"),
    [
        ("cpu", "cpu", "none", None),
        ("cuda_desktop", "cuda", "cuda_tiered_v1", 48_000_000),
        ("cuda_datacenter", "cuda", "cuda_tiered_v1", 160_000_000),
        ("cuda_h100", "cuda", "cuda_tiered_v1", 160_000_000),
    ],
)
def test_smoke_request_resolution_preserves_smoke_envelope_across_policies(
    patch_detect_hardware,
    tier: str,
    device_override: str,
    hardware_policy: str,
    expected_target_cells: int | None,
) -> None:
    patch_detect_hardware(tier, "dagzoo.core.config_resolution.detect_hardware")
    request = RequestFileConfig.from_dict(
        _request_payload(rows=1024, profile=REQUEST_PROFILE_SMOKE)
    )

    resolved_request = resolve_request_config(
        request=request,
        device_override=device_override,
        hardware_policy=hardware_policy,
    )

    assert resolved_request.config.dataset.n_train == 128
    assert resolved_request.config.dataset.n_test == 32
    assert resolved_request.config.dataset.n_features_min == 8
    assert resolved_request.config.dataset.n_features_max == 12
    assert resolved_request.config.graph.n_nodes_min == 2
    assert resolved_request.config.graph.n_nodes_max == 12
    assert resolved_request.config.dataset.rows is not None
    assert resolved_request.config.dataset.rows.mode == "fixed"
    assert resolved_request.config.dataset.rows.value == 1024
    assert resolved_request.config.runtime.fixed_layout_target_cells == expected_target_cells

    request_trace = serialize_resolution_events(resolved_request.trace_events)
    assert any(event["source"] == "request.profile_smoke" for event in request_trace)
    assert any(
        event["path"] == "dataset.rows" and event["source"] == "request.rows"
        for event in request_trace
    )
