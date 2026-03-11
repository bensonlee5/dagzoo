from __future__ import annotations

import yaml
import pytest

from dagzoo.config import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    REQUEST_FILE_VERSION_V1,
    REQUEST_PROFILE_DEFAULT,
    REQUEST_PROFILE_SMOKE,
    REQUEST_TASK_CLASSIFICATION,
    REQUEST_TASK_REGRESSION,
    RequestFileConfig,
)


def test_request_file_accepts_minimal_classification_payload() -> None:
    cfg = RequestFileConfig.from_dict(
        {
            "version": REQUEST_FILE_VERSION_V1,
            "task": REQUEST_TASK_CLASSIFICATION,
            "dataset_count": 25,
            "rows": 1024,
            "profile": REQUEST_PROFILE_DEFAULT,
            "output_root": "requests/corpus_default",
        }
    )

    assert cfg.version == REQUEST_FILE_VERSION_V1
    assert cfg.task == REQUEST_TASK_CLASSIFICATION
    assert cfg.dataset_count == 25
    assert cfg.rows.mode == "fixed"
    assert cfg.rows.value == 1024
    assert cfg.profile == REQUEST_PROFILE_DEFAULT
    assert cfg.output_root == "requests/corpus_default"
    assert cfg.missingness_profile == MISSINGNESS_MECHANISM_NONE
    assert cfg.seed is None


def test_request_file_accepts_range_rows_smoke_profile_and_seed() -> None:
    cfg = RequestFileConfig.from_dict(
        {
            "version": "V1",
            "task": REQUEST_TASK_REGRESSION,
            "dataset_count": 3,
            "rows": "1024..4096",
            "profile": "SMOKE",
            "missingness_profile": "mcar",
            "output_root": "requests/regression_smoke",
            "seed": 42,
        }
    )

    assert cfg.version == REQUEST_FILE_VERSION_V1
    assert cfg.task == REQUEST_TASK_REGRESSION
    assert cfg.rows.mode == "range"
    assert cfg.rows.start == 1024
    assert cfg.rows.stop == 4096
    assert cfg.profile == REQUEST_PROFILE_SMOKE
    assert cfg.missingness_profile == MISSINGNESS_MECHANISM_MCAR
    assert cfg.seed == 42


def test_request_file_accepts_choice_rows_and_each_missingness_profile() -> None:
    for missingness_profile in (
        MISSINGNESS_MECHANISM_NONE,
        MISSINGNESS_MECHANISM_MCAR,
        MISSINGNESS_MECHANISM_MAR,
        MISSINGNESS_MECHANISM_MNAR,
    ):
        cfg = RequestFileConfig.from_dict(
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 8,
                "rows": "1024,2048,4096",
                "profile": REQUEST_PROFILE_DEFAULT,
                "missingness_profile": missingness_profile,
                "output_root": "requests/profile_matrix",
            }
        )
        assert cfg.rows.mode == "choices"
        assert cfg.rows.choices == [1024, 2048, 4096]
        assert cfg.missingness_profile == missingness_profile


def test_request_file_from_yaml_loads_top_level_mapping(tmp_path) -> None:
    path = tmp_path / "request.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 10,
                "rows": 2048,
                "profile": REQUEST_PROFILE_DEFAULT,
                "missingness_profile": MISSINGNESS_MECHANISM_MAR,
                "output_root": "requests/mar_default",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = RequestFileConfig.from_yaml(path)

    assert cfg.task == REQUEST_TASK_CLASSIFICATION
    assert cfg.dataset_count == 10
    assert cfg.rows.mode == "fixed"
    assert cfg.rows.value == 2048
    assert cfg.missingness_profile == MISSINGNESS_MECHANISM_MAR


@pytest.mark.parametrize(
    "rows_value",
    [
        [1024, 2048],
        {"mode": "fixed", "value": 1024},
    ],
)
def test_request_file_from_yaml_rejects_non_public_rows_encodings(
    tmp_path, rows_value: object
) -> None:
    path = tmp_path / "request.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 2,
                "rows": rows_value,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"rows must use one of the public request-file encodings",
    ):
        RequestFileConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 25,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/corpus_default",
            },
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 25,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/corpus_default",
                "missingness_profile": MISSINGNESS_MECHANISM_NONE,
            },
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_REGRESSION,
                "dataset_count": 3,
                "rows": "1024..4096",
                "profile": REQUEST_PROFILE_SMOKE,
                "missingness_profile": MISSINGNESS_MECHANISM_MCAR,
                "output_root": "requests/regression_smoke",
                "seed": 42,
            },
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_REGRESSION,
                "dataset_count": 3,
                "rows": "1024..4096",
                "profile": REQUEST_PROFILE_SMOKE,
                "output_root": "requests/regression_smoke",
                "missingness_profile": MISSINGNESS_MECHANISM_MCAR,
                "seed": 42,
            },
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 8,
                "rows": "1024,2048,4096",
                "profile": REQUEST_PROFILE_DEFAULT,
                "missingness_profile": MISSINGNESS_MECHANISM_MAR,
                "output_root": "requests/profile_matrix",
            },
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 8,
                "rows": "1024,2048,4096",
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/profile_matrix",
                "missingness_profile": MISSINGNESS_MECHANISM_MAR,
            },
        ),
    ],
)
def test_request_file_to_dict_emits_public_wire_shape(
    payload: dict[str, object], expected: dict[str, object]
) -> None:
    cfg = RequestFileConfig.from_dict(payload)

    serialized = cfg.to_dict()

    assert serialized == expected
    assert RequestFileConfig.from_dict(serialized).to_dict() == expected


def test_request_file_to_dict_omits_unset_seed() -> None:
    cfg = RequestFileConfig.from_dict(
        {
            "version": REQUEST_FILE_VERSION_V1,
            "task": REQUEST_TASK_CLASSIFICATION,
            "dataset_count": 25,
            "rows": 1024,
            "profile": REQUEST_PROFILE_DEFAULT,
            "output_root": "requests/corpus_default",
        }
    )

    assert "seed" not in cfg.to_dict()


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        (
            {
                "version": "v2",
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            },
            r"version must be one of: v1",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": "clustering",
                "dataset_count": 1,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            },
            r"task must be one of: classification, regression",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 0,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            },
            r"dataset_count must be a positive integer",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": "1024",
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            },
            r"rows must use one of the public request-file encodings",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": "300..600",
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            },
            r"dataset\.rows",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": 1024,
                "profile": "benchmark_cpu",
                "output_root": "requests/out",
            },
            r"profile must be one of: default, smoke",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "missingness_profile": "custom",
                "output_root": "requests/out",
            },
            r"missingness_profile must be one of: mar, mcar, mnar, none",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
                "seed": 4294967296,
            },
            r"seed must be an integer in \[0, 4294967295\]",
        ),
        (
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "",
            },
            r"output_root must be a non-empty string",
        ),
    ],
)
def test_request_file_rejects_invalid_values(payload: dict[str, object], pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        RequestFileConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field_name", ["runtime", "filter", "graph", "dataset", "n_train", "n_test"]
)
def test_request_file_rejects_internal_config_fields(field_name: str) -> None:
    payload = {
        "version": REQUEST_FILE_VERSION_V1,
        "task": REQUEST_TASK_CLASSIFICATION,
        "dataset_count": 1,
        "rows": 1024,
        "profile": REQUEST_PROFILE_DEFAULT,
        "output_root": "requests/out",
        field_name: {},
    }
    if field_name in {"n_train", "n_test"}:
        payload[field_name] = 128

    with pytest.raises(
        ValueError,
        match=r"is not part of the public request-file contract",
    ):
        RequestFileConfig.from_dict(payload)


def test_request_file_rejects_unknown_extra_fields() -> None:
    with pytest.raises(ValueError, match=r"Unknown request-file field: 'unexpected'"):
        RequestFileConfig.from_dict(
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "rows": 1024,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
                "unexpected": "value",
            }
        )


def test_request_file_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match=r"Missing required request-file field: 'rows'"):
        RequestFileConfig.from_dict(
            {
                "version": REQUEST_FILE_VERSION_V1,
                "task": REQUEST_TASK_CLASSIFICATION,
                "dataset_count": 1,
                "profile": REQUEST_PROFILE_DEFAULT,
                "output_root": "requests/out",
            }
        )
