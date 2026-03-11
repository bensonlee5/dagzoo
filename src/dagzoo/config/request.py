"""Public request-file contract for downstream handoff workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from dagzoo.config.constants import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
)
from dagzoo.config.rows import DatasetRowsSpec, normalize_dataset_rows
from dagzoo.rng import SEED32_MAX, SEED32_MIN

REQUEST_FILE_VERSION_V1 = "v1"
REQUEST_PROFILE_DEFAULT = "default"
REQUEST_PROFILE_SMOKE = "smoke"
REQUEST_TASK_CLASSIFICATION = "classification"
REQUEST_TASK_REGRESSION = "regression"

_REQUEST_ALLOWED_KEYS = {
    "dataset_count",
    "missingness_profile",
    "output_root",
    "profile",
    "rows",
    "seed",
    "task",
    "version",
}
_REQUEST_FORBIDDEN_KEYS = {
    "benchmark",
    "dataset",
    "diagnostics",
    "filter",
    "graph",
    "mechanism",
    "n_test",
    "n_train",
    "noise",
    "output",
    "runtime",
    "shift",
}
_REQUEST_PROFILES = {REQUEST_PROFILE_DEFAULT, REQUEST_PROFILE_SMOKE}
_REQUEST_TASKS = {REQUEST_TASK_CLASSIFICATION, REQUEST_TASK_REGRESSION}
_REQUEST_MISSINGNESS_PROFILES = {
    MISSINGNESS_MECHANISM_NONE,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MNAR,
}
_REQUEST_ROWS_PUBLIC_SHAPE_ERROR = (
    "rows must use one of the public request-file encodings: "
    "fixed integer, range string start..stop, or CSV choice string."
)


def _validate_non_empty_string(*, field_name: str, value: object) -> str:
    """Validate one required non-empty string field."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _validate_positive_int(*, field_name: str, value: object) -> int:
    """Validate one strictly positive integer field."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _validate_seed(*, field_name: str, value: object) -> int:
    """Validate a 32-bit request seed."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}].")
    if value < SEED32_MIN or value > SEED32_MAX:
        raise ValueError(f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}].")
    return int(value)


def _validate_choice(*, field_name: str, value: object, choices: set[str]) -> str:
    """Validate one case-insensitive enum-like string field."""

    normalized = _validate_non_empty_string(field_name=field_name, value=value).lower()
    if normalized not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {expected} (got {value!r}).")
    return normalized


def _require_request_field(data: dict[str, Any], field_name: str) -> Any:
    """Return one required request field or raise a consistent error."""

    if field_name not in data:
        raise ValueError(f"Missing required request-file field: {field_name!r}.")
    return data[field_name]


def _validate_public_request_rows(value: object) -> DatasetRowsSpec:
    """Validate rows using only the documented public request-file encodings."""

    if isinstance(value, bool):
        raise ValueError(_REQUEST_ROWS_PUBLIC_SHAPE_ERROR)
    if isinstance(value, int | str):
        rows = normalize_dataset_rows(value)
        if rows is None:
            raise ValueError(_REQUEST_ROWS_PUBLIC_SHAPE_ERROR)
        return rows
    raise ValueError(_REQUEST_ROWS_PUBLIC_SHAPE_ERROR)


def _serialize_public_request_rows(rows: DatasetRowsSpec) -> int | str:
    """Serialize a normalized rows spec to the public request-file wire shape."""

    if rows.mode == "fixed":
        if rows.value is None:
            raise ValueError("rows fixed mode requires a value.")
        return int(rows.value)
    if rows.mode == "range":
        if rows.start is None or rows.stop is None:
            raise ValueError("rows range mode requires start/stop.")
        return f"{int(rows.start)}..{int(rows.stop)}"
    if rows.mode == "choices":
        if not rows.choices:
            raise ValueError("rows choices mode requires at least one choice.")
        return ",".join(str(int(choice)) for choice in rows.choices)
    raise ValueError(f"Unsupported request rows mode {rows.mode!r}.")


@dataclass(slots=True)
class RequestFileConfig:
    """Small public request-file schema for downstream corpus requests."""

    version: str
    task: str
    dataset_count: int
    rows: object
    profile: str
    output_root: str
    missingness_profile: str = MISSINGNESS_MECHANISM_NONE
    seed: int | None = None

    def __post_init__(self) -> None:
        self.version = _validate_choice(
            field_name="version",
            value=self.version,
            choices={REQUEST_FILE_VERSION_V1},
        )
        self.task = _validate_choice(
            field_name="task",
            value=self.task,
            choices=_REQUEST_TASKS,
        )
        self.dataset_count = _validate_positive_int(
            field_name="dataset_count",
            value=self.dataset_count,
        )
        self.rows = _validate_public_request_rows(self.rows)
        self.profile = _validate_choice(
            field_name="profile",
            value=self.profile,
            choices=_REQUEST_PROFILES,
        )
        self.output_root = _validate_non_empty_string(
            field_name="output_root",
            value=self.output_root,
        )
        self.missingness_profile = _validate_choice(
            field_name="missingness_profile",
            value=self.missingness_profile,
            choices=_REQUEST_MISSINGNESS_PROFILES,
        )
        if self.seed is not None:
            self.seed = _validate_seed(field_name="seed", value=self.seed)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RequestFileConfig:
        """Construct a request-file config from a mapping payload."""

        data = data or {}
        unknown_keys = sorted(set(data) - _REQUEST_ALLOWED_KEYS)
        if unknown_keys:
            first_key = unknown_keys[0]
            if first_key in _REQUEST_FORBIDDEN_KEYS:
                raise ValueError(
                    f"{first_key} is not part of the public request-file contract. "
                    "Use the minimal request fields instead of raw internal config sections."
                )
            raise ValueError(f"Unknown request-file field: {first_key!r}.")
        forbidden_present = sorted(set(data).intersection(_REQUEST_FORBIDDEN_KEYS))
        if forbidden_present:
            first_key = forbidden_present[0]
            raise ValueError(
                f"{first_key} is not part of the public request-file contract. "
                "Use the minimal request fields instead of raw internal config sections."
            )

        return cls(
            version=cast(str, _require_request_field(data, "version")),
            task=cast(str, _require_request_field(data, "task")),
            dataset_count=cast(int, _require_request_field(data, "dataset_count")),
            rows=_require_request_field(data, "rows"),
            profile=cast(str, _require_request_field(data, "profile")),
            output_root=cast(str, _require_request_field(data, "output_root")),
            missingness_profile=data.get("missingness_profile", MISSINGNESS_MECHANISM_NONE),
            seed=data.get("seed"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RequestFileConfig:
        """Load a request-file config from a YAML file path."""

        resolved_path = Path(path)
        with resolved_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Request file at {resolved_path} must be a mapping at the top level.")
        return cls.from_dict(loaded)

    def to_dict(self) -> dict[str, Any]:
        """Serialize request config to a plain nested dictionary."""

        rows = self.rows
        if not isinstance(rows, DatasetRowsSpec):
            raise ValueError("rows must be normalized before serialization.")

        data: dict[str, Any] = {
            "version": self.version,
            "task": self.task,
            "dataset_count": self.dataset_count,
            "rows": _serialize_public_request_rows(rows),
            "profile": self.profile,
            "output_root": self.output_root,
            "missingness_profile": self.missingness_profile,
        }
        if self.seed is not None:
            data["seed"] = self.seed
        return data
