"""Dataset row-spec models and helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from dagzoo.rng import derive_seed

from .constants import (
    DATASET_ROWS_MAX_TOTAL,
    DATASET_ROWS_MIN_TOTAL,
    RowsMode,
)
from .scalars import _validate_int_field


@dataclass(slots=True)
class DatasetRowsSpec:
    """Normalized dataset total-row sampling spec."""

    mode: RowsMode
    value: int | None = None
    start: int | None = None
    stop: int | None = None
    choices: list[int] = field(default_factory=list)


def _validate_rows_total(
    *,
    field_name: str,
    value: object,
) -> int:
    """Validate one dataset total-rows value against supported bounds."""

    return _validate_int_field(
        field_name=field_name,
        value=value,
        minimum=DATASET_ROWS_MIN_TOTAL,
        maximum=DATASET_ROWS_MAX_TOTAL,
    )


def _normalize_rows_choices(
    *,
    field_name: str,
    value: object,
) -> list[int]:
    """Normalize choice-based rows specs from list/csv input."""

    raw_items: list[object]
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(
                f"{field_name} must be a non-empty CSV list, range, or integer row count."
            )
        raw_items = [item.strip() for item in normalized.split(",")]
    elif isinstance(value, list):
        raw_items = list(value)
    else:
        raise ValueError(
            f"{field_name} choices must be provided as a CSV string or list of integers."
        )

    if not raw_items:
        raise ValueError(f"{field_name} must include at least one row count.")

    choices: list[int] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, str) and not item.strip():
            raise ValueError(f"{field_name} contains an empty row token at position {idx}.")
        choices.append(_validate_rows_total(field_name=f"{field_name}[{idx}]", value=item))
    if len(set(choices)) != len(choices):
        raise ValueError(f"{field_name} must not include duplicate row values.")
    return choices


def _normalize_dataset_rows_from_mapping(value: dict[object, object]) -> DatasetRowsSpec:
    """Normalize canonical mapping rows spec shape."""

    mode_raw = value.get("mode")
    if isinstance(mode_raw, bool) or not isinstance(mode_raw, str):
        raise ValueError(
            "dataset.rows mapping must include string key 'mode' "
            "with one of: fixed, range, choices."
        )
    mode = mode_raw.strip().lower()
    if mode == "fixed":
        if "value" not in value:
            raise ValueError("dataset.rows fixed mapping must include key 'value'.")
        fixed = _validate_rows_total(field_name="dataset.rows.value", value=value["value"])
        return DatasetRowsSpec(mode="fixed", value=fixed)

    if mode == "range":
        if "start" not in value or "stop" not in value:
            raise ValueError("dataset.rows range mapping must include 'start' and 'stop'.")
        start = _validate_rows_total(field_name="dataset.rows.start", value=value["start"])
        stop = _validate_rows_total(field_name="dataset.rows.stop", value=value["stop"])
        if start > stop:
            raise ValueError(
                f"dataset.rows range start must be <= stop, got start={start} stop={stop}."
            )
        if start == stop:
            return DatasetRowsSpec(mode="fixed", value=start)
        return DatasetRowsSpec(mode="range", start=start, stop=stop)

    if mode == "choices":
        raw_choices = value.get("choices", value.get("values"))
        choices = _normalize_rows_choices(field_name="dataset.rows.choices", value=raw_choices)
        if len(choices) == 1:
            return DatasetRowsSpec(mode="fixed", value=choices[0])
        return DatasetRowsSpec(mode="choices", choices=choices)

    raise ValueError(
        f"dataset.rows mapping mode must be one of: fixed, range, choices (got {mode_raw!r})."
    )


def normalize_dataset_rows(value: object | None) -> DatasetRowsSpec | None:
    """Normalize dataset.rows into a validated internal row-spec representation."""

    if value is None:
        return None
    if isinstance(value, DatasetRowsSpec):
        if value.mode == "fixed":
            if value.value is None:
                raise ValueError("dataset.rows fixed mode requires a value.")
            return DatasetRowsSpec(
                mode="fixed",
                value=_validate_rows_total(field_name="dataset.rows", value=value.value),
            )
        if value.mode == "range":
            if value.start is None or value.stop is None:
                raise ValueError("dataset.rows range mode requires start/stop.")
            start = _validate_rows_total(field_name="dataset.rows.start", value=value.start)
            stop = _validate_rows_total(field_name="dataset.rows.stop", value=value.stop)
            if start > stop:
                raise ValueError(
                    f"dataset.rows range start must be <= stop, got start={start} stop={stop}."
                )
            if start == stop:
                return DatasetRowsSpec(mode="fixed", value=start)
            return DatasetRowsSpec(mode="range", start=start, stop=stop)
        if value.mode == "choices":
            choices = _normalize_rows_choices(
                field_name="dataset.rows.choices", value=value.choices
            )
            if len(choices) == 1:
                return DatasetRowsSpec(mode="fixed", value=choices[0])
            return DatasetRowsSpec(mode="choices", choices=choices)
        raise ValueError(
            f"dataset.rows mode must be fixed, range, or choices (got {value.mode!r})."
        )

    if isinstance(value, bool):
        raise ValueError(
            "dataset.rows must be an integer, range string, CSV string, list, or null."
        )
    if isinstance(value, int):
        return DatasetRowsSpec(
            mode="fixed", value=_validate_rows_total(field_name="dataset.rows", value=value)
        )
    if isinstance(value, dict):
        return _normalize_dataset_rows_from_mapping(value)
    if isinstance(value, list):
        choices = _normalize_rows_choices(field_name="dataset.rows", value=value)
        if len(choices) == 1:
            return DatasetRowsSpec(mode="fixed", value=choices[0])
        return DatasetRowsSpec(mode="choices", choices=choices)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("dataset.rows must be a non-empty integer, range string, or CSV list.")
        if ".." in normalized:
            parts = normalized.split("..")
            if len(parts) != 2:
                raise ValueError(
                    "dataset.rows range must use 'start..stop' with one '..' delimiter."
                )
            start = _validate_rows_total(field_name="dataset.rows.start", value=parts[0].strip())
            stop = _validate_rows_total(field_name="dataset.rows.stop", value=parts[1].strip())
            if start > stop:
                raise ValueError(
                    f"dataset.rows range start must be <= stop, got start={start} stop={stop}."
                )
            if start == stop:
                return DatasetRowsSpec(mode="fixed", value=start)
            return DatasetRowsSpec(mode="range", start=start, stop=stop)
        if "," in normalized:
            choices = _normalize_rows_choices(field_name="dataset.rows", value=normalized)
            if len(choices) == 1:
                return DatasetRowsSpec(mode="fixed", value=choices[0])
            return DatasetRowsSpec(mode="choices", choices=choices)
        return DatasetRowsSpec(
            mode="fixed",
            value=_validate_rows_total(field_name="dataset.rows", value=normalized),
        )
    raise ValueError(
        "dataset.rows must be an integer, range string, CSV string, list, mapping, or null."
    )


def dataset_rows_bounds(rows: object | None) -> tuple[int, int] | None:
    """Return min/max total rows for a normalized rows spec."""

    normalized_rows = normalize_dataset_rows(rows)
    if normalized_rows is None:
        return None
    if normalized_rows.mode == "fixed":
        assert normalized_rows.value is not None
        return int(normalized_rows.value), int(normalized_rows.value)
    if normalized_rows.mode == "range":
        assert normalized_rows.start is not None and normalized_rows.stop is not None
        return int(normalized_rows.start), int(normalized_rows.stop)
    if normalized_rows.mode == "choices":
        if not normalized_rows.choices:
            raise ValueError("dataset.rows choices mode requires at least one choice.")
        return min(int(v) for v in normalized_rows.choices), max(
            int(v) for v in normalized_rows.choices
        )
    raise ValueError(f"Unsupported dataset.rows mode {normalized_rows.mode!r}.")


def dataset_rows_is_variable(rows: object | None) -> bool:
    """Return whether rows spec varies per dataset."""

    normalized_rows = normalize_dataset_rows(rows)
    return normalized_rows is not None and normalized_rows.mode in {"range", "choices"}


def resolve_dataset_total_rows(
    rows: object | None,
    *,
    dataset_seed: int | None,
) -> int | None:
    """Resolve one total-rows value from a normalized rows spec."""

    normalized_rows = normalize_dataset_rows(rows)
    if normalized_rows is None:
        return None
    if normalized_rows.mode == "fixed":
        assert normalized_rows.value is not None
        return int(normalized_rows.value)
    if dataset_seed is None:
        raise ValueError(
            "Variable dataset.rows modes require dataset seed context to resolve total rows."
        )
    selector_seed = derive_seed(int(dataset_seed), "rows")
    rng = random.Random(selector_seed)
    if normalized_rows.mode == "range":
        assert normalized_rows.start is not None and normalized_rows.stop is not None
        return int(rng.randint(int(normalized_rows.start), int(normalized_rows.stop)))
    if normalized_rows.mode == "choices":
        if not normalized_rows.choices:
            raise ValueError("dataset.rows choices mode requires at least one choice.")
        idx = int(rng.randrange(0, len(normalized_rows.choices)))
        return int(normalized_rows.choices[idx])
    raise ValueError(f"Unsupported dataset.rows mode {normalized_rows.mode!r}.")


def validate_class_split_feasibility(
    *,
    n_classes: int,
    n_train: int,
    n_test: int,
    context: str,
) -> None:
    """Validate whether split sizes can represent all classes in both train and test."""

    if n_classes > n_train or n_classes > n_test:
        raise ValueError(
            f"{context}: infeasible class/split combination for classification "
            f"(n_classes={n_classes}, n_train={n_train}, n_test={n_test}). "
            "Require n_train and n_test to each be >= n_classes."
        )
