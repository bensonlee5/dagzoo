"""Validation helpers for DAG lineage metadata schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn, TypeGuard, cast

LINEAGE_SCHEMA_NAME = "cauchy_generator.dag_lineage"
LINEAGE_SCHEMA_VERSION = "1.0.0"

_LINEAGE_REQUIRED_KEYS = frozenset({"schema_name", "schema_version", "graph", "assignments"})
_GRAPH_REQUIRED_KEYS = frozenset({"n_nodes", "adjacency"})
_ASSIGNMENTS_REQUIRED_KEYS = frozenset({"feature_to_node", "target_to_node"})


class LineageValidationError(ValueError):
    """Raised when a lineage payload fails schema validation."""


def _is_int(value: object) -> TypeGuard[int]:
    """Return True for int values excluding bool."""

    return isinstance(value, int) and not isinstance(value, bool)


def _raise(path: str, message: str) -> NoReturn:
    """Raise a schema validation error with a field path."""

    raise LineageValidationError(f"{path}: {message}")


def _as_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    """Validate and return a mapping payload."""

    if not isinstance(value, Mapping):
        _raise(path, "must be an object")
    return cast(Mapping[str, Any], value)


def _validate_required_and_unknown_keys(
    payload: Mapping[str, Any],
    *,
    path: str,
    required_keys: frozenset[str],
) -> None:
    """Validate required keys are present and reject unknown keys."""

    raw_keys = list(payload.keys())
    non_string_keys = [key for key in raw_keys if not isinstance(key, str)]
    if non_string_keys:
        formatted = ", ".join(repr(key) for key in non_string_keys)
        _raise(path, f"contains non-string key(s): {formatted}")

    keys = {str(key) for key in raw_keys}
    missing = sorted(required_keys - keys)
    if missing:
        _raise(path, f"missing required key(s): {', '.join(missing)}")

    unknown = sorted(keys - required_keys)
    if unknown:
        _raise(path, f"unknown key(s): {', '.join(unknown)}")


def _validate_assignment_node_index(value: object, *, n_nodes: int, path: str) -> None:
    """Validate one assignment node index with bound checks."""

    if not _is_int(value):
        _raise(path, "must be an integer node index")
    if value < 0 or value >= n_nodes:
        _raise(path, f"must be in range [0, {n_nodes - 1}]")


def _validate_lineage_payload(payload: object, *, root_path: str) -> None:
    """Validate the DAG lineage payload rooted at a given path."""

    lineage = _as_mapping(payload, path=root_path)
    _validate_required_and_unknown_keys(
        lineage,
        path=root_path,
        required_keys=_LINEAGE_REQUIRED_KEYS,
    )

    schema_name = lineage["schema_name"]
    schema_name_path = f"{root_path}.schema_name"
    if not isinstance(schema_name, str):
        _raise(schema_name_path, "must be a string")
    if schema_name != LINEAGE_SCHEMA_NAME:
        _raise(
            schema_name_path,
            f"must equal '{LINEAGE_SCHEMA_NAME}' (got '{schema_name}')",
        )

    schema_version = lineage["schema_version"]
    schema_version_path = f"{root_path}.schema_version"
    if not isinstance(schema_version, str):
        _raise(schema_version_path, "must be a string")
    if schema_version != LINEAGE_SCHEMA_VERSION:
        _raise(
            schema_version_path,
            f"must equal '{LINEAGE_SCHEMA_VERSION}' (got '{schema_version}')",
        )

    graph_path = f"{root_path}.graph"
    graph = _as_mapping(lineage["graph"], path=graph_path)
    _validate_required_and_unknown_keys(
        graph,
        path=graph_path,
        required_keys=_GRAPH_REQUIRED_KEYS,
    )

    n_nodes = graph["n_nodes"]
    n_nodes_path = f"{graph_path}.n_nodes"
    if not _is_int(n_nodes):
        _raise(n_nodes_path, "must be an integer >= 2")
    if n_nodes < 2:
        _raise(n_nodes_path, "must be >= 2")

    adjacency = graph["adjacency"]
    adjacency_path = f"{graph_path}.adjacency"
    if not isinstance(adjacency, list):
        _raise(adjacency_path, "must be a list of rows")
    if len(adjacency) != n_nodes:
        _raise(adjacency_path, f"must have {n_nodes} rows (got {len(adjacency)})")

    for row_idx, row in enumerate(adjacency):
        row_path = f"{adjacency_path}[{row_idx}]"
        if not isinstance(row, list):
            _raise(row_path, "must be a list")
        if len(row) != n_nodes:
            _raise(row_path, f"must have {n_nodes} columns (got {len(row)})")
        for col_idx, value in enumerate(row):
            value_path = f"{row_path}[{col_idx}]"
            if not _is_int(value):
                _raise(value_path, "must be integer 0 or 1")
            if value not in (0, 1):
                _raise(value_path, "must be 0 or 1")
            if row_idx == col_idx and value != 0:
                _raise(value_path, "must be 0 on the diagonal")
            if row_idx > col_idx and value != 0:
                _raise(value_path, "must be 0 for upper-triangular DAG encoding")

    assignments_path = f"{root_path}.assignments"
    assignments = _as_mapping(lineage["assignments"], path=assignments_path)
    _validate_required_and_unknown_keys(
        assignments,
        path=assignments_path,
        required_keys=_ASSIGNMENTS_REQUIRED_KEYS,
    )

    feature_to_node = assignments["feature_to_node"]
    feature_to_node_path = f"{assignments_path}.feature_to_node"
    if not isinstance(feature_to_node, list):
        _raise(feature_to_node_path, "must be a list of node indices")
    for idx, node_index in enumerate(feature_to_node):
        _validate_assignment_node_index(
            node_index,
            n_nodes=n_nodes,
            path=f"{feature_to_node_path}[{idx}]",
        )

    _validate_assignment_node_index(
        assignments["target_to_node"],
        n_nodes=n_nodes,
        path=f"{assignments_path}.target_to_node",
    )


def validate_lineage_payload(payload: Mapping[str, Any]) -> None:
    """Validate a lineage payload object rooted at `lineage`."""

    _validate_lineage_payload(payload, root_path="lineage")


def validate_metadata_lineage(metadata: Mapping[str, Any], *, required: bool = False) -> None:
    """Validate `metadata.lineage` when present.

    When `required` is False (default), missing lineage is accepted for backward compatibility.
    """

    metadata_payload = _as_mapping(metadata, path="metadata")
    if "lineage" not in metadata_payload:
        if required:
            _raise("metadata.lineage", "is required")
        return
    _validate_lineage_payload(metadata_payload["lineage"], root_path="metadata.lineage")


__all__ = [
    "LINEAGE_SCHEMA_NAME",
    "LINEAGE_SCHEMA_VERSION",
    "LineageValidationError",
    "validate_lineage_payload",
    "validate_metadata_lineage",
]
