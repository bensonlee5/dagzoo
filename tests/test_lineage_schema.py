from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from cauchy_generator.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    LineageValidationError,
    validate_lineage_payload,
    validate_metadata_lineage,
)


def _valid_lineage_payload() -> dict[str, Any]:
    return {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": 4,
            "adjacency": [
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        },
        "assignments": {
            "feature_to_node": [0, 1, 1, 2, 3],
            "target_to_node": 2,
        },
    }


def test_validate_lineage_payload_accepts_valid_payload() -> None:
    validate_lineage_payload(_valid_lineage_payload())


def test_validate_metadata_lineage_accepts_absent_payload_when_optional() -> None:
    validate_metadata_lineage({"seed": 1}, required=False)


def test_validate_metadata_lineage_rejects_missing_payload_when_required() -> None:
    with pytest.raises(LineageValidationError, match=r"metadata\.lineage: is required"):
        validate_metadata_lineage({"seed": 1}, required=True)


def test_validate_lineage_payload_rejects_wrong_schema_name() -> None:
    payload = _valid_lineage_payload()
    payload["schema_name"] = "other.schema"
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.schema_name: must equal 'cauchy_generator\.dag_lineage'",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_unknown_root_keys() -> None:
    payload = _valid_lineage_payload()
    payload["extra"] = True
    with pytest.raises(LineageValidationError, match=r"lineage: unknown key\(s\): extra"):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_non_square_adjacency() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    graph["adjacency"] = [[0, 1], [0, 0]]
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency: must have 4 rows \(got 2\)",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_non_binary_adjacency_values() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[0][1] = 2
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError, match=r"lineage\.graph\.adjacency\[0\]\[1\]: must be 0 or 1"
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_bool_adjacency_values() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[0][1] = True
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency\[0\]\[1\]: must be integer 0 or 1",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_diagonal_edges() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[2][2] = 1
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency\[2\]\[2\]: must be 0 on the diagonal",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_lower_triangle_edges() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[3][1] = 1
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency\[3\]\[1\]: must be 0 for upper-triangular DAG encoding",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_feature_assignment_out_of_range() -> None:
    payload = _valid_lineage_payload()
    assignments = deepcopy(payload["assignments"])
    assert isinstance(assignments, dict)
    assignments["feature_to_node"] = [0, 1, 4]
    payload["assignments"] = assignments
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.assignments\.feature_to_node\[2\]: must be in range \[0, 3\]",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_target_assignment_out_of_range() -> None:
    payload = _valid_lineage_payload()
    assignments = deepcopy(payload["assignments"])
    assert isinstance(assignments, dict)
    assignments["target_to_node"] = -1
    payload["assignments"] = assignments
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.assignments\.target_to_node: must be in range \[0, 3\]",
    ):
        validate_lineage_payload(payload)


def test_validate_metadata_lineage_rejects_non_object_payload() -> None:
    with pytest.raises(LineageValidationError, match=r"metadata\.lineage: must be an object"):
        validate_metadata_lineage({"lineage": 42}, required=False)
