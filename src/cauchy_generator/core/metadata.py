"""Metadata builders for dataset lineage and curriculum diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch

from cauchy_generator.core.curriculum import _CURRICULUM_MONOTONICITY_AXES
from cauchy_generator.core.shift import ShiftRuntimeParams, mechanism_nonlinear_mass
from cauchy_generator.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    validate_lineage_payload,
)


def _build_lineage_metadata(
    layout: dict[str, Any],
    *,
    feature_index_map: list[int],
) -> dict[str, Any]:
    """Build a validated DAG lineage payload from sampled layout internals."""

    n_nodes = int(layout["graph_nodes"])
    raw_adjacency = layout["adjacency"]
    if isinstance(raw_adjacency, torch.Tensor):
        adjacency_rows = raw_adjacency.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_rows = torch.as_tensor(raw_adjacency, dtype=torch.int64, device="cpu").tolist()
    adjacency = [[int(value) for value in row] for row in adjacency_rows]

    raw_feature_to_node = [
        int(node_index) for node_index in list(layout["feature_node_assignment"])
    ]
    feature_to_node = [raw_feature_to_node[int(src_col)] for src_col in feature_index_map]
    target_to_node = int(layout["target_node_assignment"])

    payload = {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": n_nodes,
            "adjacency": adjacency,
        },
        "assignments": {
            "feature_to_node": feature_to_node,
            "target_to_node": target_to_node,
        },
    }
    validate_lineage_payload(payload)
    return payload


def _build_curriculum_metadata(
    curriculum: dict[str, Any],
    *,
    layout: dict[str, Any],
    n_train: int,
    n_test: int,
    n_features: int,
) -> dict[str, Any]:
    """Attach stage complexity diagnostics to emitted curriculum metadata."""

    payload = dict(curriculum)
    stage_bounds_payload: dict[str, int | None] = {
        "n_features_min": None,
        "n_features_max": None,
        "n_nodes_min": None,
        "n_nodes_max": None,
        "depth_min": None,
        "depth_max": None,
    }
    raw_stage_bounds = layout.get("stage_bounds")
    if isinstance(raw_stage_bounds, dict):
        for key in stage_bounds_payload:
            raw_value = raw_stage_bounds.get(key)
            stage_bounds_payload[key] = int(raw_value) if raw_value is not None else None

    payload["realized_complexity"] = {
        "n_rows_total": int(n_train + n_test),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "graph_nodes": int(layout["graph_nodes"]),
        "graph_depth_nodes": int(layout["graph_depth_nodes"]),
        "graph_edge_density": float(layout["graph_edge_density"]),
    }
    payload["stage_bounds"] = stage_bounds_payload
    payload["monotonicity_axes"] = list(_CURRICULUM_MONOTONICITY_AXES)
    return payload


def _build_shift_metadata(*, shift_params: ShiftRuntimeParams) -> dict[str, Any]:
    """Build resolved shift metadata for one emitted dataset bundle."""

    nonlinear_mass = mechanism_nonlinear_mass(
        mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt)
    )
    edge_odds_multiplier = float(math.exp(shift_params.edge_logit_bias_shift))
    noise_variance_multiplier = float(shift_params.noise_sigma_multiplier**2)
    return {
        "enabled": bool(shift_params.enabled),
        "profile": str(shift_params.profile),
        "graph_scale": float(shift_params.graph_scale),
        "mechanism_scale": float(shift_params.mechanism_scale),
        "noise_scale": float(shift_params.noise_scale),
        "edge_logit_bias_shift": float(shift_params.edge_logit_bias_shift),
        "mechanism_logit_tilt": float(shift_params.mechanism_logit_tilt),
        "noise_sigma_multiplier": float(shift_params.noise_sigma_multiplier),
        "edge_odds_multiplier": float(edge_odds_multiplier),
        "noise_variance_multiplier": float(noise_variance_multiplier),
        "mechanism_nonlinear_mass": float(nonlinear_mass),
    }
