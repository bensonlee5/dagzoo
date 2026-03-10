"""Fixed-layout plan models and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import torch

from dagzoo.core.fixed_layout_plan_types import FixedLayoutExecutionPlan
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.types import DatasetBundle

_FIXED_LAYOUT_METADATA_SCHEMA_VERSION = 4


@dataclass(slots=True)
class _FixedLayoutPlan:
    """Internal pre-sampled layout bundle for canonical fixed-layout generation."""

    layout: LayoutPlan
    requested_device: str
    resolved_device: str
    plan_seed: int
    n_train: int
    n_test: int
    layout_signature: str
    candidate_attempt: int = 0
    execution_plan: FixedLayoutExecutionPlan = field(default_factory=FixedLayoutExecutionPlan)
    plan_signature: str | None = None


def _layout_to_dict(layout: LayoutPlan) -> dict[str, Any]:
    adjacency = layout.adjacency
    if isinstance(adjacency, torch.Tensor):
        adjacency_payload = adjacency.to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_payload = torch.as_tensor(adjacency, dtype=torch.int64, device="cpu").tolist()
    return {
        "n_features": int(layout.n_features),
        "n_cat": int(layout.n_cat),
        "cat_idx": [int(value) for value in layout.cat_idx],
        "cardinalities": [int(value) for value in layout.cardinalities],
        "card_by_feature": {
            str(int(key)): int(value) for key, value in layout.card_by_feature.items()
        },
        "n_classes": int(layout.n_classes),
        "feature_types": [str(value) for value in layout.feature_types],
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "graph_edge_density": float(layout.graph_edge_density),
        "adjacency": adjacency_payload,
        "feature_node_assignment": [int(value) for value in layout.feature_node_assignment],
        "target_node_assignment": int(layout.target_node_assignment),
    }


def _layout_signature(layout: LayoutPlan) -> str:
    encoded = json.dumps(
        _layout_to_dict(layout),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _annotate_fixed_layout_metadata(bundle: DatasetBundle, *, plan: _FixedLayoutPlan) -> None:
    bundle.metadata["layout_mode"] = "fixed"
    bundle.metadata["layout_plan_seed"] = int(plan.plan_seed)
    bundle.metadata["layout_signature"] = str(plan.layout_signature)
    bundle.metadata["layout_plan_schema_version"] = int(_FIXED_LAYOUT_METADATA_SCHEMA_VERSION)
    bundle.metadata["layout_execution_contract"] = str(plan.execution_plan.execution_contract)
    keyed_replay = bundle.metadata.get("keyed_replay")
    if not isinstance(keyed_replay, dict):
        keyed_replay = {}
    keyed_replay["layout_root_path"] = ["plan_candidate", int(plan.candidate_attempt), "layout"]
    keyed_replay["execution_plan_root_path"] = [
        "plan_candidate",
        int(plan.candidate_attempt),
        "execution_plan",
    ]
    bundle.metadata["keyed_replay"] = keyed_replay
    if plan.plan_signature is not None:
        bundle.metadata["layout_plan_signature"] = str(plan.plan_signature)


def _extract_emitted_schema_signature(
    bundle: DatasetBundle,
) -> tuple[int, tuple[str, ...], tuple[int, ...]]:
    n_features = int(bundle.metadata.get("n_features", int(bundle.X_train.shape[1])))
    feature_types = tuple(str(t) for t in bundle.feature_types)
    if len(feature_types) != n_features:
        raise ValueError(
            "Fixed-layout bundle emitted inconsistent feature schema metadata: "
            f"n_features={n_features}, feature_types_len={len(feature_types)}."
        )

    lineage = bundle.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Fixed-layout bundle is missing lineage metadata.")
    assignments = lineage.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Fixed-layout bundle is missing lineage assignments metadata.")
    raw_feature_to_node = assignments.get("feature_to_node")
    if not isinstance(raw_feature_to_node, list):
        raise ValueError("Fixed-layout bundle is missing lineage assignments.feature_to_node.")
    feature_to_node = tuple(int(value) for value in raw_feature_to_node)
    if len(feature_to_node) != n_features:
        raise ValueError(
            "Fixed-layout bundle emitted inconsistent lineage feature mapping: "
            f"n_features={n_features}, feature_to_node_len={len(feature_to_node)}."
        )

    return n_features, feature_types, feature_to_node


__all__ = [
    "_FixedLayoutPlan",
    "_annotate_fixed_layout_metadata",
    "_extract_emitted_schema_signature",
    "_layout_signature",
]
