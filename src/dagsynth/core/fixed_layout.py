"""Fixed-layout plan types and generation helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from dagsynth.config import GeneratorConfig
from dagsynth.core.generation_context import (
    _resolve_device,
    _resolve_run_seed,
    _resolve_split_sizes,
    _validate_class_split_for_layout,
)
from dagsynth.core.generation_engine import _generate_one_with_resolved_layout
from dagsynth.core.layout import _sample_layout
from dagsynth.core.layout_types import LayoutPlan
from dagsynth.core.shift import resolve_shift_runtime_params
from dagsynth.rng import SeedManager
from dagsynth.types import DatasetBundle


@dataclass(slots=True)
class FixedLayoutPlan:
    """Pre-sampled layout bundle for fixed-layout batch generation."""

    layout: LayoutPlan
    requested_device: str
    resolved_device: str
    plan_seed: int
    n_train: int
    n_test: int
    layout_signature: str
    compatibility_snapshot: dict[str, Any]


_FIXED_LAYOUT_COMPAT_KEYS: tuple[str, ...] = (
    "dataset.task",
    "dataset.n_train",
    "dataset.n_test",
    "dataset.n_features_min",
    "dataset.n_features_max",
    "dataset.categorical_ratio_min",
    "dataset.categorical_ratio_max",
    "dataset.max_categorical_cardinality",
    "dataset.n_classes_min",
    "dataset.n_classes_max",
    "graph.n_nodes_min",
    "graph.n_nodes_max",
    "shift.edge_logit_bias_shift",
    "runtime.resolved_device",
)


def _layout_signature(layout: LayoutPlan) -> str:
    """Return a deterministic, stable signature for a sampled layout payload."""

    adjacency = layout.adjacency
    adjacency_payload: list[list[int]]
    if isinstance(adjacency, torch.Tensor):
        adjacency_payload = adjacency.to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_payload = torch.as_tensor(adjacency, dtype=torch.int64, device="cpu").tolist()

    signature_payload = {
        "n_features": int(layout.n_features),
        "n_classes": int(layout.n_classes),
        "feature_types": list(layout.feature_types),
        "card_by_feature": {
            str(int(k)): int(v) for k, v in sorted(dict(layout.card_by_feature).items())
        },
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "feature_node_assignment": [int(v) for v in list(layout.feature_node_assignment)],
        "target_node_assignment": int(layout.target_node_assignment),
        "adjacency": adjacency_payload,
    }
    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _build_fixed_layout_compatibility_snapshot(
    config: GeneratorConfig,
    *,
    resolved_device: str,
) -> dict[str, Any]:
    """Build a fixed-layout compatibility snapshot from effective generation inputs."""

    shift_params = resolve_shift_runtime_params(config)
    return {
        "dataset.task": str(config.dataset.task),
        "dataset.n_train": int(config.dataset.n_train),
        "dataset.n_test": int(config.dataset.n_test),
        "dataset.n_features_min": int(config.dataset.n_features_min),
        "dataset.n_features_max": int(config.dataset.n_features_max),
        "dataset.categorical_ratio_min": float(config.dataset.categorical_ratio_min),
        "dataset.categorical_ratio_max": float(config.dataset.categorical_ratio_max),
        "dataset.max_categorical_cardinality": int(config.dataset.max_categorical_cardinality),
        "dataset.n_classes_min": int(config.dataset.n_classes_min),
        "dataset.n_classes_max": int(config.dataset.n_classes_max),
        "graph.n_nodes_min": int(config.graph.n_nodes_min),
        "graph.n_nodes_max": int(config.graph.n_nodes_max),
        "shift.edge_logit_bias_shift": float(shift_params.edge_logit_bias_shift),
        "runtime.resolved_device": str(resolved_device),
    }


def sample_fixed_layout(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> FixedLayoutPlan:
    """Sample one reusable layout plan for fixed-layout batch generation."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    manager = SeedManager(run_seed)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu")
    n_train, n_test = _resolve_split_sizes(config)
    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    return FixedLayoutPlan(
        layout=layout,
        requested_device=requested_device,
        resolved_device=resolved_device,
        plan_seed=int(run_seed),
        n_train=int(n_train),
        n_test=int(n_test),
        layout_signature=_layout_signature(layout),
        compatibility_snapshot=_build_fixed_layout_compatibility_snapshot(
            config,
            resolved_device=resolved_device,
        ),
    )


def _annotate_fixed_layout_metadata(bundle: DatasetBundle, *, plan: FixedLayoutPlan) -> None:
    """Attach fixed-layout provenance metadata to an emitted bundle."""

    bundle.metadata["layout_mode"] = "fixed"
    bundle.metadata["layout_plan_seed"] = int(plan.plan_seed)
    bundle.metadata["layout_signature"] = str(plan.layout_signature)


def _extract_emitted_schema_signature(
    bundle: DatasetBundle,
) -> tuple[int, tuple[str, ...], tuple[int, ...]]:
    """Extract the emitted schema signature for fixed-layout contract checks."""

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


def _validate_fixed_layout_plan_compatibility(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
) -> str:
    """Validate that a fixed-layout plan is compatible with the active config."""

    snapshot = plan.compatibility_snapshot
    if not isinstance(snapshot, dict):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: compatibility_snapshot must be a mapping."
        )

    computed_layout_signature = _layout_signature(plan.layout)
    if str(plan.layout_signature) != computed_layout_signature:
        raise ValueError(
            "Fixed-layout plan integrity mismatch: layout_signature does not match plan.layout."
        )

    missing_keys = [key for key in _FIXED_LAYOUT_COMPAT_KEYS if key not in snapshot]
    if missing_keys:
        raise ValueError(
            "Fixed-layout plan integrity mismatch: compatibility snapshot is missing keys: "
            f"{', '.join(missing_keys)}."
        )

    if int(plan.n_train) != int(snapshot["dataset.n_train"]):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan.n_train does not match "
            "compatibility snapshot."
        )
    if int(plan.n_test) != int(snapshot["dataset.n_test"]):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan.n_test does not match "
            "compatibility snapshot."
        )
    if str(plan.resolved_device) != str(snapshot["runtime.resolved_device"]):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan.resolved_device does not match "
            "compatibility snapshot."
        )

    try:
        resolved_device = _resolve_device(config, plan.requested_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "Fixed-layout plan/config mismatch: unable to resolve the plan-requested "
            f"device '{plan.requested_device}' for the current environment."
        ) from exc
    if str(plan.resolved_device) != str(resolved_device):
        raise ValueError(
            "Fixed-layout plan/config mismatch: plan.resolved_device does not match "
            f"the currently resolved backend ({plan.resolved_device!r} != {resolved_device!r})."
        )

    current_snapshot = _build_fixed_layout_compatibility_snapshot(
        config,
        resolved_device=resolved_device,
    )
    mismatches: list[str] = []
    for key in _FIXED_LAYOUT_COMPAT_KEYS:
        plan_value = snapshot[key]
        config_value = current_snapshot[key]
        if plan_value != config_value:
            mismatches.append(f"{key} (plan={plan_value!r}, config={config_value!r})")
    if mismatches:
        raise ValueError(
            "Fixed-layout plan/config mismatch for compatibility fields: " + "; ".join(mismatches)
        )
    return str(resolved_device)


def generate_batch_fixed_layout_iter(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets that share one pre-sampled fixed layout and split shape."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    validated_resolved_device = _validate_fixed_layout_plan_compatibility(config, plan=plan)
    run_seed = _resolve_run_seed(config, seed)
    manager = SeedManager(run_seed)
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...]] | None = None
    for i in range(num_datasets):
        dataset_seed = manager.child("dataset", i)
        bundle = _generate_one_with_resolved_layout(
            config,
            seed=dataset_seed,
            requested_device=plan.requested_device,
            resolved_device=validated_resolved_device,
            n_train=int(plan.n_train),
            n_test=int(plan.n_test),
            layout=plan.layout,
            preserve_feature_schema=True,
        )
        _annotate_fixed_layout_metadata(bundle, plan=plan)
        schema = _extract_emitted_schema_signature(bundle)
        if expected_schema is None:
            expected_schema = schema
        elif schema != expected_schema:
            raise ValueError(
                "Fixed-layout schema mismatch: emitted dataset does not match "
                "the first fixed-layout bundle schema."
            )
        yield bundle


def generate_batch_fixed_layout(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
) -> list[DatasetBundle]:
    """Generate a materialized fixed-layout batch using a reusable plan."""

    return list(
        generate_batch_fixed_layout_iter(
            config,
            plan=plan,
            num_datasets=num_datasets,
            seed=seed,
        )
    )
