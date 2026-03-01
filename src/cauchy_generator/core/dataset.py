"""Seeded synthetic dataset generation with Torch execution on all devices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import torch

from cauchy_generator.config import (
    CURRICULUM_STAGE_AUTO,
    GeneratorConfig,
    normalize_curriculum_stage,
    validate_class_split_feasibility,
)
from cauchy_generator.core import curriculum as curriculum_core
from cauchy_generator.core.constants import (
    NODE_SPEC_SEED_OFFSET,
    SPLIT_PERMUTATION_SEED_OFFSET,
)
from cauchy_generator.core.layout import _build_node_specs, _sample_layout
from cauchy_generator.core.metadata import (
    _build_curriculum_metadata,
    _build_lineage_metadata,
    _build_shift_metadata,
)
from cauchy_generator.core.shift import ShiftRuntimeParams, resolve_shift_runtime_params
from cauchy_generator.core.validation import _classification_split_valid, _stratified_split_indices
from cauchy_generator.filtering import apply_torch_rf_filter
from cauchy_generator.core.node_pipeline import apply_node_pipeline
from cauchy_generator.postprocess import inject_missingness, postprocess_dataset
from cauchy_generator.rng import SeedManager
from cauchy_generator.types import DatasetBundle


@dataclass(slots=True)
class FixedLayoutPlan:
    """Pre-sampled curriculum/layout bundle for fixed-layout batch generation."""

    curriculum: dict[str, Any]
    layout: dict[str, Any]
    requested_device: str
    resolved_device: str
    plan_seed: int
    mode: str | int
    auto_stage: int
    layout_signature: str


def _resolve_device(config: GeneratorConfig, device_override: str | None) -> str:
    """Resolve runtime device and hard-fail on unavailable explicit accelerators."""

    requested = (device_override or config.runtime.device or "auto").lower()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
    if requested == "mps":
        if mps_ok:
            return "mps"
        raise RuntimeError("Requested device 'mps' but MPS is not available.")
    raise ValueError(f"Unsupported device '{requested}'. Expected one of: auto, cpu, cuda, mps.")


def _torch_dtype(config: GeneratorConfig) -> torch.dtype:
    """Map string runtime dtype configuration to a torch dtype."""

    return torch.float64 if config.runtime.torch_dtype == "float64" else torch.float32


def _classification_class_structure(
    *,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    n_classes_sampled: int,
) -> dict[str, Any]:
    """Build classification label-structure metadata for one emitted bundle."""

    y_train_i64 = y_train.to(torch.int64)
    y_test_i64 = y_test.to(torch.int64)
    y_all = torch.cat([y_train_i64, y_test_i64], dim=0)
    unique_all = torch.unique(y_all, sorted=True)
    n_classes_realized = int(unique_all.numel())
    labels_contiguous = bool(
        torch.equal(
            unique_all,
            torch.arange(n_classes_realized, dtype=unique_all.dtype, device=unique_all.device),
        )
    )
    train_classes = torch.unique(y_train_i64, sorted=True)
    test_classes = torch.unique(y_test_i64, sorted=True)

    return {
        "n_classes_sampled": int(n_classes_sampled),
        "n_classes_realized": int(n_classes_realized),
        "labels_contiguous": bool(labels_contiguous),
        "train_test_class_match": bool(torch.equal(train_classes, test_classes)),
        "min_label": int(unique_all[0].item()) if n_classes_realized > 0 else None,
        "max_label": int(unique_all[-1].item()) if n_classes_realized > 0 else None,
    }


def _generate_graph_dataset_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
    *,
    n_rows: int,
    mechanism_logit_tilt: float = 0.0,
    noise_sigma_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate raw X/y tensors via the Torch graph pipeline."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    n_features = int(layout["n_features"])
    task = config.dataset.task
    n_classes = int(layout["n_classes"])
    adjacency = layout["adjacency"]
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device=device)
    n_nodes = int(layout["graph_nodes"])

    node_outputs: dict[int, torch.Tensor] = {}
    feature_values: list[torch.Tensor | None] = [None] * n_features
    target_values: torch.Tensor | None = None

    for node_idx in range(n_nodes):
        parents = torch.where(adjacency[:, node_idx])[0].tolist()
        parent_data = [node_outputs[int(p)] for p in parents if int(p) in node_outputs]

        # Build specs using a deterministic per-node generator for layout consistency
        spec_gen = torch.Generator(device="cpu")
        spec_gen.manual_seed(seed + node_idx + NODE_SPEC_SEED_OFFSET)
        specs = _build_node_specs(node_idx, layout, task, spec_gen)

        x_node, extracted = apply_node_pipeline(
            parent_data,
            n_rows,
            specs,
            generator,
            device,
            mechanism_logit_tilt=mechanism_logit_tilt,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
        node_outputs[node_idx] = x_node

        for key, values in extracted.items():
            if key.startswith("feature_"):
                col = int(key.split("_", maxsplit=1)[1])
                feature_values[col] = values
            elif key == "target":
                target_values = values

    dtype = _torch_dtype(config)
    x = torch.zeros((n_rows, n_features), dtype=dtype, device=device)
    feature_types = list(layout["feature_types"])
    card_by_feature: dict[int, int] = layout["card_by_feature"]

    for i in range(n_features):
        v = feature_values[i]
        if v is None:
            if feature_types[i] == "cat":
                card = int(card_by_feature[i])
                v = torch.randint(0, card, (n_rows,), generator=generator, device=device)
            else:
                v = torch.randn(n_rows, generator=generator, device=device)
        x[:, i] = v.to(dtype)

    if target_values is None:
        if task == "classification":
            y = torch.randint(0, n_classes, (n_rows,), generator=generator, device=device)
        else:
            y = torch.randn(n_rows, generator=generator, device=device).to(dtype)
    else:
        if task == "classification":
            y = target_values.to(torch.long) % n_classes
        else:
            y = target_values.to(dtype)

    accepted, filter_details = _apply_filter_torch(config, x, y, seed=seed)
    return x, y, {"accepted": accepted, "filter": filter_details}


def _generate_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
    *,
    n_train: int,
    n_test: int,
    curriculum: dict[str, Any],
    shift_params: ShiftRuntimeParams | None = None,
) -> DatasetBundle:
    """Generate one dataset in Torch while preserving postprocess/filter contracts."""

    shift_params = shift_params or resolve_shift_runtime_params(config)
    attempts = max(1, int(config.filter.max_attempts))
    last_reason = "unknown"
    dtype = _torch_dtype(config)
    n_rows = n_train + n_test

    for attempt in range(attempts):
        try:
            x, y, aux_meta = _generate_graph_dataset_torch(
                config,
                layout,
                seed + attempt,
                device,
                n_rows=n_rows,
                mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
                noise_sigma_multiplier=float(shift_params.noise_sigma_multiplier),
            )
        except Exception as exc:
            last_reason = f"generation_exception:{exc.__class__.__name__}"
            continue
        if not bool(aux_meta.get("accepted", True)):
            last_reason = "filtered_out"
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(seed + SPLIT_PERMUTATION_SEED_OFFSET + attempt)

        if config.dataset.task == "classification":
            try:
                train_idx, test_idx = _stratified_split_indices(y, n_train, generator, device)
            except ValueError as exc:
                if str(exc).startswith("infeasible_stratified_split"):
                    last_reason = "invalid_class_split"
                    continue
                raise
            x_train_t, x_test_t = x[train_idx], x[test_idx]
            y_train_t, y_test_t = y[train_idx], y[test_idx]
        else:
            order = torch.randperm(x.shape[0], generator=generator, device=device)
            x, y = x[order], y[order]
            x_train_t, x_test_t = x[:n_train], x[n_train:]
            y_train_t, y_test_t = y[:n_train], y[n_train:]

        x_train, y_train, x_test, y_test, feature_types, feature_index_map = postprocess_dataset(
            x_train_t,
            y_train_t,
            x_test_t,
            y_test_t,
            list(layout["feature_types"]),
            config.dataset.task,
            generator,
            device,
            return_feature_index_map=True,
        )
        x_train, x_test, missingness_summary = inject_missingness(
            x_train,
            x_test,
            dataset_cfg=config.dataset,
            seed=seed,
            attempt=attempt,
            device=device,
        )

        if config.dataset.task == "classification" and not _classification_split_valid(
            y_train, y_test
        ):
            last_reason = "invalid_class_split"
            continue

        x_train = x_train.to(dtype)
        x_test = x_test.to(dtype)
        y_dtype = torch.int64 if config.dataset.task == "classification" else dtype
        y_train = y_train.to(y_dtype)
        y_test = y_test.to(y_dtype)
        class_structure: dict[str, Any] | None = None
        n_classes: int | None = None
        if config.dataset.task == "classification":
            class_structure = _classification_class_structure(
                y_train=y_train,
                y_test=y_test,
                n_classes_sampled=int(layout["n_classes"]),
            )
            n_classes = int(class_structure["n_classes_realized"])
        curriculum_metadata = _build_curriculum_metadata(
            curriculum,
            layout=layout,
            n_train=int(x_train.shape[0]),
            n_test=int(x_test.shape[0]),
            n_features=int(x_train.shape[1]),
        )
        shift_metadata = _build_shift_metadata(shift_params=shift_params)

        metadata = {
            "backend": "torch",
            "device": device,
            "compute_backend": "torch_appendix_full",
            "n_features": int(x_train.shape[1]),
            "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
            "n_classes": n_classes,
            "graph_nodes": int(layout["graph_nodes"]),
            "graph_edges": int(layout["graph_edges"]),
            "graph_depth_nodes": int(layout["graph_depth_nodes"]),
            "graph_edge_density": float(layout["graph_edge_density"]),
            "lineage": _build_lineage_metadata(layout, feature_index_map=feature_index_map),
            "seed": seed,
            "attempt_used": attempt,
            "filter": aux_meta.get("filter", {}),
            "curriculum": curriculum_metadata,
            "shift": shift_metadata,
            "config": asdict(config),
        }
        if missingness_summary is not None:
            metadata["missingness"] = missingness_summary
        if class_structure is not None:
            metadata["class_structure"] = class_structure
        return DatasetBundle(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            feature_types=feature_types,
            metadata=metadata,
        )

    raise ValueError(
        f"Failed to generate a valid dataset after {attempts} attempts. Last reason: {last_reason}."
    )


def _apply_filter_torch(
    config: GeneratorConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Run filtering natively in Torch using random forests."""

    details: dict[str, Any] = {"enabled": config.filter.enabled}
    if not config.filter.enabled:
        return True, details

    accepted, filter_details = apply_torch_rf_filter(
        x,
        y,
        task=config.dataset.task,
        seed=seed,
        n_trees=config.filter.n_trees,
        depth=config.filter.depth,
        min_samples_leaf=config.filter.min_samples_leaf,
        max_leaf_nodes=config.filter.max_leaf_nodes,
        max_features=config.filter.max_features,
        n_split_candidates=config.filter.n_split_candidates,
        n_bootstrap=config.filter.n_bootstrap,
        threshold=config.filter.threshold,
    )
    details.update(filter_details)
    details["accepted"] = accepted
    return accepted, details


def _generate_one_seeded(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
    auto_stage: int,
) -> DatasetBundle:
    """Generate one dataset for a fully resolved seed/device/stage context."""

    manager = SeedManager(seed)
    curriculum = curriculum_core._sample_curriculum(
        config,
        manager,
        auto_stage=auto_stage,
        sample_stage_rows_fn=curriculum_core._sample_stage_rows,
        split_counts_fn=curriculum_core._split_counts,
    )
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu", curriculum=curriculum)
    return _generate_one_with_resolved_layout(
        config,
        seed=seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        curriculum=curriculum,
        layout=layout,
    )


def _resolve_auto_stage(mode: str | int, *, seed: int) -> int:
    """Resolve curriculum auto stage for a deterministic dataset seed."""

    if mode == CURRICULUM_STAGE_AUTO:
        stage_gen = SeedManager(seed).torch_rng("curriculum", "stage")
        return curriculum_core._sample_auto_stage(stage_gen)
    if isinstance(mode, int):
        return mode
    return 1


def _validate_class_split_for_layout(
    config: GeneratorConfig,
    *,
    layout: dict[str, Any],
    curriculum: dict[str, Any],
) -> None:
    """Validate class/split feasibility for one sampled layout/curriculum pair."""

    if config.dataset.task != "classification":
        return
    validate_class_split_feasibility(
        n_classes=int(layout["n_classes"]),
        n_train=int(curriculum["n_train"]),
        n_test=int(curriculum["n_test"]),
        context=(
            "sampled classification split constraints "
            f"(curriculum_mode={curriculum.get('mode')}, stage={curriculum.get('stage')})"
        ),
    )


def _generate_one_with_resolved_layout(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
    curriculum: dict[str, Any],
    layout: dict[str, Any],
) -> DatasetBundle:
    """Generate one dataset from an already-resolved curriculum/layout context."""

    _validate_class_split_for_layout(config, layout=layout, curriculum=curriculum)
    manager = SeedManager(seed)
    data_seed = manager.child("data")
    shift_params = resolve_shift_runtime_params(config)
    n_train = int(curriculum["n_train"])
    n_test = int(curriculum["n_test"])

    if requested_device == "auto" and resolved_device == "mps":
        try:
            return _generate_torch(
                config,
                layout,
                data_seed,
                resolved_device,
                n_train=n_train,
                n_test=n_test,
                curriculum=curriculum,
                shift_params=shift_params,
            )
        except Exception:
            # Keep auto mode robust on partially supported MPS runtimes by retrying on CPU.
            return _generate_torch(
                config,
                layout,
                data_seed,
                "cpu",
                n_train=n_train,
                n_test=n_test,
                curriculum=curriculum,
                shift_params=shift_params,
            )

    return _generate_torch(
        config,
        layout,
        data_seed,
        resolved_device,
        n_train=n_train,
        n_test=n_test,
        curriculum=curriculum,
        shift_params=shift_params,
    )


def _layout_signature(layout: dict[str, Any]) -> str:
    """Return a deterministic, stable signature for a sampled layout payload."""

    adjacency = layout.get("adjacency")
    adjacency_payload: list[list[int]]
    if isinstance(adjacency, torch.Tensor):
        adjacency_payload = adjacency.to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_payload = torch.as_tensor(adjacency, dtype=torch.int64, device="cpu").tolist()

    signature_payload = {
        "n_features": int(layout["n_features"]),
        "n_classes": int(layout["n_classes"]),
        "feature_types": list(layout["feature_types"]),
        "card_by_feature": {
            str(int(k)): int(v) for k, v in sorted(dict(layout["card_by_feature"]).items())
        },
        "graph_nodes": int(layout["graph_nodes"]),
        "graph_edges": int(layout["graph_edges"]),
        "graph_depth_nodes": int(layout["graph_depth_nodes"]),
        "feature_node_assignment": [int(v) for v in list(layout["feature_node_assignment"])],
        "target_node_assignment": int(layout["target_node_assignment"]),
        "adjacency": adjacency_payload,
    }
    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def sample_fixed_layout(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> FixedLayoutPlan:
    """Sample one reusable layout/curriculum plan for fixed-layout batch generation."""

    run_seed = seed if seed is not None else config.seed
    mode = normalize_curriculum_stage(config.curriculum_stage)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    auto_stage = _resolve_auto_stage(mode, seed=run_seed)
    manager = SeedManager(run_seed)
    curriculum = curriculum_core._sample_curriculum(
        config,
        manager,
        auto_stage=auto_stage,
        sample_stage_rows_fn=curriculum_core._sample_stage_rows,
        split_counts_fn=curriculum_core._split_counts,
    )
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu", curriculum=curriculum)
    _validate_class_split_for_layout(config, layout=layout, curriculum=curriculum)
    return FixedLayoutPlan(
        curriculum=curriculum,
        layout=layout,
        requested_device=requested_device,
        resolved_device=resolved_device,
        plan_seed=int(run_seed),
        mode=mode,
        auto_stage=int(auto_stage),
        layout_signature=_layout_signature(layout),
    )


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle with deterministic per-dataset randomness."""

    run_seed = seed if seed is not None else config.seed
    mode = normalize_curriculum_stage(config.curriculum_stage)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    auto_stage = 1 if mode == CURRICULUM_STAGE_AUTO else _resolve_auto_stage(mode, seed=run_seed)
    return _generate_one_seeded(
        config,
        seed=run_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        auto_stage=auto_stage,
    )


def generate_batch(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> list[DatasetBundle]:
    """Generate a batch of datasets using deterministic per-dataset child seeds."""

    return list(
        generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        )
    )


def generate_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets lazily using deterministic per-dataset child seeds."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    mode = normalize_curriculum_stage(config.curriculum_stage)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    run_seed = seed if seed is not None else config.seed
    manager = SeedManager(run_seed)
    for i in range(num_datasets):
        dataset_seed = manager.child("dataset", i)
        yield _generate_one_seeded(
            config,
            seed=dataset_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            auto_stage=_resolve_auto_stage(mode, seed=dataset_seed),
        )


def _annotate_fixed_layout_metadata(bundle: DatasetBundle, *, plan: FixedLayoutPlan) -> None:
    """Attach fixed-layout provenance metadata to an emitted bundle."""

    bundle.metadata["layout_mode"] = "fixed"
    bundle.metadata["layout_plan_seed"] = int(plan.plan_seed)
    bundle.metadata["layout_signature"] = str(plan.layout_signature)


def generate_batch_fixed_layout_iter(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets that share one pre-sampled fixed layout and curriculum."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    run_seed = seed if seed is not None else config.seed
    manager = SeedManager(run_seed)
    for i in range(num_datasets):
        dataset_seed = manager.child("dataset", i)
        bundle = _generate_one_with_resolved_layout(
            config,
            seed=dataset_seed,
            requested_device=plan.requested_device,
            resolved_device=plan.resolved_device,
            curriculum=plan.curriculum,
            layout=plan.layout,
        )
        _annotate_fixed_layout_metadata(bundle, plan=plan)
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
