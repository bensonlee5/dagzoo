"""Seeded synthetic dataset generation with Torch execution on all devices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import torch

from cauchy_generator.config import (
    NOISE_FAMILY_MIXTURE,
    GeneratorConfig,
    NoiseFamily,
    validate_class_split_feasibility,
)
from cauchy_generator.core.constants import (
    NODE_SPEC_SEED_OFFSET,
    SPLIT_PERMUTATION_SEED_OFFSET,
)
from cauchy_generator.core.layout import _build_node_specs, _sample_layout
from cauchy_generator.core.metadata import (
    _build_lineage_metadata,
    _build_shift_metadata,
)
from cauchy_generator.core.shift import ShiftRuntimeParams, resolve_shift_runtime_params
from cauchy_generator.core.validation import _classification_split_valid, _stratified_split_indices
from cauchy_generator.filtering import apply_extra_trees_filter
from cauchy_generator.core.node_pipeline import apply_node_pipeline
from cauchy_generator.postprocess import inject_missingness, postprocess_dataset
from cauchy_generator.rng import SeedManager, offset_seed32, validate_seed32
from cauchy_generator.sampling.noise import (
    NoiseSamplingSpec,
    normalize_mixture_weights,
    sample_mixture_component_family,
    sample_noise_from_spec,
)
from cauchy_generator.types import DatasetBundle


@dataclass(slots=True)
class FixedLayoutPlan:
    """Pre-sampled layout bundle for fixed-layout batch generation."""

    layout: dict[str, Any]
    requested_device: str
    resolved_device: str
    plan_seed: int
    n_train: int
    n_test: int
    layout_signature: str
    compatibility_snapshot: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class NoiseRuntimeSelection:
    """Resolved per-dataset noise-family runtime selection."""

    family_requested: NoiseFamily
    family_sampled: NoiseFamily
    sampling_strategy: str
    scale: float
    student_t_df: float
    mixture_weights: dict[str, float] | None = None


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


def _resolve_split_sizes(config: GeneratorConfig) -> tuple[int, int]:
    """Resolve explicit train/test split sizes from config."""

    return int(config.dataset.n_train), int(config.dataset.n_test)


def _resolve_run_seed(config: GeneratorConfig, seed_override: int | None) -> int:
    """Resolve and validate the run seed used by generation entrypoints."""

    if seed_override is None:
        return validate_seed32(config.seed, field_name="seed")
    return validate_seed32(seed_override, field_name="seed")


def _resolve_noise_runtime_selection(
    config: GeneratorConfig,
    *,
    run_seed: int,
) -> NoiseRuntimeSelection:
    """Resolve deterministic per-dataset noise-family selection."""

    family_requested = config.noise.family
    scale = float(config.noise.scale)
    student_t_df = float(config.noise.student_t_df)
    if family_requested != NOISE_FAMILY_MIXTURE:
        return NoiseRuntimeSelection(
            family_requested=family_requested,
            family_sampled=family_requested,
            sampling_strategy="dataset_level",
            scale=scale,
            student_t_df=student_t_df,
            mixture_weights=None,
        )

    mixture_weights_raw = (
        {str(key): float(value) for key, value in config.noise.mixture_weights.items()}
        if config.noise.mixture_weights is not None
        else None
    )
    normalized_weights = normalize_mixture_weights(mixture_weights_raw)
    selector = SeedManager(run_seed).torch_rng("noise_family", device="cpu")
    sampled_family = sample_mixture_component_family(
        generator=selector,
        device="cpu",
        mixture_weights=normalized_weights,
    )
    return NoiseRuntimeSelection(
        family_requested=family_requested,
        family_sampled=sampled_family,
        sampling_strategy="dataset_level",
        scale=scale,
        student_t_df=student_t_df,
        mixture_weights={key: float(value) for key, value in normalized_weights.items()},
    )


def _noise_sampling_spec(selection: NoiseRuntimeSelection) -> NoiseSamplingSpec:
    """Build a concrete noise sampling spec from runtime selection."""

    return NoiseSamplingSpec(
        family=selection.family_sampled,
        scale=float(selection.scale),
        student_t_df=float(selection.student_t_df),
    )


def _build_noise_metadata(selection: NoiseRuntimeSelection) -> dict[str, Any]:
    """Build per-dataset noise metadata payload."""

    return {
        "family_requested": str(selection.family_requested),
        "family_sampled": str(selection.family_sampled),
        "sampling_strategy": str(selection.sampling_strategy),
        "scale": float(selection.scale),
        "student_t_df": float(selection.student_t_df),
        "mixture_weights": (
            {key: float(value) for key, value in selection.mixture_weights.items()}
            if selection.mixture_weights is not None
            else None
        ),
    }


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
    noise_spec: NoiseSamplingSpec | None = None,
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
        spec_gen.manual_seed(offset_seed32(seed, NODE_SPEC_SEED_OFFSET + node_idx))
        specs = _build_node_specs(node_idx, layout, task, spec_gen)

        x_node, extracted = apply_node_pipeline(
            parent_data,
            n_rows,
            specs,
            generator,
            device,
            mechanism_logit_tilt=mechanism_logit_tilt,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
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
                v = sample_noise_from_spec(
                    (n_rows,),
                    generator=generator,
                    device=device,
                    noise_spec=noise_spec,
                )
        x[:, i] = v.to(dtype)

    if target_values is None:
        if task == "classification":
            y = torch.randint(0, n_classes, (n_rows,), generator=generator, device=device)
        else:
            y = sample_noise_from_spec(
                (n_rows,),
                generator=generator,
                device=device,
                noise_spec=noise_spec,
            ).to(dtype)
    else:
        if task == "classification":
            y = target_values.to(torch.long) % n_classes
        else:
            y = target_values.to(dtype)

    accepted, filter_details = _apply_filter(config, x, y, seed=seed)
    return x, y, {"accepted": accepted, "filter": filter_details}


def _generate_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
    *,
    n_train: int,
    n_test: int,
    shift_params: ShiftRuntimeParams | None = None,
    noise_runtime_selection: NoiseRuntimeSelection | None = None,
    preserve_feature_schema: bool = False,
) -> DatasetBundle:
    """Generate one dataset in Torch while preserving postprocess/filter contracts."""

    shift_params = shift_params or resolve_shift_runtime_params(config)
    attempts = max(1, int(config.filter.max_attempts))
    last_reason = "unknown"
    dtype = _torch_dtype(config)
    n_rows = n_train + n_test
    noise_runtime_selection = noise_runtime_selection or _resolve_noise_runtime_selection(
        config,
        run_seed=seed,
    )
    noise_spec = _noise_sampling_spec(noise_runtime_selection)

    for attempt in range(attempts):
        try:
            x, y, aux_meta = _generate_graph_dataset_torch(
                config,
                layout,
                offset_seed32(seed, attempt),
                device,
                n_rows=n_rows,
                mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
                noise_sigma_multiplier=float(shift_params.noise_sigma_multiplier),
                noise_spec=noise_spec,
            )
        except Exception as exc:
            last_reason = f"generation_exception:{exc.__class__.__name__}"
            continue
        if not bool(aux_meta.get("accepted", True)):
            last_reason = "filtered_out"
            continue

        # Keep split/postprocess control-plane randomness on CPU to avoid tiny-op accelerator overhead.
        split_postprocess_generator = torch.Generator(device="cpu")
        split_postprocess_generator.manual_seed(
            offset_seed32(seed, SPLIT_PERMUTATION_SEED_OFFSET + attempt)
        )

        if config.dataset.task == "classification":
            try:
                train_idx_cpu, test_idx_cpu = _stratified_split_indices(
                    y.to(device="cpu"),
                    n_train,
                    split_postprocess_generator,
                    "cpu",
                )
            except ValueError as exc:
                if str(exc).startswith("infeasible_stratified_split"):
                    last_reason = "invalid_class_split"
                    continue
                raise
            train_idx = train_idx_cpu.to(device=x.device)
            test_idx = test_idx_cpu.to(device=x.device)
            x_train_t, x_test_t = x[train_idx], x[test_idx]
            y_train_t, y_test_t = y[train_idx], y[test_idx]
        else:
            order_cpu = torch.randperm(
                x.shape[0],
                generator=split_postprocess_generator,
                device="cpu",
            )
            order = order_cpu.to(device=x.device)
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
            split_postprocess_generator,
            device,
            return_feature_index_map=True,
            preserve_feature_schema=preserve_feature_schema,
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

        x_train = x_train.to(device=device, dtype=dtype)
        x_test = x_test.to(device=device, dtype=dtype)
        y_dtype = torch.int64 if config.dataset.task == "classification" else dtype
        y_train = y_train.to(device=device, dtype=y_dtype)
        y_test = y_test.to(device=device, dtype=y_dtype)
        class_structure: dict[str, Any] | None = None
        n_classes: int | None = None
        if config.dataset.task == "classification":
            class_structure = _classification_class_structure(
                y_train=y_train,
                y_test=y_test,
                n_classes_sampled=int(layout["n_classes"]),
            )
            n_classes = int(class_structure["n_classes_realized"])
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
            "shift": shift_metadata,
            "noise": _build_noise_metadata(noise_runtime_selection),
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


def _apply_filter(
    config: GeneratorConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Run filtering via CPU ExtraTrees."""

    details: dict[str, Any] = {"enabled": config.filter.enabled}
    if not config.filter.enabled:
        return True, details

    accepted, filter_details = apply_extra_trees_filter(
        x,
        y,
        task=config.dataset.task,
        seed=seed,
        n_estimators=config.filter.n_estimators,
        max_depth=config.filter.max_depth,
        min_samples_leaf=config.filter.min_samples_leaf,
        max_leaf_nodes=config.filter.max_leaf_nodes,
        max_features=config.filter.max_features,
        n_bootstrap=config.filter.n_bootstrap,
        threshold=config.filter.threshold,
        n_jobs=config.filter.n_jobs,
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
) -> DatasetBundle:
    """Generate one dataset for a fully resolved seed/device context."""

    manager = SeedManager(seed)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu")
    n_train, n_test = _resolve_split_sizes(config)
    return _generate_one_with_resolved_layout(
        config,
        seed=seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        n_train=n_train,
        n_test=n_test,
        layout=layout,
    )


def _validate_class_split_for_layout(
    config: GeneratorConfig,
    *,
    layout: dict[str, Any],
    n_train: int,
    n_test: int,
) -> None:
    """Validate class/split feasibility for one sampled layout/split pair."""

    if config.dataset.task != "classification":
        return
    validate_class_split_feasibility(
        n_classes=int(layout["n_classes"]),
        n_train=int(n_train),
        n_test=int(n_test),
        context=(
            "sampled classification split constraints "
            f"(n_train={int(n_train)}, n_test={int(n_test)})"
        ),
    )


def _generate_one_with_resolved_layout(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
    n_train: int,
    n_test: int,
    layout: dict[str, Any],
    preserve_feature_schema: bool = False,
) -> DatasetBundle:
    """Generate one dataset from an already-resolved split/layout context."""

    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    manager = SeedManager(seed)
    data_seed = manager.child("data")
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)

    if requested_device == "auto" and resolved_device == "mps":
        try:
            return _generate_torch(
                config,
                layout,
                data_seed,
                resolved_device,
                n_train=n_train,
                n_test=n_test,
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                preserve_feature_schema=preserve_feature_schema,
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
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                preserve_feature_schema=preserve_feature_schema,
            )

    return _generate_torch(
        config,
        layout,
        data_seed,
        resolved_device,
        n_train=n_train,
        n_test=n_test,
        shift_params=shift_params,
        noise_runtime_selection=noise_runtime_selection,
        preserve_feature_schema=preserve_feature_schema,
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


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle with deterministic per-dataset randomness."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    return _generate_one_seeded(
        config,
        seed=run_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
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

    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    run_seed = _resolve_run_seed(config, seed)
    manager = SeedManager(run_seed)
    for i in range(num_datasets):
        dataset_seed = manager.child("dataset", i)
        yield _generate_one_seeded(
            config,
            seed=dataset_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
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
    if snapshot is None:
        raise ValueError(
            "Fixed-layout plan is missing compatibility snapshot. "
            "Resample with sample_fixed_layout(...) before generation."
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
