"""Seeded synthetic dataset generation with Torch execution on all devices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, fields
import math
from typing import Any

import torch

from cauchy_generator.config import (
    CURRICULUM_STAGE_AUTO,
    CURRICULUM_STAGE_OFF,
    DatasetConfig,
    GeneratorConfig,
    normalize_curriculum_stage,
)
from cauchy_generator.core.node_pipeline import (
    ConverterSpec,
    apply_node_pipeline,
)
from cauchy_generator.core.steering_metrics import extract_steering_metrics
from cauchy_generator.diagnostics.types import DatasetMetrics
from cauchy_generator.filtering import apply_torch_rf_filter
from cauchy_generator.graph import sample_cauchy_dag
from cauchy_generator.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    validate_lineage_payload,
)
from cauchy_generator.postprocess import inject_missingness, postprocess_dataset
from cauchy_generator.rng import SeedManager
from cauchy_generator.sampling import CorrelatedSampler
from cauchy_generator.types import DatasetBundle

_CURRICULUM_STAGE1_ROWS = 1024
_CURRICULUM_STAGE2_MIN_ROWS = 400
_CURRICULUM_STAGE2_MAX_ROWS = 10_240
_CURRICULUM_STAGE3_MIN_ROWS = 400
_CURRICULUM_STAGE3_MAX_ROWS = 60_000
_CURRICULUM_STAGE1_TRAIN_FRACTION_MIN = 0.30
_CURRICULUM_STAGE1_TRAIN_FRACTION_MAX = 0.90
_CURRICULUM_STAGE23_TRAIN_FRACTION = 0.80
_DEFAULT_CONFIGURED_N_TRAIN = int(DatasetConfig().n_train)
_DEFAULT_CONFIGURED_N_TEST = int(DatasetConfig().n_test)
_STEERING_SUPPORTED_METRICS = frozenset(
    field_info.name for field_info in fields(DatasetMetrics) if field_info.name != "task"
)
_STEERING_CLASSIFICATION_ONLY_METRICS = frozenset(
    {"class_entropy", "majority_minority_ratio", "n_classes"}
)


def _sample_log_uniform_int(generator: torch.Generator, device: str, low: int, high: int) -> int:
    """Sample an integer from a log-uniform range [low, high]."""

    log_low = math.log(float(low))
    log_high = math.log(float(high))
    u = torch.empty(1, device=device).uniform_(log_low, log_high, generator=generator)
    sampled = int(math.exp(u.item()))
    return max(low, min(high, sampled))


def _split_counts(n_total: int, train_fraction: float) -> tuple[int, int]:
    """Split total rows into train/test while ensuring both sides are non-empty."""

    n_total = max(2, int(n_total))
    n_train = int(round(float(train_fraction) * n_total))
    n_train = min(max(1, n_train), n_total - 1)
    n_test = n_total - n_train
    return n_train, n_test


def _sample_stage_rows(stage: int, generator: torch.Generator, device: str) -> tuple[int, float]:
    """Sample total rows and train fraction for one curriculum stage."""

    if stage == 1:
        frac = (
            torch.empty(1, device=device)
            .uniform_(
                _CURRICULUM_STAGE1_TRAIN_FRACTION_MIN,
                _CURRICULUM_STAGE1_TRAIN_FRACTION_MAX,
                generator=generator,
            )
            .item()
        )
        return (_CURRICULUM_STAGE1_ROWS, float(frac))
    if stage == 2:
        return (
            _sample_log_uniform_int(
                generator, device, _CURRICULUM_STAGE2_MIN_ROWS, _CURRICULUM_STAGE2_MAX_ROWS
            ),
            _CURRICULUM_STAGE23_TRAIN_FRACTION,
        )
    if stage == 3:
        return (
            _sample_log_uniform_int(
                generator, device, _CURRICULUM_STAGE3_MIN_ROWS, _CURRICULUM_STAGE3_MAX_ROWS
            ),
            _CURRICULUM_STAGE23_TRAIN_FRACTION,
        )
    raise ValueError(f"Unsupported curriculum stage '{stage}'. Expected 1, 2, or 3.")


def _sample_auto_stage(generator: torch.Generator) -> int:
    """Sample a curriculum stage uniformly from 1..3."""

    return int(torch.randint(1, 4, (1,), generator=generator).item())


def _sample_curriculum(
    config: GeneratorConfig,
    manager: SeedManager,
    *,
    auto_stage: int,
) -> dict[str, Any]:
    """Resolve stage and sample row/split regime for this dataset seed."""

    mode = normalize_curriculum_stage(config.curriculum_stage)
    configured_n_train = max(1, int(config.dataset.n_train))
    configured_n_test = max(1, int(config.dataset.n_test))
    configured_total = configured_n_train + configured_n_test
    if mode == CURRICULUM_STAGE_OFF:
        return {
            "mode": "off",
            "stage": None,
            "n_rows_total": configured_total,
            "n_train": configured_n_train,
            "n_test": configured_n_test,
            "train_fraction": float(configured_n_train / configured_total),
        }

    stage = auto_stage if mode == CURRICULUM_STAGE_AUTO else int(mode)
    rows_gen = manager.torch_rng("curriculum", "rows", stage)
    sampled_rows_total, train_fraction = _sample_stage_rows(stage, rows_gen, "cpu")
    configured_total = max(2, configured_total)
    n_rows_total = sampled_rows_total
    # Preserve caller-provided split-size knobs as a workload ceiling when
    # they intentionally deviate from baseline defaults.
    has_split_override = (
        configured_n_train != _DEFAULT_CONFIGURED_N_TRAIN
        or configured_n_test != _DEFAULT_CONFIGURED_N_TEST
    )
    if has_split_override:
        n_rows_total = min(sampled_rows_total, configured_total)
    n_train, n_test = _split_counts(n_rows_total, train_fraction)
    return {
        "mode": CURRICULUM_STAGE_AUTO if mode == CURRICULUM_STAGE_AUTO else "fixed",
        "stage": stage,
        "n_rows_total": n_rows_total,
        "n_train": n_train,
        "n_test": n_test,
        "train_fraction": float(train_fraction),
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


def _sample_node_count(
    n_nodes_min: int,
    n_nodes_max: int,
    generator: torch.Generator,
    device: str,
) -> int:
    """Sample graph node count using log-uniform bounds."""

    low = max(2, int(n_nodes_min))
    high = max(low, int(n_nodes_max))
    return _sample_log_uniform_int(generator, device, low, high)


def _sample_assignments(
    n_cols: int, n_nodes: int, generator: torch.Generator, device: str
) -> list[int]:
    """Assign columns to a random eligible subset of graph nodes."""

    eligible_count = int(torch.randint(1, n_nodes + 1, (1,), generator=generator).item())
    all_nodes = torch.randperm(n_nodes, generator=generator, device=device)
    eligible_nodes = all_nodes[:eligible_count]
    # Sample with replacement from eligible nodes
    indices = torch.randint(0, eligible_count, (n_cols,), generator=generator, device=device)
    return eligible_nodes[indices].tolist()


def _resolve_stagewise_layout_bounds(
    config: GeneratorConfig, curriculum: dict[str, Any]
) -> tuple[int, int, int, int]:
    """Resolve effective feature/node sampling bounds for a curriculum stage."""

    feature_min = int(config.dataset.n_features_min)
    feature_max = int(config.dataset.n_features_max)
    node_min = int(config.graph.n_nodes_min)
    node_max = int(config.graph.n_nodes_max)

    stage_raw = curriculum.get("stage")
    if stage_raw is not None:
        stage = int(stage_raw)
        stage_cfg = config.curriculum.stages.get(stage)
        if stage_cfg is not None:
            if stage_cfg.n_features_min is not None:
                feature_min = int(stage_cfg.n_features_min)
            if stage_cfg.n_features_max is not None:
                feature_max = int(stage_cfg.n_features_max)
            if stage_cfg.n_nodes_min is not None:
                node_min = int(stage_cfg.n_nodes_min)
            if stage_cfg.n_nodes_max is not None:
                node_max = int(stage_cfg.n_nodes_max)

    if feature_min > feature_max:
        raise ValueError(
            "Invalid effective feature bounds after curriculum stage resolution: "
            f"n_features_min={feature_min} > n_features_max={feature_max}."
        )
    if node_min > node_max:
        raise ValueError(
            "Invalid effective node bounds after curriculum stage resolution: "
            f"n_nodes_min={node_min} > n_nodes_max={node_max}."
        )
    return feature_min, feature_max, node_min, node_max


def _sample_layout(
    config: GeneratorConfig,
    generator: torch.Generator,
    device: str,
    *,
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    """Sample dataset layout, graph, and node assignments for one dataset instance."""

    feature_min, feature_max, node_min, node_max = _resolve_stagewise_layout_bounds(
        config, curriculum
    )
    n_features = int(
        torch.randint(
            feature_min,
            feature_max + 1,
            (1,),
            generator=generator,
        ).item()
    )

    corr = CorrelatedSampler(generator, device)
    raw_ratio = corr.sample_num(
        "categorical_ratio",
        config.dataset.categorical_ratio_min,
        config.dataset.categorical_ratio_max,
        log_scale=False,
        as_int=False,
    )
    cat_ratio = float(max(0.0, min(1.0, raw_ratio)))
    n_cat = int(round(cat_ratio * n_features))
    n_cat = max(0, min(n_features, n_cat))
    if n_cat > 0:
        cat_idx_t = torch.randperm(n_features, generator=generator, device=device)[:n_cat]
        cat_idx_t, _ = torch.sort(cat_idx_t)
        cat_idx = cat_idx_t.tolist()
    else:
        cat_idx = []

    max_card = max(2, config.dataset.max_categorical_cardinality)
    cardinalities = []
    for _ in cat_idx:
        log_low = math.log(2.0)
        log_high = math.log(float(max_card))
        u = torch.empty(1, device=device).uniform_(log_low, log_high, generator=generator)
        cardinalities.append(max(2, int(math.exp(u.item()))))
    card_by_feature = {
        int(idx): int(card) for idx, card in zip(cat_idx, cardinalities, strict=True)
    }

    n_classes = int(
        torch.randint(
            config.dataset.n_classes_min,
            config.dataset.n_classes_max + 1,
            (1,),
            generator=generator,
        ).item()
    )
    n_classes = max(2, n_classes)

    n_nodes = _sample_node_count(node_min, node_max, generator, device)
    adjacency = sample_cauchy_dag(n_nodes, generator, device)
    feature_node_assignment = _sample_assignments(n_features, n_nodes, generator, device)
    target_node_assignment = _sample_assignments(1, n_nodes, generator, device)[0]

    feature_types = ["num"] * n_features
    for i in cat_idx:
        feature_types[int(i)] = "cat"

    return {
        "n_features": n_features,
        "n_cat": n_cat,
        "cat_idx": cat_idx,
        "cardinalities": cardinalities,
        "card_by_feature": card_by_feature,
        "n_classes": n_classes,
        "feature_types": feature_types,
        "graph_nodes": n_nodes,
        "graph_edges": int(adjacency.sum().item()),
        "adjacency": adjacency,
        "feature_node_assignment": feature_node_assignment,
        "target_node_assignment": target_node_assignment,
    }


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


def _build_node_specs(
    node_index: int,
    layout: dict[str, Any],
    task: str,
    generator: torch.Generator,
) -> list[ConverterSpec]:
    """Build converter specs for one node in the graph execution order."""

    specs: list[ConverterSpec] = []
    feature_assignment = layout["feature_node_assignment"]
    feature_types = list(layout["feature_types"])
    card_by_feature: dict[int, int] = layout["card_by_feature"]

    feature_indices = [i for i, a in enumerate(feature_assignment) if a == node_index]
    for f in feature_indices:
        if feature_types[f] == "cat":
            c = int(card_by_feature[f])
            if c > 2 and torch.empty(1).uniform_(0, 1, generator=generator).item() >= 0.5:
                d = int(torch.randint(1, c, (1,), generator=generator).item())
            else:
                d = c
            specs.append(
                ConverterSpec(key=f"feature_{f}", kind="cat", dim=max(1, d), cardinality=c)
            )
        else:
            specs.append(ConverterSpec(key=f"feature_{f}", kind="num", dim=1))

    if int(layout["target_node_assignment"]) == node_index:
        if task == "classification":
            n_classes = int(layout["n_classes"])
            specs.append(
                ConverterSpec(
                    key="target",
                    kind="target_cls",
                    dim=max(2, n_classes),
                    cardinality=n_classes,
                )
            )
        else:
            specs.append(ConverterSpec(key="target", kind="target_reg", dim=1))
    return specs


def _generate_graph_dataset_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
    *,
    n_rows: int,
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
        spec_gen.manual_seed(seed + node_idx + 1000)
        specs = _build_node_specs(node_idx, layout, task, spec_gen)

        x_node, extracted = apply_node_pipeline(parent_data, n_rows, specs, generator, device)
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
) -> DatasetBundle:
    """Generate one dataset in Torch while preserving postprocess/filter contracts."""

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
            )
        except Exception as exc:
            last_reason = f"generation_exception:{exc.__class__.__name__}"
            continue
        if not bool(aux_meta.get("accepted", True)):
            last_reason = "filtered_out"
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 10_007 + attempt)

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

        metadata = {
            "backend": "torch",
            "device": device,
            "compute_backend": "torch_appendix_full",
            "n_features": int(x_train.shape[1]),
            "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
            "n_classes": (
                int(layout["n_classes"]) if config.dataset.task == "classification" else None
            ),
            "graph_nodes": int(layout["graph_nodes"]),
            "graph_edges": int(layout["graph_edges"]),
            "lineage": _build_lineage_metadata(layout, feature_index_map=feature_index_map),
            "seed": seed,
            "attempt_used": attempt,
            "filter": aux_meta.get("filter", {}),
            "curriculum": curriculum,
            "config": asdict(config),
        }
        if missingness_summary is not None:
            metadata["missingness"] = missingness_summary
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


def _stratified_split_indices(
    y: torch.Tensor,
    n_train: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_indices, test_indices) with proportional class representation.

    For classification tasks this keeps class balance close to proportional and
    ensures classes with at least two members appear in both splits. For
    infeasible combinations, this raises ``ValueError`` with an
    ``infeasible_stratified_split`` prefix.
    """
    n_total = int(y.shape[0])
    n_test = n_total - n_train
    if n_total <= 0 or n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"infeasible_stratified_split: expected 0 < n_train < n_total, got n_train={n_train}, n_total={n_total}."
        )

    classes = torch.unique(y, sorted=True)
    train_frac = n_train / n_total

    cls_indices: list[torch.Tensor] = []
    cls_values: list[int] = []
    cls_train_counts: list[int] = []
    cls_train_min: list[int] = []
    cls_train_max: list[int] = []
    cls_remainders: list[float] = []

    for cls in classes:
        idx = torch.where(y == cls)[0]
        perm = torch.randperm(idx.shape[0], generator=generator, device=device)
        cls_indices.append(idx[perm])
        cls_values.append(int(cls.item()))

        n_cls = int(idx.shape[0])
        proportional = float(n_cls * train_frac)
        base_alloc = int(math.floor(proportional))
        remainder = proportional - base_alloc
        if n_cls >= 2:
            train_min = 1
            train_max = n_cls - 1
        else:
            train_min = 0
            train_max = n_cls

        n_cls_train = max(train_min, min(base_alloc, train_max))
        cls_train_counts.append(n_cls_train)
        cls_train_min.append(train_min)
        cls_train_max.append(train_max)
        cls_remainders.append(remainder)

    deficit = n_train - sum(cls_train_counts)
    if deficit > 0:
        order = sorted(
            range(len(cls_train_counts)), key=lambda i: (-cls_remainders[i], cls_values[i])
        )
        while deficit > 0:
            progressed = False
            for i in order:
                if cls_train_counts[i] < cls_train_max[i]:
                    cls_train_counts[i] += 1
                    deficit -= 1
                    progressed = True
                    if deficit == 0:
                        break
            if not progressed:
                break
        if deficit > 0:
            raise ValueError(
                "infeasible_stratified_split: unable to allocate requested train rows while "
                f"preserving class constraints (remaining={deficit})."
            )
    elif deficit < 0:
        surplus = -deficit
        order = sorted(
            range(len(cls_train_counts)), key=lambda i: (cls_remainders[i], cls_values[i])
        )
        while surplus > 0:
            progressed = False
            for i in order:
                if cls_train_counts[i] > cls_train_min[i]:
                    cls_train_counts[i] -= 1
                    surplus -= 1
                    progressed = True
                    if surplus == 0:
                        break
            if not progressed:
                break
        if surplus > 0:
            raise ValueError(
                "infeasible_stratified_split: unable to allocate requested test rows while "
                f"preserving class constraints (remaining={surplus})."
            )

    if sum(cls_train_counts) != n_train:
        raise ValueError(
            "infeasible_stratified_split: train allocation mismatch after rebalance "
            f"(expected={n_train}, actual={sum(cls_train_counts)})."
        )

    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for idx, n_cls_train in zip(cls_indices, cls_train_counts):
        train_parts.append(idx[:n_cls_train])
        test_parts.append(idx[n_cls_train:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)
    if int(train_idx.shape[0]) != n_train or int(test_idx.shape[0]) != n_test:
        raise ValueError(
            "infeasible_stratified_split: index cardinality mismatch "
            f"(expected_train={n_train}, actual_train={int(train_idx.shape[0])}, "
            f"expected_test={n_test}, actual_test={int(test_idx.shape[0])})."
        )

    # Shuffle within each split
    train_idx = train_idx[torch.randperm(train_idx.shape[0], generator=generator, device=device)]
    test_idx = test_idx[torch.randperm(test_idx.shape[0], generator=generator, device=device)]

    return train_idx, test_idx


def _classification_split_valid(y_train: torch.Tensor, y_test: torch.Tensor) -> bool:
    """Validate classification split constraints."""

    train_classes = set(torch.unique(y_train).tolist())
    test_classes = set(torch.unique(y_test).tolist())
    return len(train_classes) >= 2 and train_classes == test_classes


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
    curriculum = _sample_curriculum(config, manager, auto_stage=auto_stage)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu", curriculum=curriculum)
    data_seed = manager.child("data")
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
            )

    return _generate_torch(
        config,
        layout,
        data_seed,
        resolved_device,
        n_train=n_train,
        n_test=n_test,
        curriculum=curriculum,
    )


def _coerce_meta_target_specs(raw: object) -> dict[str, tuple[float, float, float]]:
    """Normalize target specs into `(min, max, weight)` tuples."""

    normalized: dict[str, tuple[float, float, float]] = {}
    if not isinstance(raw, dict):
        return normalized
    for metric_name, value in raw.items():
        if not isinstance(metric_name, str) or metric_name not in _STEERING_SUPPORTED_METRICS:
            continue
        if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
            continue
        try:
            lo = float(value[0])
            hi = float(value[1])
            weight = float(value[2]) if len(value) == 3 else 1.0
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(weight)):
            continue
        if weight <= 0:
            continue
        if lo > hi:
            lo, hi = hi, lo
        normalized[metric_name] = (lo, hi, weight)
    return normalized


def _resolve_meta_target_specs(config: GeneratorConfig) -> dict[str, tuple[float, float, float]]:
    """Merge target specs from legacy diagnostics and top-level config."""

    merged = _coerce_meta_target_specs(config.diagnostics.meta_feature_targets)
    merged.update(_coerce_meta_target_specs(config.meta_feature_targets))
    return merged


def _collect_unknown_steering_target_metrics(config: GeneratorConfig) -> tuple[str, ...]:
    """Collect unsupported steering target keys from config payloads."""

    unknown: set[str] = set()
    for raw in (config.diagnostics.meta_feature_targets, config.meta_feature_targets):
        if not isinstance(raw, dict):
            continue
        for metric_name in raw:
            if isinstance(metric_name, str) and metric_name not in _STEERING_SUPPORTED_METRICS:
                unknown.add(metric_name)
    return tuple(sorted(unknown))


def _validate_target_specs_for_task(
    *,
    task: str,
    target_specs: dict[str, tuple[float, float, float]],
) -> tuple[str, ...]:
    """Return steering metrics that are incompatible with configured task."""

    if task.strip().lower() != "regression":
        return ()
    incompatible = sorted(set(target_specs).intersection(_STEERING_CLASSIFICATION_ONLY_METRICS))
    return tuple(incompatible)


def _resolve_auto_stage(mode: str | int, *, seed: int) -> int:
    """Resolve curriculum auto stage for a deterministic dataset seed."""

    if mode == CURRICULUM_STAGE_AUTO:
        stage_gen = SeedManager(seed).torch_rng("curriculum", "stage")
        return _sample_auto_stage(stage_gen)
    if isinstance(mode, int):
        return mode
    return 1


def _resolve_steering_settings(
    config: GeneratorConfig,
) -> tuple[bool, dict[str, tuple[float, float, float]], int, float, bool]:
    """Resolve steering policy and validate runtime knobs when enabled."""

    unknown_metrics = _collect_unknown_steering_target_metrics(config)
    target_specs = _resolve_meta_target_specs(config)
    enabled = bool(config.steering.enabled and target_specs)
    max_attempts = max(1, int(config.steering.max_attempts))
    temperature = float(config.steering.temperature)
    if bool(config.steering.enabled):
        if unknown_metrics:
            unknown = ", ".join(unknown_metrics)
            supported = ", ".join(sorted(_STEERING_SUPPORTED_METRICS))
            raise ValueError(
                f"Unsupported steering target metric(s): {unknown}. Supported metrics: {supported}."
            )
        incompatible = _validate_target_specs_for_task(
            task=str(config.dataset.task),
            target_specs=target_specs,
        )
        if incompatible:
            metrics = ", ".join(incompatible)
            raise ValueError(
                "Incompatible steering target metrics for regression task: "
                f"{metrics}. Remove these metrics or switch task=classification."
            )
        if int(config.steering.max_attempts) <= 0:
            raise ValueError(
                f"steering.max_attempts must be >= 1, got {config.steering.max_attempts!r}."
            )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                f"steering.temperature must be a finite value > 0, got {temperature!r}."
            )
    include_spearman = bool(
        "spearman_abs_mean" in target_specs or "spearman_abs_max" in target_specs
    )
    return enabled, target_specs, max_attempts, temperature, include_spearman


def _distance_to_band(value: float, *, lo: float, hi: float) -> tuple[float, bool]:
    """Return normalized distance to a target band and in-band flag."""

    width = max(1e-6, hi - lo)
    if value < lo:
        return (lo - value) / width, False
    if value > hi:
        return (value - hi) / width, False
    return 0.0, True


def _score_candidate_against_targets(
    *,
    metrics: Any,
    target_specs: dict[str, tuple[float, float, float]],
    steering_state: dict[str, dict[str, int]],
) -> tuple[float, dict[str, bool], dict[str, float | None]]:
    """Score one candidate against target bands with under-coverage weighting."""

    score_sum = 0.0
    weight_sum = 0.0
    in_band_flags: dict[str, bool] = {}
    metric_values: dict[str, float | None] = {}
    for metric_name, (lo, hi, base_weight) in sorted(target_specs.items()):
        state = steering_state[metric_name]
        selected_count = int(state["selected"])
        in_band_count = int(state["in_band"])
        in_band_fraction = float(in_band_count / selected_count) if selected_count > 0 else 0.0
        under_coverage = max(0.0, 1.0 - in_band_fraction)
        effective_weight = float(base_weight) * (1.0 + under_coverage)

        if isinstance(metrics, dict):
            raw_value = metrics.get(metric_name)
        else:
            raw_value = getattr(metrics, metric_name, None)
        value = float(raw_value) if isinstance(raw_value, (int, float)) else None
        if value is not None and not math.isfinite(value):
            value = None

        if value is None:
            distance = 1.0
            in_band = False
        else:
            distance, in_band = _distance_to_band(value, lo=lo, hi=hi)
        in_band_flags[metric_name] = in_band
        metric_values[metric_name] = value
        score_sum += effective_weight * distance
        weight_sum += effective_weight

    score = score_sum / weight_sum if weight_sum > 0 else 0.0
    return float(score), in_band_flags, metric_values


def _select_softmax_candidate(
    scores: list[float],
    *,
    temperature: float,
    seed: int,
) -> tuple[int, list[float]]:
    """Sample one candidate index using a deterministic softmax selector."""

    if not scores:
        raise ValueError("Cannot select steering candidate from an empty score set.")
    if len(scores) == 1:
        return 0, [1.0]
    score_arr = torch.tensor(scores, dtype=torch.float64, device="cpu")
    shifted = score_arr - torch.min(score_arr)
    logits = -(shifted / float(temperature))
    logits = logits - torch.max(logits)
    probs = torch.exp(logits)
    prob_sum = float(torch.sum(probs).item())
    if not math.isfinite(prob_sum) or prob_sum <= 0:
        probs = torch.full(
            (score_arr.shape[0],),
            1.0 / score_arr.shape[0],
            dtype=torch.float64,
            device="cpu",
        )
    else:
        probs = probs / prob_sum
    selector_rng = SeedManager(seed).torch_rng("steering", "select", device="cpu")
    idx = int(torch.multinomial(probs, 1, generator=selector_rng).item())
    return idx, probs.tolist()


def _target_payload(
    target_specs: dict[str, tuple[float, float, float]],
) -> dict[str, dict[str, float]]:
    """Build a JSON-safe target payload for steering metadata."""

    return {
        metric_name: {"min": lo, "max": hi, "weight": weight}
        for metric_name, (lo, hi, weight) in sorted(target_specs.items())
    }


def _generate_one_steered(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
    mode: str | int,
    target_specs: dict[str, tuple[float, float, float]],
    max_attempts: int,
    temperature: float,
    include_spearman: bool,
    steering_state: dict[str, dict[str, int]],
) -> DatasetBundle:
    """Generate bounded steering candidates and soft-select one deterministically."""

    seed_manager = SeedManager(seed)
    candidate_bundles: list[DatasetBundle] = []
    candidate_seeds: list[int] = []
    candidate_scores: list[float] = []
    candidate_in_band: list[dict[str, bool]] = []
    candidate_metric_values: list[dict[str, float | None]] = []
    last_error = "unknown"

    for candidate_idx in range(max_attempts):
        candidate_seed = (
            seed
            if candidate_idx == 0
            else seed_manager.child("steering", "candidate", candidate_idx)
        )
        auto_stage = _resolve_auto_stage(mode, seed=candidate_seed)
        try:
            bundle = _generate_one_seeded(
                config,
                seed=candidate_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
                auto_stage=auto_stage,
            )
            metrics = extract_steering_metrics(
                bundle,
                target_metric_names=set(target_specs),
                include_spearman=include_spearman,
            )
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            continue
        score, in_band_flags, metric_values = _score_candidate_against_targets(
            metrics=metrics,
            target_specs=target_specs,
            steering_state=steering_state,
        )
        candidate_bundles.append(bundle)
        candidate_seeds.append(candidate_seed)
        candidate_scores.append(score)
        candidate_in_band.append(in_band_flags)
        candidate_metric_values.append(metric_values)

    if not candidate_bundles:
        raise ValueError(
            "Steering failed to produce a valid candidate after "
            f"{max_attempts} attempts. Last error: {last_error}"
        )

    selected_idx, probabilities = _select_softmax_candidate(
        candidate_scores,
        temperature=temperature,
        seed=seed,
    )
    selected_bundle = candidate_bundles[selected_idx]
    selected_in_band = candidate_in_band[selected_idx]
    for metric_name, in_band in selected_in_band.items():
        state = steering_state[metric_name]
        state["selected"] += 1
        if in_band:
            state["in_band"] += 1

    selected_bundle.metadata["steering"] = {
        "enabled": True,
        "max_attempts": int(max_attempts),
        "temperature": float(temperature),
        "candidate_count": len(candidate_bundles),
        "candidate_seeds": candidate_seeds,
        "scores": [float(score) for score in candidate_scores],
        "probabilities": [float(prob) for prob in probabilities],
        "selected_candidate_index": int(selected_idx),
        "selected_candidate_seed": int(candidate_seeds[selected_idx]),
        "selected_score": float(candidate_scores[selected_idx]),
        "selected_in_band": dict(sorted(selected_in_band.items())),
        "selected_metric_values": dict(sorted(candidate_metric_values[selected_idx].items())),
        "targets": _target_payload(target_specs),
    }
    return selected_bundle


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle with deterministic per-dataset randomness."""

    mode = normalize_curriculum_stage(config.curriculum_stage)
    (
        steering_enabled,
        target_specs,
        steering_max_attempts,
        steering_temperature,
        steering_include_spearman,
    ) = _resolve_steering_settings(config)
    auto_stage = 1
    if isinstance(mode, int):
        auto_stage = mode
    run_seed = seed if seed is not None else config.seed
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    if steering_enabled:
        steering_mode: str | int = 1 if mode == CURRICULUM_STAGE_AUTO else mode
        steering_state = {
            metric_name: {"selected": 0, "in_band": 0} for metric_name in sorted(target_specs)
        }
        return _generate_one_steered(
            config,
            seed=run_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            mode=steering_mode,
            target_specs=target_specs,
            max_attempts=steering_max_attempts,
            temperature=steering_temperature,
            include_spearman=steering_include_spearman,
            steering_state=steering_state,
        )
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
    (
        steering_enabled,
        target_specs,
        steering_max_attempts,
        steering_temperature,
        steering_include_spearman,
    ) = _resolve_steering_settings(config)
    steering_state = {
        metric_name: {"selected": 0, "in_band": 0} for metric_name in sorted(target_specs)
    }
    run_seed = seed if seed is not None else config.seed
    manager = SeedManager(run_seed)
    for i in range(num_datasets):
        dataset_seed = manager.child("dataset", i)
        if steering_enabled:
            yield _generate_one_steered(
                config,
                seed=dataset_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
                mode=mode,
                target_specs=target_specs,
                max_attempts=steering_max_attempts,
                temperature=steering_temperature,
                include_spearman=steering_include_spearman,
                steering_state=steering_state,
            )
            continue

        yield _generate_one_seeded(
            config,
            seed=dataset_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            auto_stage=_resolve_auto_stage(mode, seed=dataset_seed),
        )
