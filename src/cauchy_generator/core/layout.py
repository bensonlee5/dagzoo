"""Layout and node-spec sampling helpers for dataset generation."""

from __future__ import annotations

import math
from typing import Any

import torch

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.graph import dag_edge_density, dag_longest_path_nodes, sample_cauchy_dag
from cauchy_generator.core.shift import resolve_shift_runtime_params
from cauchy_generator.core.node_pipeline import ConverterSpec
from cauchy_generator.sampling import CorrelatedSampler


def _sample_log_uniform_int(generator: torch.Generator, device: str, low: int, high: int) -> int:
    """Sample an integer from a log-uniform range [low, high]."""

    log_low = math.log(float(low))
    log_high = math.log(float(high))
    u = torch.empty(1, device=device).uniform_(log_low, log_high, generator=generator)
    sampled = int(math.exp(u.item()))
    return max(low, min(high, sampled))


def _sample_node_count(
    n_nodes_min: int,
    n_nodes_max: int,
    generator: torch.Generator,
    device: str,
) -> int:
    """Sample graph node count using log-uniform bounds."""

    low = max(2, int(n_nodes_min))
    high = max(2, int(n_nodes_max))
    if low > high:
        raise ValueError(f"graph.n_nodes_min must be <= n_nodes_max, got {low} > {high}.")
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


def _sample_layout(
    config: GeneratorConfig,
    generator: torch.Generator,
    device: str,
) -> dict[str, Any]:
    """Sample dataset layout, graph, and node assignments for one dataset instance."""

    shift_params = resolve_shift_runtime_params(config)
    sampled_feature_min = int(config.dataset.n_features_min)
    sampled_feature_max = int(config.dataset.n_features_max)
    if sampled_feature_min > sampled_feature_max:
        raise ValueError(
            "dataset.n_features_min must be <= n_features_max, "
            f"got {sampled_feature_min} > {sampled_feature_max}."
        )
    n_features = int(
        torch.randint(
            sampled_feature_min,
            sampled_feature_max + 1,
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

    n_nodes = _sample_node_count(
        int(config.graph.n_nodes_min),
        int(config.graph.n_nodes_max),
        generator,
        device,
    )
    adjacency = sample_cauchy_dag(
        n_nodes,
        generator,
        edge_logit_bias=float(shift_params.edge_logit_bias_shift),
    )
    graph_depth_nodes = dag_longest_path_nodes(adjacency)
    graph_edge_density = dag_edge_density(adjacency)
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
        "graph_depth_nodes": int(graph_depth_nodes),
        "graph_edge_density": float(graph_edge_density),
        "adjacency": adjacency,
        "feature_node_assignment": feature_node_assignment,
        "target_node_assignment": target_node_assignment,
    }


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
