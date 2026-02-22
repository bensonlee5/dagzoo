"""Component-level microbenchmarks for hot path sanity checks."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_one
from cauchy_generator.core.node_pipeline import ConverterSpec, apply_node_pipeline_torch
from cauchy_generator.functions.random_functions import apply_random_function


def _time_ms(func: Callable[[], None], repeats: int) -> float:
    """Run ``func`` repeatedly and return the average runtime in milliseconds."""

    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / max(1, repeats)) * 1000.0


def _micro_config(config: GeneratorConfig) -> GeneratorConfig:
    """Create a lightweight config variant for inexpensive microbenchmarks."""

    c = GeneratorConfig.from_dict(config.to_dict())
    c.dataset.n_train = min(128, c.dataset.n_train)
    c.dataset.n_test = min(64, c.dataset.n_test)
    c.dataset.n_features_min = min(16, c.dataset.n_features_min)
    c.dataset.n_features_max = min(16, c.dataset.n_features_max)
    c.graph.n_nodes_min = min(8, c.graph.n_nodes_min)
    c.graph.n_nodes_max = min(8, c.graph.n_nodes_max)
    return c


def run_microbenchmarks(
    config: GeneratorConfig,
    *,
    device: str | None = None,
    repeats: int = 5,
) -> dict[str, float | int]:
    """Run targeted microbenchmarks for core generation components."""

    rng = np.random.default_rng(config.seed + 41)
    x = rng.normal(size=(512, 64)).astype(np.float32)

    def run_linear() -> None:
        local_rng = np.random.default_rng(1)
        _ = apply_random_function(x, local_rng, out_dim=64, function_type="linear")

    parent_data = [
        torch.randn(512, 24),
        torch.randn(512, 24),
    ]
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=4, cardinality=6),
        ConverterSpec(key="target", kind="target_cls", dim=3, cardinality=3),
    ]

    def run_node_pipeline() -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(2)
        _ = apply_node_pipeline_torch(parent_data, 512, specs, g, "cpu")

    micro_cfg = _micro_config(config)

    def run_generate_one() -> None:
        _ = generate_one(micro_cfg, seed=micro_cfg.seed + 99, device=device)

    return {
        "micro_repeats": int(max(1, repeats)),
        "micro_random_function_linear_ms": _time_ms(run_linear, repeats),
        "micro_node_pipeline_ms": _time_ms(run_node_pipeline, repeats),
        "micro_generate_one_ms": _time_ms(run_generate_one, repeats),
    }
