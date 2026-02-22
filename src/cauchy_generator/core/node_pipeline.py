"""Node-level generation pipeline (Appendix E.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cauchy_generator.converters.categorical import (
    apply_categorical_converter_torch,
)
from cauchy_generator.converters.numeric import (
    apply_numeric_converter_torch,
)
from cauchy_generator.functions.multi import (
    apply_multi_function_torch,
)
from cauchy_generator.math_utils import (
    log_uniform_torch as _log_uniform_torch,
    standardize_torch as _standardize_torch,
)
from cauchy_generator.sampling.random_points import (
    sample_random_points_torch,
)
from cauchy_generator.sampling.random_weights import (
    sample_random_weights_torch,
)


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: str
    dim: int
    cardinality: int | None = None


def apply_node_pipeline_torch(
    parent_data: list[torch.Tensor],
    n_rows: int,
    converter_specs: list[ConverterSpec],
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Appendix E.5-style node transform in torch."""
    required_dim = int(sum(max(1, s.dim) for s in converter_specs))
    latent_extra = int(_log_uniform_torch(generator, 1.0, 32.0, device))
    total_dim = required_dim + max(1, latent_extra)

    if parent_data:
        x = apply_multi_function_torch(parent_data, generator, out_dim=total_dim)
    else:
        x = sample_random_points_torch(n_rows, total_dim, generator, device)

    x = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    x = torch.clamp(x, -1e6, 1e6)
    x = _standardize_torch(x)

    w = sample_random_weights_torch(x.shape[1], generator, device)
    x = x * w.unsqueeze(0)

    mean_l2 = torch.mean(torch.norm(x, dim=1))
    x = x / torch.clamp(mean_l2, min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    cursor = 0
    for spec in converter_specs:
        d = max(1, int(spec.dim))
        if cursor + d > x.shape[1]:
            pad = torch.randn(
                x.shape[0], cursor + d - x.shape[1], generator=generator, device=device
            )
            x = torch.cat([x, pad], dim=1)

        view = x[:, cursor : cursor + d]
        if spec.kind == "cat":
            if spec.cardinality is None:
                raise ValueError(f"Missing cardinality for categorical spec: {spec.key}")
            x_prime, v = apply_categorical_converter_torch(
                view, generator, n_categories=int(spec.cardinality)
            )
            extracted[spec.key] = v
        elif spec.kind == "num":
            x_prime, v = apply_numeric_converter_torch(view[:, :1], generator)
            extracted[spec.key] = v
        elif spec.kind == "target_cls":
            cls = max(2, int(spec.cardinality or 2))
            x_prime, v = apply_categorical_converter_torch(view, generator, n_categories=cls)
            extracted[spec.key] = v
        elif spec.kind == "target_reg":
            x_prime, v = apply_numeric_converter_torch(view[:, :1], generator)
            extracted[spec.key] = v
        else:
            raise ValueError(f"Unknown converter kind: {spec.kind}")

        if x_prime.shape[1] != d:
            if x_prime.shape[1] > d:
                x_prime = x_prime[:, :d]
            else:
                x_prime = torch.nn.functional.pad(x_prime, (0, d - x_prime.shape[1]))
        x[:, cursor : cursor + d] = x_prime
        cursor += d

    scale = _log_uniform_torch(generator, 0.1, 10.0, device)
    x *= scale
    return x, extracted
