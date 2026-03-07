"""Batched fixed-layout execution-plan sampling and generation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.converters.categorical import _JOINT_VARIANTS
from dagzoo.core.layout import _build_node_specs
from dagzoo.core.layout_types import AggregationKind, LayoutPlan, MechanismFamily
from dagzoo.core.node_pipeline import ConverterSpec
from dagzoo.core.trees import compute_odt_leaf_indices, sample_odt_splits
from dagzoo.functions._rng_helpers import rand_scalar, randint_scalar
from dagzoo.functions.activations import _fixed_activation, fixed_activation_names
from dagzoo.functions.random_functions import _sample_function_family
from dagzoo.linalg.random_matrices import sample_random_matrix
from dagzoo.math_utils import log_uniform as _log_uniform
from dagzoo.rng import SeedManager
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec
from dagzoo.sampling.random_weights import sample_random_weights

_MATRIX_KIND_CHOICES: tuple[str, ...] = (
    "gaussian",
    "weights",
    "singular_values",
    "kernel",
    "activation",
)
_MATRIX_BASE_KIND_CHOICES: tuple[str, ...] = (
    "gaussian",
    "weights",
    "singular_values",
    "kernel",
)
_ROOT_BASE_KIND_CHOICES: tuple[str, ...] = ("normal", "uniform", "unit_ball", "normal_cov")
_PARAM_ACTIVATION_CHOICES: tuple[str, ...] = ("relu_pow", "signed_pow", "inv_pow", "poly")
_AGGREGATION_KIND_ORDER: tuple[AggregationKind, ...] = ("sum", "product", "max", "logsumexp")
_PRODUCT_COMPONENT_FAMILIES: tuple[MechanismFamily, ...] = (
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
)


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _device_generator(seed: int, *, device: str) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _parent_index_lists(layout: LayoutPlan) -> list[list[int]]:
    adjacency = layout.adjacency
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device="cpu")
    return [
        sorted(int(parent_index) for parent_index in torch.where(adjacency[:, node_index])[0])
        for node_index in range(int(layout.graph_nodes))
    ]


def _sample_bool(generator: torch.Generator, *, p: float = 0.5) -> bool:
    return bool(rand_scalar(generator) < p)


def _sample_activation_plan(generator: torch.Generator) -> dict[str, Any]:
    if rand_scalar(generator) < (1.0 / 3.0):
        choice = _PARAM_ACTIVATION_CHOICES[
            int(randint_scalar(0, len(_PARAM_ACTIVATION_CHOICES), generator))
        ]
        payload: dict[str, Any] = {"mode": "parametric", "kind": choice}
        if choice == "poly":
            payload["poly_power"] = int(randint_scalar(2, 6, generator))
        return payload
    fixed = fixed_activation_names()
    name = fixed[int(randint_scalar(0, len(fixed), generator))]
    return {"mode": "fixed", "name": name}


def _sample_matrix_plan(generator: torch.Generator) -> dict[str, Any]:
    kind = _MATRIX_KIND_CHOICES[int(randint_scalar(0, len(_MATRIX_KIND_CHOICES), generator))]
    if kind != "activation":
        return {"kind": kind}
    base_kind = _MATRIX_BASE_KIND_CHOICES[
        int(randint_scalar(0, len(_MATRIX_BASE_KIND_CHOICES), generator))
    ]
    return {
        "kind": "activation",
        "base_kind": base_kind,
        "activation": _sample_activation_plan(generator),
    }


def _sample_function_plan_for_family(
    generator: torch.Generator,
    *,
    family: MechanismFamily,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> dict[str, Any]:
    plan: dict[str, Any] = {"family": family}
    if family == "linear":
        plan["matrix"] = _sample_matrix_plan(generator)
        return plan
    if family == "quadratic":
        plan["matrix"] = _sample_matrix_plan(generator)
        return plan
    if family == "nn":
        n_layers = int(randint_scalar(1, 4, generator))
        hidden_width = int(_log_uniform(generator, 1.0, 127.0, "cpu"))
        plan["n_layers"] = n_layers
        plan["hidden_width"] = max(1, hidden_width)
        plan["apply_input_activation"] = _sample_bool(generator)
        if plan["apply_input_activation"]:
            plan["input_activation"] = _sample_activation_plan(generator)
        plan["apply_output_activation"] = _sample_bool(generator)
        if plan["apply_output_activation"]:
            plan["output_activation"] = _sample_activation_plan(generator)
        layer_count = max(1, n_layers)
        plan["layer_matrices"] = [_sample_matrix_plan(generator) for _ in range(layer_count)]
        plan["hidden_activations"] = [
            _sample_activation_plan(generator) for _ in range(max(0, layer_count - 1))
        ]
        return plan
    if family == "tree":
        n_trees = int(_log_uniform(generator, 1.0, 32.0, "cpu"))
        n_trees = max(1, n_trees)
        plan["n_trees"] = n_trees
        plan["depths"] = [int(randint_scalar(1, 8, generator)) for _ in range(n_trees)]
        return plan
    if family == "discretization":
        n_centers = int(_log_uniform(generator, 2.0, 128.0, "cpu"))
        plan["n_centers"] = max(2, n_centers)
        plan["linear_matrix"] = _sample_matrix_plan(generator)
        return plan
    if family == "gp":
        plan["branch_kind"] = "ha" if _sample_bool(generator) else "projected"
        return plan
    if family == "em":
        m_val = int(_log_uniform(generator, 2.0, float(max(16, 2 * out_dim)), "cpu"))
        plan["m_val"] = max(2, m_val)
        plan["linear_matrix"] = _sample_matrix_plan(generator)
        return plan
    if family == "product":
        eligible = list(_PRODUCT_COMPONENT_FAMILIES)
        if function_family_mix is not None:
            eligible = [
                component
                for component in _PRODUCT_COMPONENT_FAMILIES
                if float(function_family_mix.get(component, 0.0)) > 0.0
            ]
        if not eligible:
            raise ValueError(
                "mechanism.function_family_mix enables 'product' but disables all product "
                "component families for fixed-layout plan sampling."
            )
        lhs_family = eligible[int(randint_scalar(0, len(eligible), generator))]
        rhs_family = eligible[int(randint_scalar(0, len(eligible), generator))]
        plan["lhs"] = _sample_function_plan_for_family(
            generator,
            family=lhs_family,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
        plan["rhs"] = _sample_function_plan_for_family(
            generator,
            family=rhs_family,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
        return plan
    raise ValueError(f"Unsupported mechanism family in fixed-layout plan sampling: {family!r}")


def _sample_function_plan(
    generator: torch.Generator,
    *,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> dict[str, Any]:
    family = _sample_function_family(
        generator,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
    )
    return _sample_function_plan_for_family(
        generator,
        family=family,
        out_dim=out_dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
    )


def _sample_converter_plan(
    generator: torch.Generator,
    spec: ConverterSpec,
    *,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> dict[str, Any]:
    if spec.kind in {"num", "target_reg"}:
        return {
            "kind": str(spec.kind),
            "warp_enabled": not _sample_bool(generator),
        }

    idx_joint = randint_scalar(0, len(_JOINT_VARIANTS), generator)
    selected_method, variant = _JOINT_VARIANTS[int(idx_joint)]
    payload: dict[str, Any] = {
        "kind": str(spec.kind),
        "method": selected_method,
        "variant": variant,
    }
    if variant == "center_random_fn":
        payload["function"] = _sample_function_plan(
            generator,
            out_dim=max(1, int(spec.dim)),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
    return payload


def build_fixed_layout_execution_plans(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    plan_seed: int,
) -> list[dict[str, Any]]:
    """Build one reusable per-node execution-plan payload for fixed-layout batches."""

    manager = SeedManager(plan_seed)
    task = str(config.dataset.task)
    node_plans: list[dict[str, Any]] = []
    for node_index, parent_indices in enumerate(_parent_index_lists(layout)):
        spec_gen = _cpu_generator(manager.child("node_spec", node_index))
        plan_gen = _cpu_generator(manager.child("node_plan", node_index))
        converter_specs = _build_node_specs(node_index, layout, task, spec_gen)
        required_dim = int(sum(max(1, int(spec.dim)) for spec in converter_specs))
        latent_extra = max(1, int(_log_uniform(plan_gen, 1.0, 32.0, "cpu")))
        total_dim = int(required_dim + latent_extra)
        node_plan: dict[str, Any] = {
            "node_index": int(node_index),
            "parent_indices": [int(parent_index) for parent_index in parent_indices],
            "converter_specs": [
                {
                    "key": str(spec.key),
                    "kind": str(spec.kind),
                    "dim": int(spec.dim),
                    "cardinality": None if spec.cardinality is None else int(spec.cardinality),
                }
                for spec in converter_specs
            ],
            "converter_plans": [
                _sample_converter_plan(
                    plan_gen,
                    spec,
                    mechanism_logit_tilt=0.0,
                    function_family_mix=config.mechanism.function_family_mix,
                )
                for spec in converter_specs
            ],
            "latent": {
                "required_dim": required_dim,
                "extra_dim": latent_extra,
                "total_dim": total_dim,
            },
        }
        if parent_indices:
            combine_kind = "concat" if _sample_bool(plan_gen) else "stack"
            node_plan["source_kind"] = "multi"
            node_plan["combine_kind"] = combine_kind
            if combine_kind == "concat":
                node_plan["function"] = _sample_function_plan(
                    plan_gen,
                    out_dim=total_dim,
                    mechanism_logit_tilt=0.0,
                    function_family_mix=config.mechanism.function_family_mix,
                )
            else:
                node_plan["aggregation_kind"] = _AGGREGATION_KIND_ORDER[
                    int(randint_scalar(0, len(_AGGREGATION_KIND_ORDER), plan_gen))
                ]
                node_plan["parent_functions"] = [
                    _sample_function_plan(
                        plan_gen,
                        out_dim=total_dim,
                        mechanism_logit_tilt=0.0,
                        function_family_mix=config.mechanism.function_family_mix,
                    )
                    for _ in parent_indices
                ]
        else:
            node_plan["source_kind"] = "random_points"
            node_plan["base_kind"] = _ROOT_BASE_KIND_CHOICES[
                int(randint_scalar(0, len(_ROOT_BASE_KIND_CHOICES), plan_gen))
            ]
            node_plan["function"] = _sample_function_plan(
                plan_gen,
                out_dim=total_dim,
                mechanism_logit_tilt=0.0,
                function_family_mix=config.mechanism.function_family_mix,
            )
        node_plans.append(node_plan)
    return node_plans


def fixed_layout_plan_signature(node_plans: list[dict[str, Any]]) -> str:
    """Return a deterministic signature for one fixed-layout execution plan payload."""

    encoded = json.dumps(node_plans, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _batch_standardize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=1, keepdim=True)
    std = torch.std(x, dim=1, keepdim=True)
    return (x - mean) / torch.clamp(std, min=1e-6)


def _stack_from_generators(
    generators: list[torch.Generator],
    fn: Any,
) -> torch.Tensor:
    return torch.stack([fn(generator) for generator in generators], dim=0)


def _sample_uniform_scalars(
    generators: list[torch.Generator],
    *,
    low: float,
    high: float,
    device: str,
) -> torch.Tensor:
    return torch.as_tensor(
        [
            float(torch.empty(1, device=device).uniform_(low, high, generator=generator).item())
            for generator in generators
        ],
        dtype=torch.float32,
        device=device,
    )


def _sample_log_uniform_scalars(
    generators: list[torch.Generator],
    *,
    low: float,
    high: float,
    device: str,
) -> torch.Tensor:
    return torch.as_tensor(
        [_log_uniform(generator, low, high, device) for generator in generators],
        dtype=torch.float32,
        device=device,
    )


def _batched_randperm_indices(
    generators: list[torch.Generator],
    *,
    length: int,
    sample_size: int,
    device: str,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(length, generator=generator, device=device)[:sample_size]
            for generator in generators
        ],
        dim=0,
    )


def _apply_activation_plan(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    with_standardize: bool,
) -> torch.Tensor:
    y = x.to(torch.float32)
    if y.dim() == 2:
        y = y.unsqueeze(0)

    device = str(y.device)
    if with_standardize:
        y = _batch_standardize(y)
        a = _sample_log_uniform_scalars(generators, low=1.0, high=10.0, device=device)
        row_idx = torch.as_tensor(
            [int(randint_scalar(0, y.shape[1], generator)) for generator in generators],
            dtype=torch.long,
            device=y.device,
        )
        offsets = y[torch.arange(y.shape[0], device=y.device), row_idx].unsqueeze(1)
        y = a.view(-1, 1, 1) * (y - offsets)

    if str(plan["mode"]) == "parametric":
        kind = str(plan["kind"])
        if kind == "relu_pow":
            q = _sample_log_uniform_scalars(generators, low=0.1, high=10.0, device=device)
            y = torch.pow(torch.clamp(y, min=0.0), q.view(-1, 1, 1))
        elif kind == "signed_pow":
            q = _sample_log_uniform_scalars(generators, low=0.1, high=10.0, device=device)
            y = torch.sign(y) * torch.pow(torch.abs(y), q.view(-1, 1, 1))
        elif kind == "inv_pow":
            q = _sample_log_uniform_scalars(generators, low=0.1, high=10.0, device=device)
            y = torch.pow(torch.abs(y) + 1e-3, -q.view(-1, 1, 1))
        elif kind == "poly":
            y = torch.pow(y, float(int(plan["poly_power"])))
        else:
            raise ValueError(f"Unknown activation plan kind: {kind!r}")
    else:
        name = str(plan["name"])
        y = torch.stack([_fixed_activation(value, name) for value in y], dim=0)

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    if with_standardize:
        y = _batch_standardize(y)
    return y.to(torch.float32)


def _sample_random_matrix_from_plan(
    plan: dict[str, Any],
    *,
    out_dim: int,
    in_dim: int,
    generator: torch.Generator,
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    kind = str(plan["kind"])
    if kind != "activation":
        return sample_random_matrix(
            out_dim,
            in_dim,
            generator,
            device,
            kind=kind,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    matrix = sample_random_matrix(
        out_dim,
        in_dim,
        generator,
        device,
        kind=str(plan["base_kind"]),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    matrix = _apply_activation_plan(
        matrix.unsqueeze(0),
        [generator],
        plan["activation"],
        with_standardize=False,
    )[0]
    matrix = matrix + 1e-6 * sample_noise_from_spec(
        matrix.shape,
        generator=generator,
        device=device,
        noise_spec=noise_spec,
    )
    norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
    return matrix / torch.clamp(norms, min=1e-6)


def _apply_linear_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    matrices = _stack_from_generators(
        generators,
        lambda generator: _sample_random_matrix_from_plan(
            plan["matrix"],
            out_dim=out_dim,
            in_dim=int(x.shape[2]),
            generator=generator,
            device=str(x.device),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )
    return torch.einsum("bni,boi->bno", x, matrices)


def _apply_quadratic_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    feature_cap = min(int(x.shape[2]), 20)
    if int(x.shape[2]) > feature_cap:
        indices = _batched_randperm_indices(
            generators,
            length=int(x.shape[2]),
            sample_size=feature_cap,
            device=str(x.device),
        )
        x_sub = torch.stack(
            [x[idx, :, indices[idx]] for idx in range(x.shape[0])],
            dim=0,
        )
    else:
        x_sub = x
    ones = torch.ones((x_sub.shape[0], x_sub.shape[1], 1), device=x.device, dtype=x_sub.dtype)
    x_aug = torch.cat([x_sub, ones], dim=2)
    matrices = _stack_from_generators(
        generators,
        lambda generator: torch.stack(
            [
                _sample_random_matrix_from_plan(
                    plan["matrix"],
                    out_dim=int(x_aug.shape[2]),
                    in_dim=int(x_aug.shape[2]),
                    generator=generator,
                    device=str(x.device),
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=noise_spec,
                )
                for _ in range(out_dim)
            ],
            dim=0,
        ),
    )
    return torch.einsum("bni,boij,bnj->bno", x_aug, matrices, x_aug)


def _sample_unit_ball_batch(
    generators: list[torch.Generator],
    *,
    n_rows: int,
    dim: int,
    device: str,
) -> torch.Tensor:
    vectors = _stack_from_generators(
        generators,
        lambda generator: torch.randn(n_rows, dim, generator=generator, device=device),
    )
    vectors = vectors / torch.clamp(torch.norm(vectors, dim=2, keepdim=True), min=1e-6)
    radii = _stack_from_generators(
        generators,
        lambda generator: torch.empty(n_rows, 1, device=device).uniform_(
            0.0, 1.0, generator=generator
        ),
    )
    return vectors * torch.pow(radii, 1.0 / max(1, dim))


def _sample_random_points_batch(
    generators: list[torch.Generator],
    *,
    n_rows: int,
    dim: int,
    device: str,
    base_kind: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    if base_kind == "normal":
        return _stack_from_generators(
            generators,
            lambda generator: sample_noise_from_spec(
                (n_rows, dim),
                generator=generator,
                device=device,
                noise_spec=noise_spec,
            ),
        )
    if base_kind == "uniform":
        return _stack_from_generators(
            generators,
            lambda generator: torch.empty(n_rows, dim, device=device).uniform_(
                -1.0, 1.0, generator=generator
            ),
        )
    if base_kind == "unit_ball":
        return _sample_unit_ball_batch(generators, n_rows=n_rows, dim=dim, device=device)

    points = _stack_from_generators(
        generators,
        lambda generator: sample_noise_from_spec(
            (n_rows, dim),
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        ),
    )
    weights = _stack_from_generators(
        generators,
        lambda generator: sample_random_weights(
            dim,
            generator,
            device,
            sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )
    matrices = _stack_from_generators(
        generators,
        lambda generator: sample_noise_from_spec(
            (dim, dim),
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        ),
    )
    return torch.einsum("bni,bi,bij->bnj", points, weights, matrices.transpose(1, 2))


def _apply_nn_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    y = x
    if bool(plan.get("apply_input_activation")):
        y = _apply_activation_plan(y, generators, plan["input_activation"], with_standardize=True)

    hidden_width = max(1, int(plan["hidden_width"]))
    n_layers = max(1, int(plan["n_layers"]))
    layer_dims = [int(x.shape[2])]
    for _ in range(max(0, n_layers - 1)):
        layer_dims.append(hidden_width)
    layer_dims.append(int(out_dim))

    for layer_index, (din, dout) in enumerate(zip(layer_dims[:-1], layer_dims[1:], strict=True)):
        matrices = _stack_from_generators(
            generators,
            lambda generator: _sample_random_matrix_from_plan(
                plan["layer_matrices"][layer_index],
                out_dim=int(dout),
                in_dim=int(din),
                generator=generator,
                device=str(x.device),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            ),
        )
        y = torch.einsum("bni,boi->bno", y, matrices)
        if layer_index < len(layer_dims) - 2:
            y = _apply_activation_plan(
                y,
                generators,
                plan["hidden_activations"][layer_index],
                with_standardize=True,
            )

    if bool(plan.get("apply_output_activation")):
        y = _apply_activation_plan(y, generators, plan["output_activation"], with_standardize=True)
    return y


def _apply_tree_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size = int(x.shape[0])
    outputs = torch.zeros((batch_size, x.shape[1], out_dim), device=x.device, dtype=torch.float32)
    std = torch.std(x, dim=1)
    totals = torch.sum(std, dim=1)
    for tree_index, depth in enumerate(plan["depths"]):
        for batch_index, generator in enumerate(generators):
            local_std = std[batch_index]
            total = float(totals[batch_index].item())
            if not math.isfinite(total) or total <= 1e-12:
                probs = torch.ones_like(local_std) / max(1, len(local_std))
            else:
                probs = torch.clamp(local_std, min=0.0)
                probs /= torch.clamp(torch.sum(probs), min=1e-12)
                if not torch.all(torch.isfinite(probs)) or torch.any(probs < 0):
                    probs = torch.ones_like(local_std) / max(1, len(local_std))
            split_dims, thresholds = sample_odt_splits(
                x[batch_index],
                int(depth),
                generator,
                feature_probs=probs,
            )
            leaf_idx = compute_odt_leaf_indices(x[batch_index], split_dims, thresholds)
            n_leaves = 2 ** int(depth)
            leaf_vals = sample_noise_from_spec(
                (n_leaves, out_dim),
                generator=generator,
                device=str(x.device),
                noise_spec=noise_spec,
            )
            outputs[batch_index] += leaf_vals[leaf_idx]
    return outputs / float(max(1, int(plan["n_trees"])))


def _apply_discretization_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    n_centers = min(int(plan["n_centers"]), int(x.shape[1]))
    center_idx = _batched_randperm_indices(
        generators,
        length=int(x.shape[1]),
        sample_size=n_centers,
        device=str(x.device),
    )
    centers = torch.stack([x[idx, center_idx[idx]] for idx in range(x.shape[0])], dim=0)
    p = _sample_log_uniform_scalars(generators, low=0.5, high=4.0, device=str(x.device))
    dist = torch.pow(
        torch.abs(x.unsqueeze(2) - centers.unsqueeze(1)),
        p.view(-1, 1, 1, 1),
    ).sum(dim=3)
    nearest = torch.argmin(dist, dim=2)
    gathered = torch.stack(
        [centers[idx, nearest[idx]] for idx in range(x.shape[0])],
        dim=0,
    )
    linear_plan = {"family": "linear", "matrix": dict(plan["linear_matrix"])}
    return _apply_linear_batch(
        gathered,
        generators,
        linear_plan,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _sample_radial_ha_batch(
    generators: list[torch.Generator],
    *,
    n: int,
    a: torch.Tensor,
    device: str,
) -> torch.Tensor:
    u = _stack_from_generators(
        generators,
        lambda generator: torch.empty(n, device=device).uniform_(0.0, 1.0, generator=generator),
    )
    return torch.pow(1.0 - u, 1.0 / (1.0 - a.view(-1, 1))) - 1.0


def _apply_gp_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size, _, din = x.shape
    p = 256
    device = str(x.device)
    a = _sample_log_uniform_scalars(generators, low=2.0, high=20.0, device=device)

    if str(plan["branch_kind"]) == "ha":
        r = _sample_radial_ha_batch(generators, n=p * din, a=a, device=device).view(
            batch_size, p, din
        )
        signs = _stack_from_generators(
            generators,
            lambda generator: torch.where(
                torch.empty(p, din, device=device).uniform_(0.0, 1.0, generator=generator) < 0.5,
                -1.0,
                1.0,
            ),
        )
        omega = r * signs
        x_proj = x
    else:
        z = _stack_from_generators(
            generators,
            lambda generator: sample_noise_from_spec(
                (p, din),
                generator=generator,
                device=device,
                noise_spec=noise_spec,
            ),
        )
        z = z / torch.clamp(torch.norm(z, dim=2, keepdim=True), min=1e-6)
        r = _sample_radial_ha_batch(generators, n=p, a=a, device=device)
        omega = z * r.unsqueeze(2)
        weights = _stack_from_generators(
            generators,
            lambda generator: sample_random_weights(
                din,
                generator,
                device,
                sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            ),
        )
        alpha = _sample_log_uniform_scalars(generators, low=0.5, high=10.0, device=device)
        a_mat = _stack_from_generators(
            generators,
            lambda generator: sample_noise_from_spec(
                (din, din),
                generator=generator,
                device=device,
                noise_spec=noise_spec,
            ),
        )
        matrices = alpha.view(-1, 1, 1) * (weights.unsqueeze(2) * a_mat)
        x_proj = torch.einsum("bni,bij->bnj", x, matrices.transpose(1, 2))

    b = _stack_from_generators(
        generators,
        lambda generator: torch.empty(p, device=device).uniform_(
            0.0, 2.0 * math.pi, generator=generator
        ),
    )
    phi = torch.cos(torch.einsum("bnd,bpd->bnp", x_proj, omega) + b.unsqueeze(1))
    z_out = _stack_from_generators(
        generators,
        lambda generator: sample_noise_from_spec(
            (out_dim, p),
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        ),
    )
    return torch.einsum("bnp,bop->bno", phi, z_out) / math.sqrt(float(p))


def _apply_em_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    m_val = max(2, int(plan["m_val"]))
    base_idx = torch.stack(
        [
            torch.randint(0, x.shape[1], (m_val,), generator=generator, device=x.device)
            for generator in generators
        ],
        dim=0,
    )
    centers = torch.stack([x[idx, base_idx[idx]] for idx in range(x.shape[0])], dim=0)
    centers = centers + _stack_from_generators(
        generators,
        lambda generator: sample_noise_from_spec(
            (m_val, x.shape[2]),
            generator=generator,
            device=str(x.device),
            noise_spec=noise_spec,
        ),
    )
    sigma = torch.exp(
        _stack_from_generators(
            generators,
            lambda generator: sample_noise_from_spec(
                (m_val,),
                generator=generator,
                device=str(x.device),
                noise_spec=noise_spec,
                scale_multiplier=0.1,
            ),
        )
    )
    p_val = _sample_log_uniform_scalars(generators, low=1.0, high=4.0, device=str(x.device))
    q_val = _sample_log_uniform_scalars(generators, low=1.0, high=2.0, device=str(x.device))
    dist_p = torch.pow(
        torch.abs(x.unsqueeze(2) - centers.unsqueeze(1)),
        p_val.view(-1, 1, 1, 1),
    ).sum(dim=3) ** (1.0 / p_val.view(-1, 1, 1))
    logits = -0.5 * torch.log(2.0 * math.pi * sigma**2).unsqueeze(1) - torch.pow(
        dist_p / torch.clamp(sigma.unsqueeze(1), min=1e-6),
        q_val.view(-1, 1, 1),
    )
    probs = torch.softmax(logits, dim=2)
    linear_plan = {"family": "linear", "matrix": dict(plan["linear_matrix"])}
    return _apply_linear_batch(
        probs,
        generators,
        linear_plan,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def apply_function_plan_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    """Apply one frozen function-family plan across a batch of datasets."""

    y = _batch_standardize(x.to(torch.float32))
    family = str(plan["family"])
    if family == "linear":
        return _apply_linear_batch(
            y,
            generators,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "quadratic":
        return _apply_quadratic_batch(
            y,
            generators,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "nn":
        return _apply_nn_batch(
            y,
            generators,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "tree":
        return _apply_tree_batch(y, generators, plan, out_dim=out_dim, noise_spec=noise_spec)
    if family == "discretization":
        return _apply_discretization_batch(
            y,
            generators,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "gp":
        return _apply_gp_batch(
            y,
            generators,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "em":
        return _apply_em_batch(
            y,
            generators,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "product":
        lhs = apply_function_plan_batch(
            y,
            generators,
            plan["lhs"],
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        rhs = apply_function_plan_batch(
            y,
            generators,
            plan["rhs"],
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        return lhs * rhs
    raise ValueError(f"Unsupported fixed-layout function family: {family!r}")


def _apply_numeric_converter_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    if y.dim() == 2:
        y = y.unsqueeze(2)
    values = y[:, :, 0].clone()
    if not bool(plan.get("warp_enabled")):
        return y, values

    a = _sample_log_uniform_scalars(generators, low=0.2, high=5.0, device=str(x.device))
    b = _sample_log_uniform_scalars(generators, low=0.2, high=5.0, device=str(x.device))
    lo = torch.min(y, dim=1, keepdim=True).values
    hi = torch.max(y, dim=1, keepdim=True).values
    scaled = (y - lo) / torch.clamp(hi - lo, min=1e-6)
    warped = 1.0 - torch.pow(
        1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a.view(-1, 1, 1)),
        b.view(-1, 1, 1),
    )
    return warped, values


def _apply_categorical_converter_batch(
    x: torch.Tensor,
    generators: list[torch.Generator],
    plan: dict[str, Any],
    *,
    n_categories: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    if y.dim() == 2:
        y = y.unsqueeze(2)

    device = str(y.device)
    category_count = max(2, int(n_categories))
    method = str(plan["method"])
    variant = str(plan["variant"])

    centers: torch.Tensor | None = None
    if method == "neighbor":
        n_centers = min(category_count, int(y.shape[1]))
        idx = _batched_randperm_indices(
            generators,
            length=int(y.shape[1]),
            sample_size=n_centers,
            device=device,
        )
        centers = torch.stack([y[row, idx[row]] for row in range(y.shape[0])], dim=0)
        p = _sample_log_uniform_scalars(generators, low=0.5, high=4.0, device=device)
        dist = torch.pow(
            torch.abs(y.unsqueeze(2) - centers.unsqueeze(1)),
            p.view(-1, 1, 1, 1),
        ).sum(dim=3)
        labels = torch.argmin(dist, dim=2)
        if n_centers < category_count:
            labels = labels % category_count
    else:
        if y.shape[2] != category_count:
            projections = _stack_from_generators(
                generators,
                lambda generator: torch.randn(
                    int(y.shape[2]),
                    category_count,
                    generator=generator,
                    device=device,
                ),
            )
            logits_in = torch.einsum("bni,bic->bnc", y, projections)
        else:
            logits_in = y
        logits_std = _batch_standardize(logits_in)
        a = _sample_log_uniform_scalars(generators, low=0.1, high=10.0, device=device)
        w = _stack_from_generators(
            generators,
            lambda generator: torch.empty(category_count, device=device).uniform_(
                0.0, 1.0, generator=generator
            ),
        )
        b = torch.log(w + 1e-4)
        logits = a.view(-1, 1, 1) * logits_std + b.unsqueeze(1)
        probs = torch.softmax(logits, dim=2)
        labels = torch.stack(
            [
                torch.multinomial(probs[row], 1, generator=generators[row]).squeeze(1)
                for row in range(probs.shape[0])
            ],
            dim=0,
        )

    width = int(y.shape[2])
    if variant == "input":
        out = y
    elif variant == "index_repeat":
        out = labels.unsqueeze(2).repeat(1, 1, width).to(torch.float32)
    elif variant == "center":
        if centers is None:
            out = y
        else:
            gathered = torch.stack(
                [centers[row, labels[row] % centers.shape[1]] for row in range(y.shape[0])],
                dim=0,
            )
            if gathered.shape[2] != width:
                if gathered.shape[2] > width:
                    gathered = gathered[:, :, :width]
                else:
                    gathered = torch.nn.functional.pad(gathered, (0, width - gathered.shape[2]))
            out = gathered
    elif variant == "center_random_fn":
        nested_input = (
            y
            if centers is None
            else torch.stack(
                [centers[row, labels[row] % centers.shape[1]] for row in range(y.shape[0])],
                dim=0,
            )
        )
        out = apply_function_plan_batch(
            nested_input,
            generators,
            plan["function"],
            out_dim=width,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    elif variant == "softmax_points":
        points = _stack_from_generators(
            generators,
            lambda generator: torch.randn(
                category_count, width, generator=generator, device=device
            ),
        )
        out = torch.stack(
            [points[row, labels[row] % category_count] for row in range(y.shape[0])],
            dim=0,
        )
    else:
        out = y

    out = torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    labels = torch.remainder(labels.to(torch.int64), category_count)
    return out, labels


def _apply_node_plan_batch(
    config: GeneratorConfig,
    node_plan: dict[str, Any],
    parent_data: list[torch.Tensor],
    *,
    n_rows: int,
    generators: list[torch.Generator],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_dim = int(node_plan["latent"]["total_dim"])
    required_dim = int(node_plan["latent"]["required_dim"])
    if parent_data:
        if str(node_plan["combine_kind"]) == "concat":
            concat = torch.cat(parent_data, dim=2)
            latent = apply_function_plan_batch(
                concat,
                generators,
                node_plan["function"],
                out_dim=total_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
        else:
            transformed = [
                apply_function_plan_batch(
                    parent_tensor,
                    generators,
                    node_plan["parent_functions"][plan_index],
                    out_dim=total_dim,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=noise_spec,
                )
                for plan_index, parent_tensor in enumerate(parent_data)
            ]
            stacked = torch.stack(transformed, dim=2)
            aggregation_kind = str(node_plan["aggregation_kind"])
            if aggregation_kind == "sum":
                latent = torch.sum(stacked, dim=2)
            elif aggregation_kind == "product":
                latent = torch.prod(stacked, dim=2)
            elif aggregation_kind == "max":
                latent = torch.max(stacked, dim=2).values
            else:
                latent = torch.logsumexp(stacked, dim=2)
    else:
        base = _sample_random_points_batch(
            generators,
            n_rows=n_rows,
            dim=total_dim,
            device=device,
            base_kind=str(node_plan["base_kind"]),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        latent = apply_function_plan_batch(
            base,
            generators,
            node_plan["function"],
            out_dim=total_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _batch_standardize(latent)

    weights = _stack_from_generators(
        generators,
        lambda generator: sample_random_weights(
            int(latent.shape[2]),
            generator,
            device,
            sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )
    latent = latent * weights.unsqueeze(1)
    mean_l2 = torch.mean(torch.norm(latent, dim=2), dim=1)
    latent = latent / torch.clamp(mean_l2.view(-1, 1, 1), min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    column_cursor = 0
    converter_specs = node_plan["converter_specs"]
    converter_plans = node_plan["converter_plans"]
    for spec_index, spec_payload in enumerate(converter_specs):
        spec_dim = max(1, int(spec_payload["dim"]))
        if column_cursor + spec_dim > required_dim or column_cursor + spec_dim > int(
            latent.shape[2]
        ):
            raise RuntimeError(
                "Latent width underflow in fixed-layout batched node pipeline: "
                f"required={required_dim}, available={int(latent.shape[2])}, "
                f"requested_end={column_cursor + spec_dim}."
            )
        view = latent[:, :, column_cursor : column_cursor + spec_dim]
        converter_plan = converter_plans[spec_index]
        kind = str(spec_payload["kind"])
        if kind == "cat":
            x_prime, values = _apply_categorical_converter_batch(
                view,
                generators,
                converter_plan,
                n_categories=int(spec_payload["cardinality"] or 2),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
        elif kind == "target_cls":
            x_prime, values = _apply_categorical_converter_batch(
                view,
                generators,
                converter_plan,
                n_categories=max(2, int(spec_payload["cardinality"] or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
        elif kind in {"num", "target_reg"}:
            x_prime, values = _apply_numeric_converter_batch(
                view[:, :, :1], generators, converter_plan
            )
        else:
            raise ValueError(f"Unknown converter kind in fixed-layout plan: {kind!r}")

        if int(x_prime.shape[2]) != spec_dim:
            if int(x_prime.shape[2]) > spec_dim:
                x_prime = x_prime[:, :, :spec_dim]
            else:
                x_prime = torch.nn.functional.pad(x_prime, (0, spec_dim - int(x_prime.shape[2])))
        latent[:, :, column_cursor : column_cursor + spec_dim] = x_prime
        extracted[str(spec_payload["key"])] = values
        column_cursor += spec_dim

    scale = _sample_log_uniform_scalars(generators, low=0.1, high=10.0, device=device)
    latent = latent * scale.view(-1, 1, 1)
    return latent, extracted


def generate_fixed_layout_graph_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    node_plans: list[dict[str, Any]],
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Generate one fixed-layout microbatch of raw `x`/`y` tensors."""

    if not dataset_seeds:
        raise ValueError("dataset_seeds must be non-empty.")
    batch_size = len(dataset_seeds)
    n_rows = int(config.dataset.n_train + config.dataset.n_test)
    num_features = int(layout.n_features)
    dtype = torch.float32
    generators = [_device_generator(seed, device=device) for seed in dataset_seeds]

    node_outputs: list[torch.Tensor | None] = [None] * int(layout.graph_nodes)
    feature_values: list[torch.Tensor | None] = [None] * num_features
    target_values: torch.Tensor | None = None
    aux_meta_batch = [{"filter": {"mode": "deferred", "status": "not_run"}} for _ in dataset_seeds]

    for node_index, node_plan in enumerate(node_plans):
        parent_tensors = []
        for parent_index in node_plan["parent_indices"]:
            parent_output = node_outputs[int(parent_index)]
            if parent_output is not None:
                parent_tensors.append(parent_output)
        latent, extracted = _apply_node_plan_batch(
            config,
            node_plan,
            parent_tensors,
            n_rows=n_rows,
            generators=generators,
            device=device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        node_outputs[node_index] = latent
        for key, values in extracted.items():
            if key.startswith("feature_"):
                feature_index = int(key.split("_", 1)[1])
                feature_values[feature_index] = values
            elif key == "target":
                target_values = values
            else:
                raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")

    x = torch.zeros((batch_size, n_rows, num_features), dtype=dtype, device=device)
    feature_types = list(layout.feature_types)
    card_by_feature = dict(layout.card_by_feature)
    for feature_index in range(num_features):
        feature_tensor: torch.Tensor | None = feature_values[feature_index]
        if feature_tensor is None:
            if feature_types[feature_index] == "cat":
                cardinality = int(card_by_feature[feature_index])
                feature_tensor = _stack_from_generators(
                    generators,
                    lambda generator: torch.randint(
                        0,
                        cardinality,
                        (n_rows,),
                        generator=generator,
                        device=device,
                    ),
                )
            else:
                feature_tensor = _stack_from_generators(
                    generators,
                    lambda generator: sample_noise_from_spec(
                        (n_rows,),
                        generator=generator,
                        device=device,
                        noise_spec=noise_spec,
                    ),
                )
        x[:, :, feature_index] = feature_tensor.to(dtype)

    if target_values is None:
        if str(config.dataset.task) == "classification":
            y = _stack_from_generators(
                generators,
                lambda generator: torch.randint(
                    0,
                    int(layout.n_classes),
                    (n_rows,),
                    generator=generator,
                    device=device,
                ),
            ).to(torch.int64)
        else:
            y = _stack_from_generators(
                generators,
                lambda generator: sample_noise_from_spec(
                    (n_rows,),
                    generator=generator,
                    device=device,
                    noise_spec=noise_spec,
                ),
            ).to(dtype)
    else:
        if str(config.dataset.task) == "classification":
            y = target_values.to(torch.int64) % int(layout.n_classes)
        else:
            y = target_values.to(dtype)
    return x, y, aux_meta_batch
