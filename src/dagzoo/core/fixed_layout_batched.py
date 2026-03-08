"""Batched fixed-layout execution-plan sampling and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from dagzoo.core.trees import (
    compute_odt_leaf_indices_batch,
    sample_odt_splits_batch,
)
from dagzoo.functions._rng_helpers import rand_scalar, randint_scalar
from dagzoo.functions.activations import _fixed_activation, fixed_activation_names
from dagzoo.functions.random_functions import _sample_function_family
from dagzoo.math_utils import log_uniform as _log_uniform
from dagzoo.rng import SeedManager
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

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

_FIXED_LAYOUT_EXECUTION_CONTRACT = "chunk_batched_v1"


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
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
    mechanism_logit_tilt: float,
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
        converter_spec_payloads: list[dict[str, Any]] = []
        column_cursor = 0
        for spec in converter_specs:
            spec_dim = max(1, int(spec.dim))
            converter_spec_payloads.append(
                {
                    "key": str(spec.key),
                    "kind": str(spec.kind),
                    "dim": int(spec.dim),
                    "cardinality": None if spec.cardinality is None else int(spec.cardinality),
                    "column_start": int(column_cursor),
                    "column_end": int(column_cursor + spec_dim),
                }
            )
            column_cursor += spec_dim
        converter_plan_payloads = [
            _sample_converter_plan(
                plan_gen,
                spec,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=config.mechanism.function_family_mix,
            )
            for spec in converter_specs
        ]
        node_plan: dict[str, Any] = {
            "node_index": int(node_index),
            "parent_indices": [int(parent_index) for parent_index in parent_indices],
            "converter_specs": converter_spec_payloads,
            "converter_plans": converter_plan_payloads,
            "latent": {
                "required_dim": required_dim,
                "extra_dim": latent_extra,
                "total_dim": total_dim,
            },
            "execution_contract": _FIXED_LAYOUT_EXECUTION_CONTRACT,
        }
        if parent_indices:
            combine_kind = "concat" if _sample_bool(plan_gen) else "stack"
            node_plan["source_kind"] = "multi"
            node_plan["combine_kind"] = combine_kind
            if combine_kind == "concat":
                node_plan["function"] = _sample_function_plan(
                    plan_gen,
                    out_dim=total_dim,
                    mechanism_logit_tilt=mechanism_logit_tilt,
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
                        mechanism_logit_tilt=mechanism_logit_tilt,
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
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=config.mechanism.function_family_mix,
            )
        node_plan["converter_groups"] = _build_converter_groups(
            converter_spec_payloads,
            converter_plan_payloads,
        )
        node_plans.append(node_plan)
    return node_plans


def fixed_layout_plan_signature(node_plans: list[dict[str, Any]]) -> str:
    """Return a deterministic signature for one fixed-layout execution plan payload."""

    encoded = json.dumps(
        [_signature_node_plan_payload(plan) for plan in node_plans],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def normalize_fixed_layout_node_plans(node_plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backfill derived converter metadata for older fixed-layout plan payloads."""

    normalized: list[dict[str, Any]] = []
    for raw_plan in node_plans:
        node_plan = dict(raw_plan)
        raw_specs = [dict(spec) for spec in list(node_plan.get("converter_specs", []))]
        cursor = 0
        normalized_specs: list[dict[str, Any]] = []
        for spec in raw_specs:
            spec_dim = max(1, int(spec["dim"]))
            if "column_start" not in spec:
                spec["column_start"] = int(cursor)
            if "column_end" not in spec:
                spec["column_end"] = int(spec["column_start"]) + spec_dim
            cursor = int(spec["column_end"])
            normalized_specs.append(spec)
        node_plan["converter_specs"] = normalized_specs
        converter_plans = [dict(plan) for plan in list(node_plan.get("converter_plans", []))]
        node_plan["converter_plans"] = converter_plans
        if "converter_groups" not in node_plan:
            node_plan["converter_groups"] = _build_converter_groups(
                normalized_specs,
                converter_plans,
            )
        node_plan["execution_contract"] = str(
            node_plan.get("execution_contract", _FIXED_LAYOUT_EXECUTION_CONTRACT)
        )
        normalized.append(node_plan)
    return normalized


def _signature_node_plan_payload(node_plan: dict[str, Any]) -> dict[str, Any]:
    payload = dict(node_plan)
    payload.pop("converter_groups", None)
    payload.pop("execution_contract", None)
    stripped_specs: list[dict[str, Any]] = []
    for spec in list(payload.get("converter_specs", [])):
        spec_payload = dict(spec)
        spec_payload.pop("column_start", None)
        spec_payload.pop("column_end", None)
        stripped_specs.append(spec_payload)
    payload["converter_specs"] = stripped_specs
    return payload


def _converter_function_signature(plan: dict[str, Any]) -> str | None:
    if "function" not in plan:
        return None
    encoded = json.dumps(plan["function"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=8).hexdigest()


def _converter_group_payload(
    spec_payload: dict[str, Any],
    converter_plan: dict[str, Any],
) -> dict[str, Any]:
    kind = str(spec_payload["kind"])
    if kind in {"num", "target_reg"}:
        return {
            "group_kind": "numeric",
            "group_key": "numeric",
        }
    variant = str(converter_plan["variant"])
    return {
        "group_kind": "categorical",
        "group_key": json.dumps(
            {
                "kind": kind,
                "method": str(converter_plan["method"]),
                "variant": variant,
                "dim": int(spec_payload["dim"]),
                "cardinality": None
                if spec_payload["cardinality"] is None
                else int(spec_payload["cardinality"]),
                "function_signature": (
                    _converter_function_signature(converter_plan)
                    if variant == "center_random_fn"
                    else None
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "method": str(converter_plan["method"]),
        "variant": variant,
        "dim": int(spec_payload["dim"]),
        "cardinality": (
            None if spec_payload["cardinality"] is None else int(spec_payload["cardinality"])
        ),
        "function_signature": (
            _converter_function_signature(converter_plan) if variant == "center_random_fn" else None
        ),
    }


def _build_converter_groups(
    converter_specs: list[dict[str, Any]],
    converter_plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for spec_index, (spec_payload, converter_plan) in enumerate(
        zip(converter_specs, converter_plans, strict=True)
    ):
        payload = _converter_group_payload(spec_payload, converter_plan)
        key = str(payload["group_key"])
        if key not in groups:
            group = dict(payload)
            group["spec_indices"] = []
            ordered_keys.append(key)
            groups[key] = group
        groups[key]["spec_indices"].append(int(spec_index))
    return [groups[key] for key in ordered_keys]


def _batch_standardize(x: torch.Tensor) -> torch.Tensor:
    correction = 1 if int(x.shape[1]) > 1 else 0
    var, mean = torch.var_mean(x, dim=1, keepdim=True, correction=correction)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return (x - mean) / torch.clamp(std, min=1e-6)


@dataclass(slots=True)
class FixedLayoutBatchRng:
    """Chunk-scoped RNG for fixed-layout batched generation."""

    seed: int
    batch_size: int
    device: str
    generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(self.seed))

    def normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device)

    def uniform(self, shape: tuple[int, ...], *, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(
            float(low), float(high), generator=self.generator
        )

    def randint(self, low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(
            int(low),
            int(high),
            shape,
            generator=self.generator,
            device=self.device,
        )

    def log_uniform(self, shape: tuple[int, ...], *, low: float, high: float) -> torch.Tensor:
        samples = self.uniform(
            shape,
            low=math.log(float(low)),
            high=math.log(float(high)),
        )
        return torch.exp(samples)

    def randperm_indices(
        self,
        *,
        length: int,
        sample_size: int,
        leading_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        scores = self.uniform(
            (self.batch_size, *leading_shape, int(length)),
            low=0.0,
            high=1.0,
        )
        return torch.argsort(scores, dim=-1)[..., : int(sample_size)].to(torch.long)

    def categorical(self, probs: torch.Tensor) -> torch.Tensor:
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, 1, generator=self.generator).squeeze(1)
        return sampled.reshape(probs.shape[:-1]).to(torch.long)


def _row_normalize_batch(matrix: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(matrix, dim=-1, keepdim=True)
    return matrix / torch.clamp(norms, min=1e-6)


def _sample_random_weights_batch(
    rng: FixedLayoutBatchRng,
    *,
    dim: int,
    leading_shape: tuple[int, ...] = (),
    parameter_shape: tuple[int, ...] | None = None,
    sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    q: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    normalized_leading_shape = tuple(int(value) for value in leading_shape)
    normalized_parameter_shape = (
        normalized_leading_shape
        if parameter_shape is None
        else tuple(int(value) for value in parameter_shape)
    )
    if len(normalized_parameter_shape) > len(normalized_leading_shape):
        raise ValueError("parameter_shape cannot be longer than leading_shape.")

    leading = (rng.batch_size, *normalized_leading_shape, int(dim))
    q_shape = (rng.batch_size, *normalized_parameter_shape)
    if q is None:
        q_low = 0.1 / math.log(dim + 1.0)
        q = rng.log_uniform(q_shape, low=q_low, high=6.0)
    if sigma is None:
        sigma = rng.log_uniform(q_shape, low=1e-4, high=10.0)
    base_noise = sample_noise_from_spec(
        leading,
        generator=rng.generator,
        device=rng.device,
        noise_spec=noise_spec,
        scale_multiplier=float(sigma_multiplier),
    )
    broadcast_tail = len(normalized_leading_shape) - len(normalized_parameter_shape) + 1
    q_view = q.view(rng.batch_size, *normalized_parameter_shape, *([1] * broadcast_tail))
    sigma_view = sigma.view(rng.batch_size, *normalized_parameter_shape, *([1] * broadcast_tail))
    noise = base_noise * sigma_view
    ranks = torch.arange(1, dim + 1, dtype=torch.float32, device=rng.device).view(
        *([1] * (len(normalized_leading_shape) + 1)),
        dim,
    )
    log_w = (-q_view * torch.log(ranks)) + noise
    log_w = torch.nan_to_num(log_w, nan=0.0, posinf=60.0, neginf=-60.0)
    log_w = torch.clamp(log_w, min=-60.0, max=60.0)
    log_w = log_w - torch.max(log_w, dim=-1, keepdim=True).values
    weights = torch.clamp(torch.exp(log_w), min=1e-12)
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
    perm = torch.argsort(rng.uniform(leading, low=0.0, high=1.0), dim=-1)
    return torch.gather(weights, -1, perm)


def _apply_activation_plan(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    with_standardize: bool,
) -> torch.Tensor:
    y = x.to(torch.float32)
    squeezed = False
    if y.dim() == 2:
        y = y.unsqueeze(0)
        squeezed = True
    leading_shape = tuple(int(dim) for dim in y.shape[:-2])
    if with_standardize:
        y = _batch_standardize(y)
        a = rng.log_uniform((y.shape[0],), low=1.0, high=10.0)
        row_idx = rng.randint(0, y.shape[1], (y.shape[0],))
        offsets = y[torch.arange(y.shape[0], device=y.device), row_idx].unsqueeze(1)
        y = a.view(-1, 1, 1) * (y - offsets)

    if str(plan["mode"]) == "parametric":
        kind = str(plan["kind"])
        if kind == "relu_pow":
            q = rng.log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.pow(torch.clamp(y, min=0.0), q.reshape(*leading_shape, 1, 1))
        elif kind == "signed_pow":
            q = rng.log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.sign(y) * torch.pow(torch.abs(y), q.reshape(*leading_shape, 1, 1))
        elif kind == "inv_pow":
            q = rng.log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.pow(torch.abs(y) + 1e-3, -q.reshape(*leading_shape, 1, 1))
        elif kind == "poly":
            y = torch.pow(y, float(int(plan["poly_power"])))
        else:
            raise ValueError(f"Unknown activation plan kind: {kind!r}")
    else:
        name = str(plan["name"])
        y = _fixed_activation(y.reshape(-1, y.shape[-1]), name).reshape_as(y)

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    if with_standardize:
        y = _batch_standardize(y)
    if squeezed:
        y = y.squeeze(0)
    return y.to(torch.float32)


def _sample_random_matrix_from_plan_batch(
    plan: dict[str, Any],
    *,
    out_dim: int,
    in_dim: int,
    rng: FixedLayoutBatchRng,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    matrix_count: int | None = None,
) -> torch.Tensor:
    leading_shape = () if matrix_count is None else (int(matrix_count),)
    shape = (rng.batch_size, *leading_shape, int(out_dim), int(in_dim))
    kind = str(plan["kind"])
    if kind == "gaussian":
        matrix = sample_noise_from_spec(
            shape,
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    elif kind == "weights":
        g = sample_noise_from_spec(
            shape,
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        q_low = 0.1 / math.log(int(in_dim) + 1.0)
        shared_q = rng.log_uniform((rng.batch_size, *leading_shape), low=q_low, high=6.0)
        shared_sigma = rng.log_uniform((rng.batch_size, *leading_shape), low=1e-4, high=10.0)
        rows = _sample_random_weights_batch(
            rng,
            dim=int(in_dim),
            leading_shape=(*leading_shape, int(out_dim)),
            parameter_shape=leading_shape,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
            q=shared_q,
            sigma=shared_sigma,
        )
        matrix = g * rows
    elif kind == "singular_values":
        d = min(int(out_dim), int(in_dim))
        u_shape = (rng.batch_size, *leading_shape, int(out_dim), d)
        v_shape = (rng.batch_size, *leading_shape, d, int(in_dim))
        u = sample_noise_from_spec(
            u_shape,
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        v = sample_noise_from_spec(
            v_shape,
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        weights = _sample_random_weights_batch(
            rng,
            dim=d,
            leading_shape=leading_shape,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        matrix = torch.matmul(u * weights.unsqueeze(-2), v)
    elif kind == "kernel":
        pts = sample_noise_from_spec(
            (rng.batch_size, *leading_shape, int(out_dim) + int(in_dim), 3),
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        gamma = rng.log_uniform((rng.batch_size, *leading_shape), low=0.1, high=10.0)
        left = pts[..., : int(out_dim), :].unsqueeze(-2)
        right = pts[..., int(out_dim) :, :].unsqueeze(-3)
        dist = torch.norm(left - right, dim=-1)
        kernel = torch.exp(-gamma.unsqueeze(-1).unsqueeze(-1) * dist)
        sign = torch.where(
            rng.uniform(shape, low=0.0, high=1.0) < 0.5,
            -1.0,
            1.0,
        )
        matrix = kernel * sign
    elif kind == "activation":
        base_plan = {"kind": str(plan["base_kind"])}
        matrix = _sample_random_matrix_from_plan_batch(
            base_plan,
            out_dim=int(out_dim),
            in_dim=int(in_dim),
            rng=rng,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            matrix_count=matrix_count,
        )
        matrix = _apply_activation_plan(
            matrix,
            rng,
            plan["activation"],
            with_standardize=False,
        )
        matrix = matrix + 1e-3 * sample_noise_from_spec(
            matrix.shape,
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    else:
        raise ValueError(f"Unknown matrix kind: {kind!r}")

    matrix = matrix + 1e-6 * sample_noise_from_spec(
        matrix.shape,
        generator=rng.generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return _row_normalize_batch(matrix)


def _apply_linear_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    matrices = _sample_random_matrix_from_plan_batch(
        plan["matrix"],
        out_dim=int(out_dim),
        in_dim=int(x.shape[2]),
        rng=rng,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return torch.einsum("bni,boi->bno", x, matrices)


def _apply_quadratic_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    feature_cap = min(int(x.shape[2]), 20)
    if int(x.shape[2]) > feature_cap:
        indices = rng.randperm_indices(
            length=int(x.shape[2]),
            sample_size=feature_cap,
        )
        x_sub = torch.gather(
            x,
            2,
            indices.unsqueeze(1).expand(-1, x.shape[1], -1),
        )
    else:
        x_sub = x
    ones = torch.ones((x_sub.shape[0], x_sub.shape[1], 1), device=x.device, dtype=x_sub.dtype)
    x_aug = torch.cat([x_sub, ones], dim=2)
    matrices = _sample_random_matrix_from_plan_batch(
        plan["matrix"],
        out_dim=int(x_aug.shape[2]),
        in_dim=int(x_aug.shape[2]),
        rng=rng,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        matrix_count=int(out_dim),
    )
    return torch.einsum("bni,boij,bnj->bno", x_aug, matrices, x_aug)


def _sample_unit_ball_batch(
    rng: FixedLayoutBatchRng,
    *,
    n_rows: int,
    dim: int,
) -> torch.Tensor:
    vectors = rng.normal((rng.batch_size, n_rows, dim))
    vectors = vectors / torch.clamp(torch.norm(vectors, dim=2, keepdim=True), min=1e-6)
    radii = rng.uniform((rng.batch_size, n_rows, 1), low=0.0, high=1.0)
    return vectors * torch.pow(radii, 1.0 / max(1, dim))


def _sample_random_points_batch(
    rng: FixedLayoutBatchRng,
    *,
    n_rows: int,
    dim: int,
    base_kind: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    if base_kind == "normal":
        return sample_noise_from_spec(
            (rng.batch_size, n_rows, dim),
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    if base_kind == "uniform":
        return rng.uniform((rng.batch_size, n_rows, dim), low=-1.0, high=1.0)
    if base_kind == "unit_ball":
        return _sample_unit_ball_batch(rng, n_rows=n_rows, dim=dim)

    points = sample_noise_from_spec(
        (rng.batch_size, n_rows, dim),
        generator=rng.generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    weights = _sample_random_weights_batch(
        rng,
        dim=dim,
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    matrices = sample_noise_from_spec(
        (rng.batch_size, dim, dim),
        generator=rng.generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return torch.einsum("bni,bi,bij->bnj", points, weights, matrices.transpose(1, 2))


def _apply_nn_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    y = x
    if bool(plan.get("apply_input_activation")):
        y = _apply_activation_plan(y, rng, plan["input_activation"], with_standardize=True)

    hidden_width = max(1, int(plan["hidden_width"]))
    n_layers = max(1, int(plan["n_layers"]))
    layer_dims = [int(x.shape[2])]
    for _ in range(max(0, n_layers - 1)):
        layer_dims.append(hidden_width)
    layer_dims.append(int(out_dim))

    for layer_index, (din, dout) in enumerate(zip(layer_dims[:-1], layer_dims[1:], strict=True)):
        matrices = _sample_random_matrix_from_plan_batch(
            plan["layer_matrices"][layer_index],
            out_dim=int(dout),
            in_dim=int(din),
            rng=rng,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        y = torch.einsum("bni,boi->bno", y, matrices)
        if layer_index < len(layer_dims) - 2:
            y = _apply_activation_plan(
                y,
                rng,
                plan["hidden_activations"][layer_index],
                with_standardize=True,
            )

    if bool(plan.get("apply_output_activation")):
        y = _apply_activation_plan(y, rng, plan["output_activation"], with_standardize=True)
    return y


def _apply_tree_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size = int(x.shape[0])
    outputs = torch.zeros((batch_size, x.shape[1], out_dim), device=x.device, dtype=torch.float32)
    correction = 1 if int(x.shape[1]) > 1 else 0
    var, _mean = torch.var_mean(x, dim=1, correction=correction)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    probs = torch.clamp(std, min=0.0)
    totals = torch.sum(probs, dim=1, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / max(1, int(probs.shape[1])))
    valid = (
        torch.isfinite(probs).all(dim=1, keepdim=True) & torch.isfinite(totals) & (totals > 1e-12)
    )
    probs = torch.where(valid, probs / torch.clamp(totals, min=1e-12), uniform)
    for depth in plan["depths"]:
        split_dims, thresholds = sample_odt_splits_batch(
            x,
            int(depth),
            rng.generator,
            feature_probs=probs,
        )
        leaf_idx = compute_odt_leaf_indices_batch(x, split_dims, thresholds)
        n_leaves = 2 ** int(depth)
        leaf_vals = sample_noise_from_spec(
            (batch_size, n_leaves, out_dim),
            generator=rng.generator,
            device=str(x.device),
            noise_spec=noise_spec,
        )
        outputs += torch.gather(
            leaf_vals,
            1,
            leaf_idx.unsqueeze(-1).expand(-1, -1, out_dim),
        )
    return outputs / float(max(1, int(plan["n_trees"])))


def _apply_discretization_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    n_centers = min(int(plan["n_centers"]), int(x.shape[1]))
    center_idx = rng.randperm_indices(length=int(x.shape[1]), sample_size=n_centers)
    centers = torch.gather(
        x,
        1,
        center_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    p = rng.log_uniform((rng.batch_size,), low=0.5, high=4.0)
    dist = torch.pow(
        torch.abs(x.unsqueeze(2) - centers.unsqueeze(1)),
        p.view(-1, 1, 1, 1),
    ).sum(dim=3)
    nearest = torch.argmin(dist, dim=2)
    gathered = torch.gather(
        centers,
        1,
        nearest.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    linear_plan = {"family": "linear", "matrix": dict(plan["linear_matrix"])}
    return _apply_linear_batch(
        gathered,
        rng,
        linear_plan,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _sample_radial_ha_batch(
    rng: FixedLayoutBatchRng,
    *,
    n: int,
    a: torch.Tensor,
) -> torch.Tensor:
    u = rng.uniform((rng.batch_size, n), low=0.0, high=1.0)
    return torch.pow(1.0 - u, 1.0 / (1.0 - a.view(-1, 1))) - 1.0


def _apply_gp_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size, _, din = x.shape
    p = 256
    a = rng.log_uniform((rng.batch_size,), low=2.0, high=20.0)

    if str(plan["branch_kind"]) == "ha":
        r = _sample_radial_ha_batch(rng, n=p * din, a=a).view(batch_size, p, din)
        signs = torch.where(
            rng.uniform((batch_size, p, din), low=0.0, high=1.0) < 0.5,
            -1.0,
            1.0,
        )
        omega = r * signs
        x_proj = x
    else:
        z = sample_noise_from_spec(
            (batch_size, p, din),
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        z = z / torch.clamp(torch.norm(z, dim=2, keepdim=True), min=1e-6)
        r = _sample_radial_ha_batch(rng, n=p, a=a)
        omega = z * r.unsqueeze(2)
        weights = _sample_random_weights_batch(
            rng,
            dim=din,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        alpha = rng.log_uniform((batch_size,), low=0.5, high=10.0)
        a_mat = sample_noise_from_spec(
            (batch_size, din, din),
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        matrices = alpha.view(-1, 1, 1) * (weights.unsqueeze(2) * a_mat)
        x_proj = torch.einsum("bni,bij->bnj", x, matrices.transpose(1, 2))

    b = rng.uniform((batch_size, p), low=0.0, high=2.0 * math.pi)
    phi = torch.cos(torch.einsum("bnd,bpd->bnp", x_proj, omega) + b.unsqueeze(1))
    z_out = sample_noise_from_spec(
        (batch_size, out_dim, p),
        generator=rng.generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return torch.einsum("bnp,bop->bno", phi, z_out) / math.sqrt(float(p))


def _apply_em_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: dict[str, Any],
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    m_val = max(2, int(plan["m_val"]))
    base_idx = rng.randint(0, x.shape[1], (rng.batch_size, m_val))
    centers = torch.gather(
        x,
        1,
        base_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    centers = centers + sample_noise_from_spec(
        (rng.batch_size, m_val, x.shape[2]),
        generator=rng.generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    sigma = torch.exp(
        sample_noise_from_spec(
            (rng.batch_size, m_val),
            generator=rng.generator,
            device=rng.device,
            noise_spec=noise_spec,
            scale_multiplier=0.1,
        )
    )
    p_val = rng.log_uniform((rng.batch_size,), low=1.0, high=4.0)
    q_val = rng.log_uniform((rng.batch_size,), low=1.0, high=2.0)
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
        rng,
        linear_plan,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def apply_function_plan_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
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
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "quadratic":
        return _apply_quadratic_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "nn":
        return _apply_nn_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "tree":
        return _apply_tree_batch(y, rng, plan, out_dim=out_dim, noise_spec=noise_spec)
    if family == "discretization":
        return _apply_discretization_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "gp":
        return _apply_gp_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "em":
        return _apply_em_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if family == "product":
        lhs = apply_function_plan_batch(
            y,
            rng,
            plan["lhs"],
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        rhs = apply_function_plan_batch(
            y,
            rng,
            plan["rhs"],
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        return lhs * rhs
    raise ValueError(f"Unsupported fixed-layout function family: {family!r}")


def _apply_numeric_converter_group_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    warp_enabled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    values = y.clone()
    if not bool(torch.any(warp_enabled)):
        return y, values
    a = rng.log_uniform((y.shape[0], y.shape[2]), low=0.2, high=5.0)
    b = rng.log_uniform((y.shape[0], y.shape[2]), low=0.2, high=5.0)
    lo = torch.min(y, dim=1, keepdim=True).values
    hi = torch.max(y, dim=1, keepdim=True).values
    scaled = (y - lo) / torch.clamp(hi - lo, min=1e-6)
    warped = 1.0 - torch.pow(
        1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a.view(y.shape[0], 1, y.shape[2])),
        b.view(y.shape[0], 1, y.shape[2]),
    )
    return torch.where(warp_enabled.view(1, 1, -1), warped, y), values


def _categorical_group_input_views(
    latent: torch.Tensor,
    spec_payloads: list[dict[str, Any]],
) -> torch.Tensor:
    views = [
        latent[:, :, int(spec["column_start"]) : int(spec["column_end"])] for spec in spec_payloads
    ]
    return torch.stack(views, dim=2)


def _gather_group_centers(y: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # y: [B, rows, G, D], indices: [B, G, K] -> [B, G, K, D]
    y_perm = y.permute(0, 2, 1, 3)
    return torch.gather(
        y_perm,
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, y.shape[3]),
    )


def _apply_categorical_group_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    converter_plan: dict[str, Any],
    *,
    n_categories: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    batch_size, n_rows, group_size, width = y.shape
    category_count = max(2, int(n_categories))
    method = str(converter_plan["method"])
    variant = str(converter_plan["variant"])

    centers: torch.Tensor | None = None
    if method == "neighbor":
        n_centers = min(category_count, n_rows)
        center_idx = rng.randperm_indices(
            length=n_rows,
            sample_size=n_centers,
            leading_shape=(group_size,),
        )
        centers = _gather_group_centers(y, center_idx)
        p = rng.log_uniform((batch_size, group_size), low=0.5, high=4.0)
        dist = torch.pow(
            torch.abs(y.permute(0, 2, 1, 3).unsqueeze(3) - centers.unsqueeze(2)),
            p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        ).sum(dim=4)
        labels_bg = torch.argmin(dist, dim=3)
        if n_centers < category_count:
            labels_bg = labels_bg % category_count
        labels = labels_bg.permute(0, 2, 1)
    else:
        if width != category_count:
            projections = rng.normal((batch_size, group_size, width, category_count))
            logits_in = torch.einsum("brgd,bgdc->brgc", y, projections)
        else:
            logits_in = y
        logits_std = _batch_standardize(logits_in)
        a = rng.log_uniform((batch_size, group_size), low=0.1, high=10.0)
        w = rng.uniform((batch_size, group_size, category_count), low=0.0, high=1.0)
        b = torch.log(w + 1e-4)
        logits = a.unsqueeze(1).unsqueeze(-1) * logits_std + b.unsqueeze(1)
        probs = torch.softmax(logits, dim=3)
        labels = rng.categorical(probs)

    if variant == "input":
        out = y
    elif variant == "index_repeat":
        out = labels.unsqueeze(-1).repeat(1, 1, 1, width).to(torch.float32)
    elif variant == "center":
        if centers is None:
            out = y
        else:
            labels_bg = labels.permute(0, 2, 1)
            gathered = torch.gather(
                centers,
                2,
                labels_bg.unsqueeze(-1).expand(-1, -1, -1, width),
            )
            out = gathered.permute(0, 2, 1, 3)
    elif variant == "center_random_fn":
        nested_input = y
        if centers is not None:
            labels_bg = labels.permute(0, 2, 1)
            nested_input = torch.gather(
                centers,
                2,
                labels_bg.unsqueeze(-1).expand(-1, -1, -1, width),
            ).permute(0, 2, 1, 3)
        if group_size != 1:
            raise ValueError("center_random_fn converter groups must have size 1.")
        nested_out = apply_function_plan_batch(
            nested_input[:, :, 0, :],
            rng,
            converter_plan["function"],
            out_dim=width,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        out = nested_out.unsqueeze(2)
    elif variant == "softmax_points":
        points = rng.normal((batch_size, group_size, category_count, width))
        labels_bg = labels.permute(0, 2, 1)
        out = torch.gather(
            points,
            2,
            labels_bg.unsqueeze(-1).expand(-1, -1, -1, width),
        ).permute(0, 2, 1, 3)
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
    rng: FixedLayoutBatchRng,
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_dim = int(node_plan["latent"]["total_dim"])
    if parent_data:
        if str(node_plan["combine_kind"]) == "concat":
            concat = torch.cat(parent_data, dim=2)
            latent = apply_function_plan_batch(
                concat,
                rng,
                node_plan["function"],
                out_dim=total_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
        else:
            transformed = [
                apply_function_plan_batch(
                    parent_tensor,
                    rng,
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
            rng,
            n_rows=n_rows,
            dim=total_dim,
            base_kind=str(node_plan["base_kind"]),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        latent = apply_function_plan_batch(
            base,
            rng,
            node_plan["function"],
            out_dim=total_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _batch_standardize(latent)

    weights = _sample_random_weights_batch(
        rng,
        dim=int(latent.shape[2]),
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    latent = latent * weights.unsqueeze(1)
    mean_l2 = torch.mean(torch.norm(latent, dim=2), dim=1)
    latent = latent / torch.clamp(mean_l2.view(-1, 1, 1), min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    converter_specs = node_plan["converter_specs"]
    converter_plans = node_plan["converter_plans"]
    spec_by_index = {int(idx): converter_specs[int(idx)] for idx in range(len(converter_specs))}
    plan_by_index = {int(idx): converter_plans[int(idx)] for idx in range(len(converter_plans))}
    for group in node_plan.get("converter_groups", []):
        spec_indices = [int(value) for value in list(group["spec_indices"])]
        if str(group["group_kind"]) == "numeric":
            spec_payloads = [spec_by_index[idx] for idx in spec_indices]
            columns = [int(spec["column_start"]) for spec in spec_payloads]
            views = latent[:, :, columns]
            warp_enabled = torch.as_tensor(
                [bool(plan_by_index[idx].get("warp_enabled")) for idx in spec_indices],
                dtype=torch.bool,
                device=latent.device,
            )
            x_prime, values = _apply_numeric_converter_group_batch(
                views,
                rng,
                warp_enabled,
            )
            latent[:, :, columns] = x_prime
            for local_index, spec_payload in enumerate(spec_payloads):
                extracted[str(spec_payload["key"])] = values[:, :, local_index]
            continue

        spec_payloads = [spec_by_index[idx] for idx in spec_indices]
        first_plan = plan_by_index[spec_indices[0]]
        view = _categorical_group_input_views(latent, spec_payloads)
        x_prime, values = _apply_categorical_group_batch(
            view,
            rng,
            first_plan,
            n_categories=max(2, int(spec_payloads[0]["cardinality"] or 2)),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        for local_index, spec_payload in enumerate(spec_payloads):
            start = int(spec_payload["column_start"])
            end = int(spec_payload["column_end"])
            spec_out = x_prime[:, :, local_index, :]
            if int(spec_out.shape[2]) != (end - start):
                if int(spec_out.shape[2]) > (end - start):
                    spec_out = spec_out[:, :, : (end - start)]
                else:
                    spec_out = torch.nn.functional.pad(
                        spec_out, (0, (end - start) - int(spec_out.shape[2]))
                    )
            latent[:, :, start:end] = spec_out
            extracted[str(spec_payload["key"])] = values[:, :, local_index]

    scale = rng.log_uniform((rng.batch_size,), low=0.1, high=10.0)
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
    normalized_node_plans = normalize_fixed_layout_node_plans(node_plans)
    batch_seed = SeedManager(int(dataset_seeds[0])).child("fixed_layout_chunk", batch_size)
    rng = FixedLayoutBatchRng(seed=batch_seed, batch_size=batch_size, device=device)

    node_outputs: list[torch.Tensor | None] = [None] * int(layout.graph_nodes)
    feature_values: list[torch.Tensor | None] = [None] * num_features
    target_values: torch.Tensor | None = None
    aux_meta_batch = [{"filter": {"mode": "deferred", "status": "not_run"}} for _ in dataset_seeds]

    for node_index, node_plan in enumerate(normalized_node_plans):
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
            rng=rng,
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
                feature_tensor = rng.randint(0, cardinality, (batch_size, n_rows))
            else:
                feature_tensor = sample_noise_from_spec(
                    (batch_size, n_rows),
                    generator=rng.generator,
                    device=device,
                    noise_spec=noise_spec,
                )
        x[:, :, feature_index] = feature_tensor.to(dtype)

    if target_values is None:
        if str(config.dataset.task) == "classification":
            y = rng.randint(0, int(layout.n_classes), (batch_size, n_rows)).to(torch.int64)
        else:
            y = sample_noise_from_spec(
                (batch_size, n_rows),
                generator=rng.generator,
                device=device,
                noise_spec=noise_spec,
            ).to(dtype)
    else:
        if str(config.dataset.task) == "classification":
            y = target_values.to(torch.int64) % int(layout.n_classes)
        else:
            y = target_values.to(dtype)
    return x, y, aux_meta_batch
