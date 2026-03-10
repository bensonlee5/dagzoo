"""Batched fixed-layout execution-plan sampling and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.execution_semantics import (
    sample_node_plan,
)
from dagzoo.core.layout import _build_node_specs
from dagzoo.core.fixed_layout_plan_types import (
    ActivationMatrixPlan,
    CategoricalConverterPlan,
    ConcatNodeSource,
    DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedLayoutActivationPlan,
    FixedLayoutConverterSpec,
    FixedLayoutExecutionPlan,
    FixedLayoutFunctionPlan,
    FixedLayoutMatrixPlan,
    FixedLayoutMatrixBaseKind,
    FixedLayoutNodePlan,
    FixedLayoutRootBaseKind,
    GaussianMatrixPlan,
    GpFunctionPlan,
    KernelMatrixPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterGroup,
    NumericConverterPlan,
    ParametricActivationPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    SingularValuesMatrixPlan,
    StackedNodeSource,
    TreeFunctionPlan,
    WeightsMatrixPlan,
    fixed_layout_signature_payloads,
)
from dagzoo.core.layout_types import AggregationKind, LayoutPlan
from dagzoo.core.trees import (
    compute_odt_leaf_indices_batch,
    sample_odt_splits_batch,
)
from dagzoo.functions import activations as activations_module
from dagzoo.rng import KeyedRng
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

_PAIRWISE_CENTER_BLOCK_SIZE = 32

_FIXED_LAYOUT_EXECUTION_CONTRACT = DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT


def _parent_index_lists(layout: LayoutPlan) -> list[list[int]]:
    adjacency = layout.adjacency
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device="cpu")
    return [
        sorted(int(parent_index) for parent_index in torch.where(adjacency[:, node_index])[0])
        for node_index in range(int(layout.graph_nodes))
    ]


def build_fixed_layout_execution_plan(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    plan_seed: int,
    mechanism_logit_tilt: float,
) -> FixedLayoutExecutionPlan:
    """Build one reusable per-node execution-plan payload for fixed-layout batches."""

    plan_root = KeyedRng(int(plan_seed))
    task = str(config.dataset.task)
    node_plans: list[FixedLayoutNodePlan] = []
    for node_index, parent_indices in enumerate(_parent_index_lists(layout)):
        spec_root = plan_root.keyed("node_spec", node_index)
        node_root = plan_root.keyed("node_plan", node_index)
        converter_specs = _build_node_specs(node_index, layout, task, spec_root)
        node_plans.append(
            sample_node_plan(
                node_index=int(node_index),
                parent_indices=parent_indices,
                converter_specs=converter_specs,
                keyed_rng=node_root,
                device="cpu",
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=config.mechanism.function_family_mix,
            )
        )
    return FixedLayoutExecutionPlan(
        node_plans=tuple(node_plans),
        execution_contract=_FIXED_LAYOUT_EXECUTION_CONTRACT,
    )


def fixed_layout_plan_signature(execution_plan: FixedLayoutExecutionPlan) -> str:
    """Return a deterministic signature for one fixed-layout execution plan payload."""

    encoded = json.dumps(
        fixed_layout_signature_payloads(execution_plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _batch_standardize(x: torch.Tensor) -> torch.Tensor:
    correction = 1 if int(x.shape[1]) > 1 else 0
    var, mean = torch.var_mean(x, dim=1, keepdim=True, correction=correction)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return (x - mean) / torch.clamp(std, min=1e-6)


def _sanitize_and_batch_standardize(x: torch.Tensor) -> torch.Tensor:
    y = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    return _batch_standardize(y)


def _aggregate_batch_incrementally(
    aggregate: torch.Tensor,
    transformed: torch.Tensor,
    *,
    aggregation_kind: AggregationKind,
) -> torch.Tensor:
    if aggregation_kind == "sum":
        return aggregate + transformed
    if aggregation_kind == "product":
        return aggregate * transformed
    if aggregation_kind == "max":
        return torch.maximum(aggregate, transformed)
    raise ValueError(f"Unknown aggregation kind: {aggregation_kind!r}")


def _aggregate_parent_outputs_batch(
    stacked: torch.Tensor,
    *,
    aggregation_kind: AggregationKind,
) -> torch.Tensor:
    if aggregation_kind == "sum":
        return torch.sum(stacked, dim=2)
    if aggregation_kind == "product":
        return torch.prod(stacked, dim=2)
    if aggregation_kind == "max":
        return torch.max(stacked, dim=2).values
    if aggregation_kind == "logsumexp":
        return torch.logsumexp(stacked, dim=2)
    raise ValueError(f"Unknown aggregation kind: {aggregation_kind!r}")


def _flatten_leading_dims(
    x: torch.Tensor, *, trailing_dims: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    leading_shape = tuple(int(dim) for dim in x.shape[:-trailing_dims])
    flat = max(1, math.prod(leading_shape))
    return x.reshape(flat, *x.shape[-trailing_dims:]), leading_shape


def _nearest_lp_center_indices(
    x: torch.Tensor,
    centers: torch.Tensor,
    *,
    p: torch.Tensor,
    block_size: int = _PAIRWISE_CENTER_BLOCK_SIZE,
) -> torch.Tensor:
    x_flat, leading_shape = _flatten_leading_dims(x, trailing_dims=2)
    centers_flat, _ = _flatten_leading_dims(centers, trailing_dims=2)
    p_flat = p.reshape(-1)

    best_distance: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, int(centers_flat.shape[1]), max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), int(centers_flat.shape[1]))
        center_block = centers_flat[:, start:stop, :]
        diff = torch.abs(x_flat.unsqueeze(2) - center_block.unsqueeze(1))
        block_distances = torch.pow(diff, p_flat.view(-1, 1, 1, 1)).sum(dim=3)
        local_distance, local_indices = torch.min(block_distances, dim=2)
        local_indices = local_indices + int(start)
        if best_distance is None or best_indices is None:
            best_distance = local_distance
            best_indices = local_indices
            continue
        use_local = local_distance < best_distance
        best_distance = torch.where(use_local, local_distance, best_distance)
        best_indices = torch.where(use_local, local_indices, best_indices)

    if best_indices is None:
        raise RuntimeError("Expected at least one center when computing nearest center indices.")
    return best_indices.reshape(*leading_shape, int(x.shape[-2]))


def _lp_distances_to_centers(
    x: torch.Tensor,
    centers: torch.Tensor,
    *,
    p: torch.Tensor,
    take_root: bool,
    block_size: int = _PAIRWISE_CENTER_BLOCK_SIZE,
) -> torch.Tensor:
    x_flat, leading_shape = _flatten_leading_dims(x, trailing_dims=2)
    centers_flat, _ = _flatten_leading_dims(centers, trailing_dims=2)
    p_flat = p.reshape(-1)
    blocks: list[torch.Tensor] = []
    for start in range(0, int(centers_flat.shape[1]), max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), int(centers_flat.shape[1]))
        center_block = centers_flat[:, start:stop, :]
        diff = torch.abs(x_flat.unsqueeze(2) - center_block.unsqueeze(1))
        block_distances = torch.pow(diff, p_flat.view(-1, 1, 1, 1)).sum(dim=3)
        if take_root:
            block_distances = torch.pow(
                block_distances,
                p_flat.reciprocal().view(-1, 1, 1),
            )
        blocks.append(block_distances)
    if not blocks:
        raise RuntimeError("Expected at least one center block when computing distances.")
    distances = torch.cat(blocks, dim=2)
    return distances.reshape(*leading_shape, int(x.shape[-2]), int(centers.shape[-2]))


@dataclass(slots=True)
class FixedLayoutBatchRng:
    """Chunk-scoped RNG for fixed-layout batched generation."""

    seed: int | None
    batch_size: int
    device: str
    generator: torch.Generator | None = field(default=None, repr=False)
    keyed_root: KeyedRng | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.keyed_root is None and self.seed is not None:
            self.keyed_root = KeyedRng(int(self.seed))
        if self.generator is None:
            if self.seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(self.seed))
                self.generator = generator
            elif self.keyed_root is not None:
                self.generator = self.keyed_root.torch_rng(device=self.device)
            else:
                raise ValueError("FixedLayoutBatchRng requires either seed or generator.")

    @classmethod
    def from_generator(
        cls,
        generator: torch.Generator,
        *,
        batch_size: int,
        device: str,
    ) -> FixedLayoutBatchRng:
        return cls(
            seed=None,
            batch_size=int(batch_size),
            device=device,
            generator=generator,
        )

    @classmethod
    def from_keyed_rng(
        cls,
        keyed_rng: KeyedRng,
        *,
        batch_size: int,
        device: str,
    ) -> FixedLayoutBatchRng:
        return cls(
            seed=None,
            batch_size=int(batch_size),
            device=device,
            keyed_root=keyed_rng,
        )

    @property
    def torch_generator(self) -> torch.Generator:
        if self.generator is None:
            raise RuntimeError("FixedLayoutBatchRng generator was not initialized.")
        return self.generator

    def keyed(self, *components: str | int) -> FixedLayoutBatchRng:
        if self.keyed_root is None:
            return self
        return FixedLayoutBatchRng.from_keyed_rng(
            self.keyed_root.keyed(*components),
            batch_size=self.batch_size,
            device=self.device,
        )

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
        return torch.topk(
            scores,
            k=int(sample_size),
            dim=-1,
            largest=False,
            sorted=True,
        ).indices.to(torch.long)

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
        q = rng.keyed("q").log_uniform(q_shape, low=q_low, high=6.0)
    if sigma is None:
        sigma = rng.keyed("sigma").log_uniform(q_shape, low=1e-4, high=10.0)
    base_noise = sample_noise_from_spec(
        leading,
        generator=rng.keyed("noise").torch_generator,
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
    perm = torch.argsort(rng.keyed("perm").uniform(leading, low=0.0, high=1.0), dim=-1)
    return torch.gather(weights, -1, perm)


def _apply_activation_plan(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutActivationPlan,
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
        a = rng.keyed("standardize_scale").log_uniform((y.shape[0],), low=1.0, high=10.0)
        row_idx = rng.keyed("standardize_row_index").randint(0, y.shape[1], (y.shape[0],))
        offsets = y[torch.arange(y.shape[0], device=y.device), row_idx].unsqueeze(1)
        y = a.view(-1, 1, 1) * (y - offsets)

    if isinstance(plan, ParametricActivationPlan):
        kind = str(plan.kind)
        if kind == "relu_pow":
            q = rng.keyed("relu_pow").log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.pow(torch.clamp(y, min=0.0), q.reshape(*leading_shape, 1, 1))
        elif kind == "signed_pow":
            q = rng.keyed("signed_pow").log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.sign(y) * torch.pow(torch.abs(y), q.reshape(*leading_shape, 1, 1))
        elif kind == "inv_pow":
            q = rng.keyed("inv_pow").log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.pow(torch.abs(y) + 1e-3, -q.reshape(*leading_shape, 1, 1))
        elif kind == "poly":
            if plan.poly_power is None:
                raise ValueError("poly activation plan requires poly_power.")
            y = torch.pow(y, float(int(plan.poly_power)))
        else:
            raise ValueError(f"Unknown activation plan kind: {kind!r}")
    else:
        y = activations_module._fixed_activation(
            y.reshape(-1, y.shape[-1]),
            str(plan.name),
        ).reshape_as(y)

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    if with_standardize:
        y = _batch_standardize(y)
    if squeezed:
        y = y.squeeze(0)
    return y.to(torch.float32)


def _base_matrix_plan(kind: FixedLayoutMatrixBaseKind) -> FixedLayoutMatrixPlan:
    if kind == "gaussian":
        return GaussianMatrixPlan()
    if kind == "weights":
        return WeightsMatrixPlan()
    if kind == "singular_values":
        return SingularValuesMatrixPlan()
    if kind == "kernel":
        return KernelMatrixPlan()
    raise ValueError(f"Unsupported fixed-layout matrix base kind: {kind!r}")


def _sample_random_matrix_from_plan_batch(
    plan: FixedLayoutMatrixPlan,
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
    if isinstance(plan, GaussianMatrixPlan):
        matrix = sample_noise_from_spec(
            shape,
            generator=rng.keyed("gaussian").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    elif isinstance(plan, WeightsMatrixPlan):
        g = sample_noise_from_spec(
            shape,
            generator=rng.keyed("gaussian").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        q_low = 0.1 / math.log(int(in_dim) + 1.0)
        shared_q = rng.keyed("shared_q").log_uniform(
            (rng.batch_size, *leading_shape), low=q_low, high=6.0
        )
        shared_sigma = rng.keyed("shared_sigma").log_uniform(
            (rng.batch_size, *leading_shape), low=1e-4, high=10.0
        )
        rows = _sample_random_weights_batch(
            rng.keyed("row_weights"),
            dim=int(in_dim),
            leading_shape=(*leading_shape, int(out_dim)),
            parameter_shape=leading_shape,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
            q=shared_q,
            sigma=shared_sigma,
        )
        matrix = g * rows
    elif isinstance(plan, SingularValuesMatrixPlan):
        d = min(int(out_dim), int(in_dim))
        u_shape = (rng.batch_size, *leading_shape, int(out_dim), d)
        v_shape = (rng.batch_size, *leading_shape, d, int(in_dim))
        u = sample_noise_from_spec(
            u_shape,
            generator=rng.keyed("u").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        v = sample_noise_from_spec(
            v_shape,
            generator=rng.keyed("v").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        weights = _sample_random_weights_batch(
            rng.keyed("singular_values"),
            dim=d,
            leading_shape=leading_shape,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        matrix = torch.matmul(u * weights.unsqueeze(-2), v)
    elif isinstance(plan, KernelMatrixPlan):
        pts = sample_noise_from_spec(
            (rng.batch_size, *leading_shape, int(out_dim) + int(in_dim), 3),
            generator=rng.keyed("points").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        gamma = rng.keyed("gamma").log_uniform((rng.batch_size, *leading_shape), low=0.1, high=10.0)
        left = pts[..., : int(out_dim), :].unsqueeze(-2)
        right = pts[..., int(out_dim) :, :].unsqueeze(-3)
        dist = torch.norm(left - right, dim=-1)
        kernel = torch.exp(-gamma.unsqueeze(-1).unsqueeze(-1) * dist)
        sign = torch.where(
            rng.keyed("sign").uniform(shape, low=0.0, high=1.0) < 0.5,
            -1.0,
            1.0,
        )
        matrix = kernel * sign
    elif isinstance(plan, ActivationMatrixPlan):
        matrix = _sample_random_matrix_from_plan_batch(
            _base_matrix_plan(plan.base_kind),
            out_dim=int(out_dim),
            in_dim=int(in_dim),
            rng=rng.keyed("base"),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            matrix_count=matrix_count,
        )
        matrix = _apply_activation_plan(
            matrix,
            rng.keyed("activation"),
            plan.activation,
            with_standardize=False,
        )
        matrix = matrix + 1e-3 * sample_noise_from_spec(
            matrix.shape,
            generator=rng.keyed("activation_noise").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    else:
        raise ValueError(f"Unknown matrix plan: {plan!r}")

    matrix = matrix + 1e-6 * sample_noise_from_spec(
        matrix.shape,
        generator=rng.keyed("jitter").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return _row_normalize_batch(matrix)


def _apply_linear_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: LinearFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    matrices = _sample_random_matrix_from_plan_batch(
        plan.matrix,
        out_dim=int(out_dim),
        in_dim=int(x.shape[2]),
        rng=rng.keyed("matrix"),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return torch.einsum("bni,boi->bno", x, matrices)


def _apply_quadratic_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: QuadraticFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    feature_cap = min(int(x.shape[2]), 20)
    if int(x.shape[2]) > feature_cap:
        indices = rng.keyed("feature_subset").randperm_indices(
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
        plan.matrix,
        out_dim=int(x_aug.shape[2]),
        in_dim=int(x_aug.shape[2]),
        rng=rng.keyed("matrix"),
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
    vectors = rng.keyed("vectors").normal((rng.batch_size, n_rows, dim))
    vectors = vectors / torch.clamp(torch.norm(vectors, dim=2, keepdim=True), min=1e-6)
    radii = rng.keyed("radii").uniform((rng.batch_size, n_rows, 1), low=0.0, high=1.0)
    return vectors * torch.pow(radii, 1.0 / max(1, dim))


def _sample_random_points_batch(
    rng: FixedLayoutBatchRng,
    *,
    n_rows: int,
    dim: int,
    base_kind: FixedLayoutRootBaseKind,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    if base_kind == "normal":
        return sample_noise_from_spec(
            (rng.batch_size, n_rows, dim),
            generator=rng.keyed("normal").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    if base_kind == "uniform":
        return rng.keyed("uniform").uniform((rng.batch_size, n_rows, dim), low=-1.0, high=1.0)
    if base_kind == "unit_ball":
        return _sample_unit_ball_batch(rng.keyed("unit_ball"), n_rows=n_rows, dim=dim)

    points = sample_noise_from_spec(
        (rng.batch_size, n_rows, dim),
        generator=rng.keyed("points").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    weights = _sample_random_weights_batch(
        rng.keyed("weights"),
        dim=dim,
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    matrices = sample_noise_from_spec(
        (rng.batch_size, dim, dim),
        generator=rng.keyed("matrix").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return torch.einsum("bni,bi,bij->bnj", points, weights, matrices.transpose(1, 2))


def _apply_nn_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: NeuralNetFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    y = x
    if plan.input_activation is not None:
        y = _apply_activation_plan(
            y,
            rng.keyed("input_activation"),
            plan.input_activation,
            with_standardize=True,
        )

    hidden_width = max(1, int(plan.hidden_width))
    n_layers = max(1, int(plan.n_layers))
    layer_dims = [int(x.shape[2])]
    for _ in range(max(0, n_layers - 1)):
        layer_dims.append(hidden_width)
    layer_dims.append(int(out_dim))

    for layer_index, (din, dout) in enumerate(zip(layer_dims[:-1], layer_dims[1:], strict=True)):
        matrices = _sample_random_matrix_from_plan_batch(
            plan.layer_matrices[layer_index],
            out_dim=int(dout),
            in_dim=int(din),
            rng=rng.keyed("layer_matrix", layer_index),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        y = torch.einsum("bni,boi->bno", y, matrices)
        if layer_index < len(layer_dims) - 2:
            y = _apply_activation_plan(
                y,
                rng.keyed("hidden_activation", layer_index),
                plan.hidden_activations[layer_index],
                with_standardize=True,
            )

    if plan.output_activation is not None:
        y = _apply_activation_plan(
            y,
            rng.keyed("output_activation"),
            plan.output_activation,
            with_standardize=True,
        )
    return y


def _apply_tree_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: TreeFunctionPlan,
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
    for tree_index, depth in enumerate(plan.depths):
        tree_rng = rng.keyed("tree", tree_index)
        split_dims, thresholds = sample_odt_splits_batch(
            x,
            int(depth),
            tree_rng.keyed("splits").torch_generator,
            feature_probs=probs,
        )
        leaf_idx = compute_odt_leaf_indices_batch(x, split_dims, thresholds)
        n_leaves = 2 ** int(depth)
        leaf_vals = sample_noise_from_spec(
            (batch_size, n_leaves, out_dim),
            generator=tree_rng.keyed("leaf_values").torch_generator,
            device=str(x.device),
            noise_spec=noise_spec,
        )
        outputs += torch.gather(
            leaf_vals,
            1,
            leaf_idx.unsqueeze(-1).expand(-1, -1, out_dim),
        )
    return outputs / float(max(1, int(plan.n_trees)))


def _apply_discretization_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: DiscretizationFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    n_centers = min(int(plan.n_centers), int(x.shape[1]))
    center_idx = rng.keyed("center_index").randperm_indices(
        length=int(x.shape[1]),
        sample_size=n_centers,
    )
    centers = torch.gather(
        x,
        1,
        center_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    p = rng.keyed("lp_norm").log_uniform((rng.batch_size,), low=0.5, high=4.0)
    nearest = _nearest_lp_center_indices(x, centers, p=p)
    gathered = torch.gather(
        centers,
        1,
        nearest.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    return _apply_linear_batch(
        gathered,
        rng.keyed("linear"),
        LinearFunctionPlan(matrix=plan.linear_matrix),
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
    u = rng.keyed("u").uniform((rng.batch_size, n), low=0.0, high=1.0)
    return torch.pow(1.0 - u, 1.0 / (1.0 - a.view(-1, 1))) - 1.0


def _apply_gp_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: GpFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size, _, din = x.shape
    p = 256
    a = rng.keyed("a").log_uniform((rng.batch_size,), low=2.0, high=20.0)

    if str(plan.branch_kind) == "ha":
        r = _sample_radial_ha_batch(rng.keyed("ha_radius"), n=p * din, a=a).view(batch_size, p, din)
        signs = torch.where(
            rng.keyed("ha_sign").uniform((batch_size, p, din), low=0.0, high=1.0) < 0.5,
            -1.0,
            1.0,
        )
        omega = r * signs
        x_proj = x
    else:
        z = sample_noise_from_spec(
            (batch_size, p, din),
            generator=rng.keyed("projected_direction").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        z = z / torch.clamp(torch.norm(z, dim=2, keepdim=True), min=1e-6)
        r = _sample_radial_ha_batch(rng.keyed("projected_radius"), n=p, a=a)
        omega = z * r.unsqueeze(2)
        weights = _sample_random_weights_batch(
            rng.keyed("weights"),
            dim=din,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        alpha = rng.keyed("alpha").log_uniform((batch_size,), low=0.5, high=10.0)
        a_mat = sample_noise_from_spec(
            (batch_size, din, din),
            generator=rng.keyed("matrix").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        matrices = alpha.view(-1, 1, 1) * (weights.unsqueeze(2) * a_mat)
        x_proj = torch.einsum("bni,bij->bnj", x, matrices.transpose(1, 2))

    b = rng.keyed("phase").uniform((batch_size, p), low=0.0, high=2.0 * math.pi)
    phi = torch.cos(torch.einsum("bnd,bpd->bnp", x_proj, omega) + b.unsqueeze(1))
    z_out = sample_noise_from_spec(
        (batch_size, out_dim, p),
        generator=rng.keyed("output_matrix").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return torch.einsum("bnp,bop->bno", phi, z_out) / math.sqrt(float(p))


def _apply_em_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: EmFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    m_val = max(2, int(plan.m_val))
    base_idx = rng.keyed("base_index").randint(0, x.shape[1], (rng.batch_size, m_val))
    centers = torch.gather(
        x,
        1,
        base_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    centers = centers + sample_noise_from_spec(
        (rng.batch_size, m_val, x.shape[2]),
        generator=rng.keyed("center_noise").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    sigma = torch.exp(
        sample_noise_from_spec(
            (rng.batch_size, m_val),
            generator=rng.keyed("sigma").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
            scale_multiplier=0.1,
        )
    )
    p_val = rng.keyed("p_val").log_uniform((rng.batch_size,), low=1.0, high=4.0)
    q_val = rng.keyed("q_val").log_uniform((rng.batch_size,), low=1.0, high=2.0)
    dist_p = _lp_distances_to_centers(
        x,
        centers,
        p=p_val,
        take_root=True,
    )
    logits = -0.5 * torch.log(2.0 * math.pi * sigma**2).unsqueeze(1) - torch.pow(
        dist_p / torch.clamp(sigma.unsqueeze(1), min=1e-6),
        q_val.view(-1, 1, 1),
    )
    probs = torch.softmax(logits, dim=2)
    return _apply_linear_batch(
        probs,
        rng.keyed("linear"),
        LinearFunctionPlan(matrix=plan.linear_matrix),
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def apply_function_plan_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    standardize_input: bool = True,
) -> torch.Tensor:
    """Apply one frozen function-family plan across a batch of datasets."""

    y = x.to(torch.float32)
    if standardize_input:
        y = _batch_standardize(y)
    if isinstance(plan, LinearFunctionPlan):
        return _apply_linear_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, QuadraticFunctionPlan):
        return _apply_quadratic_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, NeuralNetFunctionPlan):
        return _apply_nn_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, TreeFunctionPlan):
        return _apply_tree_batch(y, rng, plan, out_dim=out_dim, noise_spec=noise_spec)
    if isinstance(plan, DiscretizationFunctionPlan):
        return _apply_discretization_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, GpFunctionPlan):
        return _apply_gp_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, EmFunctionPlan):
        return _apply_em_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, ProductFunctionPlan):
        lhs = apply_function_plan_batch(
            y,
            rng.keyed("product", "lhs"),
            plan.lhs,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            standardize_input=True,
        )
        rhs = apply_function_plan_batch(
            y,
            rng.keyed("product", "rhs"),
            plan.rhs,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            standardize_input=True,
        )
        return lhs * rhs
    raise ValueError(f"Unsupported fixed-layout function plan: {plan!r}")


def apply_numeric_converter_plan_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: NumericConverterPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one vector-valued numeric converter plan across a batch of datasets."""

    y = x.to(torch.float32)
    values = y[:, :, 0].clone()
    if not plan.warp_enabled:
        return y, values

    a = rng.keyed("a").log_uniform((y.shape[0],), low=0.2, high=5.0)
    b = rng.keyed("b").log_uniform((y.shape[0],), low=0.2, high=5.0)
    lo = torch.min(y, dim=1, keepdim=True).values
    hi = torch.max(y, dim=1, keepdim=True).values
    scaled = (y - lo) / torch.clamp(hi - lo, min=1e-6)
    warped = 1.0 - torch.pow(
        1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a.view(y.shape[0], 1, 1)),
        b.view(y.shape[0], 1, 1),
    )
    return warped, values


def _apply_numeric_converter_group_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    warp_enabled: torch.Tensor,
    *,
    spec_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    values = y.clone()
    if not bool(torch.any(warp_enabled)):
        return y, values
    if spec_indices is None:
        a = rng.keyed("a").log_uniform((y.shape[0], y.shape[2]), low=0.2, high=5.0)
        b = rng.keyed("b").log_uniform((y.shape[0], y.shape[2]), low=0.2, high=5.0)
    else:
        if len(spec_indices) != int(y.shape[2]):
            raise ValueError("spec_indices must align with numeric converter group width.")
        a = torch.stack(
            [
                rng.keyed("converter", spec_index)
                .keyed("a")
                .log_uniform(
                    (y.shape[0],),
                    low=0.2,
                    high=5.0,
                )
                for spec_index in spec_indices
            ],
            dim=1,
        )
        b = torch.stack(
            [
                rng.keyed("converter", spec_index)
                .keyed("b")
                .log_uniform(
                    (y.shape[0],),
                    low=0.2,
                    high=5.0,
                )
                for spec_index in spec_indices
            ],
            dim=1,
        )
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
    spec_payloads: list[FixedLayoutConverterSpec],
) -> torch.Tensor:
    views = [latent[:, :, int(spec.column_start) : int(spec.column_end)] for spec in spec_payloads]
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
    converter_plan: CategoricalConverterPlan,
    *,
    n_categories: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    spec_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    batch_size, n_rows, group_size, width = y.shape
    category_count = max(2, int(n_categories))
    method = str(converter_plan.method)
    variant = str(converter_plan.variant)
    if spec_indices is not None and len(spec_indices) != int(group_size):
        raise ValueError("spec_indices must align with categorical converter group width.")

    centers: torch.Tensor | None = None
    if method == "neighbor":
        n_centers = min(category_count, n_rows)
        if spec_indices is None:
            center_idx = rng.keyed("center_index").randperm_indices(
                length=n_rows,
                sample_size=n_centers,
                leading_shape=(group_size,),
            )
        else:
            center_idx = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("center_index")
                    .randperm_indices(
                        length=n_rows,
                        sample_size=n_centers,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        centers = _gather_group_centers(y, center_idx)
        if spec_indices is None:
            p = rng.keyed("lp_norm").log_uniform((batch_size, group_size), low=0.5, high=4.0)
        else:
            p = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("lp_norm")
                    .log_uniform(
                        (batch_size,),
                        low=0.5,
                        high=4.0,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        labels_bg = _nearest_lp_center_indices(
            y.permute(0, 2, 1, 3),
            centers,
            p=p,
        )
        if n_centers < category_count:
            labels_bg = labels_bg % category_count
        labels = labels_bg.permute(0, 2, 1)
    else:
        if width != category_count:
            if spec_indices is None:
                projections = rng.keyed("projections").normal(
                    (batch_size, group_size, width, category_count)
                )
            else:
                projections = torch.stack(
                    [
                        rng.keyed("converter", spec_index)
                        .keyed("projections")
                        .normal((batch_size, width, category_count))
                        for spec_index in spec_indices
                    ],
                    dim=1,
                )
            logits_in = torch.einsum("brgd,bgdc->brgc", y, projections)
        else:
            logits_in = y
        logits_std = _batch_standardize(logits_in)
        if spec_indices is None:
            a = rng.keyed("softmax_scale").log_uniform((batch_size, group_size), low=0.1, high=10.0)
            w = rng.keyed("softmax_bias").uniform(
                (batch_size, group_size, category_count),
                low=0.0,
                high=1.0,
            )
        else:
            a = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("softmax_scale")
                    .log_uniform(
                        (batch_size,),
                        low=0.1,
                        high=10.0,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
            w = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("softmax_bias")
                    .uniform(
                        (batch_size, category_count),
                        low=0.0,
                        high=1.0,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        b = torch.log(w + 1e-4)
        logits = a.unsqueeze(1).unsqueeze(-1) * logits_std + b.unsqueeze(1)
        probs = torch.softmax(logits, dim=3)
        if spec_indices is None:
            labels = rng.keyed("labels").categorical(probs)
        else:
            labels = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("labels")
                    .categorical(probs[:, :, group_index, :])
                    for group_index, spec_index in enumerate(spec_indices)
                ],
                dim=2,
            )

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
        nested_function = converter_plan.function
        if nested_function is None:
            raise ValueError("center_random_fn converter plan requires a nested function.")
        nested_rng = (
            rng.keyed("nested_function")
            if spec_indices is None
            else rng.keyed("converter", spec_indices[0]).keyed("nested_function")
        )
        nested_out = apply_function_plan_batch(
            nested_input[:, :, 0, :],
            nested_rng,
            nested_function,
            out_dim=width,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        out = nested_out.unsqueeze(2)
    elif variant == "softmax_points":
        if spec_indices is None:
            points = rng.keyed("softmax_points").normal(
                (batch_size, group_size, category_count, width)
            )
        else:
            points = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("softmax_points")
                    .normal((batch_size, category_count, width))
                    for spec_index in spec_indices
                ],
                dim=1,
            )
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
    config: GeneratorConfig | None,
    node_plan: FixedLayoutNodePlan,
    parent_data: list[torch.Tensor],
    *,
    n_rows: int,
    rng: FixedLayoutBatchRng,
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _ = config
    _ = device
    total_dim = int(node_plan.latent.total_dim)
    if parent_data:
        source = node_plan.source
        if isinstance(source, ConcatNodeSource):
            concat = _sanitize_and_batch_standardize(torch.cat(parent_data, dim=2))
            latent = apply_function_plan_batch(
                concat,
                rng.keyed("function"),
                source.function,
                out_dim=total_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                standardize_input=False,
            )
        else:
            if not isinstance(source, StackedNodeSource):
                raise ValueError("Parent-driven fixed-layout node must use a multi-input source.")
            aggregation_kind = source.aggregation_kind
            if aggregation_kind == "logsumexp":
                transformed_outputs = [
                    apply_function_plan_batch(
                        _sanitize_and_batch_standardize(parent_tensor),
                        rng.keyed("parent", plan_index),
                        source.parent_functions[plan_index],
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                        standardize_input=False,
                    )
                    for plan_index, parent_tensor in enumerate(parent_data)
                ]
                stacked = torch.stack(transformed_outputs, dim=2)
                latent = _aggregate_parent_outputs_batch(
                    stacked,
                    aggregation_kind=source.aggregation_kind,
                )
            else:
                aggregate: torch.Tensor | None = None
                for plan_index, parent_tensor in enumerate(parent_data):
                    transformed_output = apply_function_plan_batch(
                        _sanitize_and_batch_standardize(parent_tensor),
                        rng.keyed("parent", plan_index),
                        source.parent_functions[plan_index],
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                        standardize_input=False,
                    )
                    if aggregate is None:
                        aggregate = transformed_output
                    else:
                        aggregate = _aggregate_batch_incrementally(
                            aggregate,
                            transformed_output,
                            aggregation_kind=aggregation_kind,
                        )
                if aggregate is None:
                    raise RuntimeError("Expected at least one parent tensor for stacked node plan.")
                latent = aggregate
    else:
        source = node_plan.source
        if not isinstance(source, RandomPointsNodeSource):
            raise ValueError("Root fixed-layout node must use a random-points source.")
        source_rng = rng.keyed("source")
        base = _sample_random_points_batch(
            source_rng.keyed("base"),
            n_rows=n_rows,
            dim=total_dim,
            base_kind=source.base_kind,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        latent = apply_function_plan_batch(
            base,
            source_rng.keyed("function"),
            source.function,
            out_dim=total_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _batch_standardize(latent)

    weights = _sample_random_weights_batch(
        rng.keyed("latent_weights"),
        dim=int(latent.shape[2]),
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    latent = latent * weights.unsqueeze(1)
    mean_l2 = torch.mean(torch.norm(latent, dim=2), dim=1)
    latent = latent / torch.clamp(mean_l2.view(-1, 1, 1), min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    converter_specs = node_plan.converter_specs
    converter_plans = node_plan.converter_plans
    for group in node_plan.converter_groups:
        spec_indices = [int(value) for value in group.spec_indices]
        if isinstance(group, NumericConverterGroup):
            spec_payloads = [converter_specs[idx] for idx in spec_indices]
            numeric_plans: list[NumericConverterPlan] = []
            for spec_index in spec_indices:
                plan = converter_plans[spec_index]
                if not isinstance(plan, NumericConverterPlan):
                    raise ValueError(
                        "Numeric converter group must reference numeric converter plans."
                    )
                numeric_plans.append(plan)
            if all(
                int(spec_payload.column_end - spec_payload.column_start) == 1
                for spec_payload in spec_payloads
            ):
                grouped_input = torch.cat(
                    [
                        latent[:, :, int(spec_payload.column_start) : int(spec_payload.column_end)]
                        for spec_payload in spec_payloads
                    ],
                    dim=2,
                )
                warp_enabled = torch.tensor(
                    [plan.warp_enabled for plan in numeric_plans],
                    device=latent.device,
                    dtype=torch.bool,
                )
                x_prime, values = _apply_numeric_converter_group_batch(
                    grouped_input,
                    rng,
                    warp_enabled,
                    spec_indices=tuple(spec_indices),
                )
                for local_index, spec_payload in enumerate(spec_payloads):
                    start = int(spec_payload.column_start)
                    end = int(spec_payload.column_end)
                    latent[:, :, start:end] = x_prime[:, :, local_index : local_index + 1]
                    extracted[str(spec_payload.key)] = values[:, :, local_index]
                continue
            for local_index, spec_payload in enumerate(spec_payloads):
                spec_index = spec_indices[local_index]
                start = int(spec_payload.column_start)
                end = int(spec_payload.column_end)
                spec_out, values = apply_numeric_converter_plan_batch(
                    latent[:, :, start:end],
                    rng.keyed("converter", spec_index),
                    numeric_plans[local_index],
                )
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec_payload.key)] = values
            continue

        spec_payloads = [converter_specs[idx] for idx in spec_indices]
        first_plan = converter_plans[spec_indices[0]]
        if not isinstance(first_plan, CategoricalConverterPlan):
            raise ValueError(
                "Categorical converter group must reference categorical converter plans."
            )
        if first_plan.variant != "center_random_fn":
            x_prime, values = _apply_categorical_group_batch(
                _categorical_group_input_views(latent, spec_payloads),
                rng,
                first_plan,
                n_categories=max(2, int(spec_payloads[0].cardinality or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                spec_indices=tuple(spec_indices),
            )
            for local_index, spec_payload in enumerate(spec_payloads):
                start = int(spec_payload.column_start)
                end = int(spec_payload.column_end)
                spec_out = x_prime[:, :, local_index, :]
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec_payload.key)] = values[:, :, local_index]
            continue
        for local_index, spec_payload in enumerate(spec_payloads):
            spec_index = spec_indices[local_index]
            start = int(spec_payload.column_start)
            end = int(spec_payload.column_end)
            spec_view = latent[:, :, start:end].unsqueeze(2)
            x_prime, values = _apply_categorical_group_batch(
                spec_view,
                rng.keyed("converter", spec_index),
                first_plan,
                n_categories=max(2, int(spec_payload.cardinality or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            spec_out = x_prime[:, :, 0, :]
            if int(spec_out.shape[2]) != (end - start):
                if int(spec_out.shape[2]) > (end - start):
                    spec_out = spec_out[:, :, : (end - start)]
                else:
                    spec_out = torch.nn.functional.pad(
                        spec_out, (0, (end - start) - int(spec_out.shape[2]))
                    )
            latent[:, :, start:end] = spec_out
            extracted[str(spec_payload.key)] = values[:, :, 0]

    scale = rng.keyed("latent_scale").log_uniform((rng.batch_size,), low=0.1, high=10.0)
    latent = latent * scale.view(-1, 1, 1)
    return latent, extracted


def _generate_fixed_layout_raw_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    emit_features: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, list[dict[str, Any]]]:
    """Generate one fixed-layout microbatch of raw tensors."""

    if not dataset_seeds:
        raise ValueError("dataset_seeds must be non-empty.")
    batch_size = len(dataset_seeds)
    n_rows = int(config.dataset.n_train + config.dataset.n_test)
    num_features = int(layout.n_features)
    dtype = torch.float32
    batch_seed = KeyedRng(int(dataset_seeds[0])).child_seed("fixed_layout_chunk", batch_size)
    rng = FixedLayoutBatchRng(seed=batch_seed, batch_size=batch_size, device=device)

    node_outputs: list[torch.Tensor | None] = [None] * int(layout.graph_nodes)
    feature_values: list[torch.Tensor | None] | None = (
        [None] * num_features if emit_features else None
    )
    target_values: torch.Tensor | None = None
    aux_meta_batch = [{"filter": {"mode": "deferred", "status": "not_run"}} for _ in dataset_seeds]

    for node_index, node_plan in enumerate(execution_plan.node_plans):
        parent_tensors = []
        for parent_index in node_plan.parent_indices:
            parent_output = node_outputs[int(parent_index)]
            if parent_output is not None:
                parent_tensors.append(parent_output)
        node_rng = rng.keyed("node", node_index)
        latent, extracted = _apply_node_plan_batch(
            config,
            node_plan,
            parent_tensors,
            n_rows=n_rows,
            rng=node_rng,
            device=device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        node_outputs[node_index] = latent
        for key, values in extracted.items():
            if key.startswith("feature_"):
                if feature_values is None:
                    continue
                feature_index = int(key.split("_", 1)[1])
                feature_values[feature_index] = values
            elif key == "target":
                target_values = values
            else:
                raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")

    x: torch.Tensor | None = None
    if feature_values is not None:
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
                        generator=rng.torch_generator,
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
                generator=rng.torch_generator,
                device=device,
                noise_spec=noise_spec,
            ).to(dtype)
    else:
        if str(config.dataset.task) == "classification":
            y = target_values.to(torch.int64) % int(layout.n_classes)
        else:
            y = target_values.to(dtype)
    return x, y, aux_meta_batch


def generate_fixed_layout_graph_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Generate one fixed-layout microbatch of raw `x`/`y` tensors."""

    x, y, aux_meta_batch = _generate_fixed_layout_raw_batch(
        config,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=dataset_seeds,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        emit_features=True,
    )
    if x is None:
        raise RuntimeError("Expected fixed-layout feature batch to be materialized.")
    return x, y, aux_meta_batch


def generate_fixed_layout_label_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Generate one fixed-layout microbatch of raw target tensors only."""

    _x, y, aux_meta_batch = _generate_fixed_layout_raw_batch(
        config,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=dataset_seeds,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        emit_features=False,
    )
    return y, aux_meta_batch
