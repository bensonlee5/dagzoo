"""Shared typed execution-plan sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, cast

import torch

from dagzoo.core.fixed_layout_plan_types import (
    ActivationMatrixPlan,
    CategoricalConverterPlan,
    ConcatNodeSource,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedActivationPlan,
    FixedLayoutActivationKind,
    FixedLayoutActivationPlan,
    FixedLayoutConverterMethod,
    FixedLayoutConverterPlan,
    FixedLayoutConverterSpec,
    FixedLayoutConverterVariant,
    FixedLayoutFunctionPlan,
    FixedLayoutLatentPlan,
    FixedLayoutMatrixBaseKind,
    FixedLayoutMatrixPlan,
    FixedLayoutNodePlan,
    FixedLayoutRootBaseKind,
    GaussianMatrixPlan,
    GpFunctionPlan,
    KernelMatrixPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterPlan,
    ParametricActivationPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    SingularValuesMatrixPlan,
    StackedNodeSource,
    TreeFunctionPlan,
    WeightsMatrixPlan,
    fixed_layout_converter_groups,
)
from dagzoo.core.layout_types import AggregationKind, ConverterKind, MechanismFamily
from dagzoo.core.shift import MECHANISM_FAMILY_ORDER, mechanism_family_probabilities
from dagzoo.functions.activations import fixed_activation_names
from dagzoo.math_utils import log_uniform as _log_uniform

_MATRIX_KIND_CHOICES: tuple[str, ...] = (
    "gaussian",
    "weights",
    "singular_values",
    "kernel",
    "activation",
)
_MATRIX_BASE_KIND_CHOICES: tuple[FixedLayoutMatrixBaseKind, ...] = (
    "gaussian",
    "weights",
    "singular_values",
    "kernel",
)
_ROOT_BASE_KIND_CHOICES: tuple[FixedLayoutRootBaseKind, ...] = (
    "normal",
    "uniform",
    "unit_ball",
    "normal_cov",
)
_PARAM_ACTIVATION_CHOICES: tuple[FixedLayoutActivationKind, ...] = (
    "relu_pow",
    "signed_pow",
    "inv_pow",
    "poly",
)
_AGGREGATION_KIND_ORDER: tuple[AggregationKind, ...] = ("sum", "product", "max", "logsumexp")
_PRODUCT_COMPONENT_FAMILIES: tuple[MechanismFamily, ...] = (
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
)
_JOINT_VARIANTS: tuple[tuple[FixedLayoutConverterMethod, FixedLayoutConverterVariant], ...] = (
    ("neighbor", "input"),
    ("neighbor", "index_repeat"),
    ("neighbor", "center"),
    ("neighbor", "center_random_fn"),
    ("softmax", "input"),
    ("softmax", "index_repeat"),
    ("softmax", "softmax_points"),
)


class ConverterSpecLike(Protocol):
    """Minimal protocol shared by scalar and fixed-layout converter specs."""

    key: str
    kind: ConverterKind
    dim: int
    cardinality: int | None


def _rand_scalar(generator: torch.Generator) -> float:
    return torch.rand(1, generator=generator, device=generator.device).item()


def _randint_scalar(low: int, high: int, generator: torch.Generator) -> int:
    return int(torch.randint(low, high, (1,), generator=generator, device=generator.device).item())


def sample_function_family(
    generator: torch.Generator,
    *,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None = None,
) -> MechanismFamily:
    """Sample one mechanism family with optional logit tilt."""

    if mechanism_logit_tilt <= 0.0 and function_family_mix is None:
        idx = _randint_scalar(0, len(MECHANISM_FAMILY_ORDER), generator)
        return MECHANISM_FAMILY_ORDER[int(idx)]

    probs_by_family = mechanism_family_probabilities(
        mechanism_logit_tilt=mechanism_logit_tilt,
        families=MECHANISM_FAMILY_ORDER,
        family_weights=function_family_mix,
    )
    positive_families = [
        family for family in MECHANISM_FAMILY_ORDER if float(probs_by_family.get(family, 0.0)) > 0.0
    ]
    if not positive_families:
        raise ValueError("No eligible mechanism families are available for sampling.")
    draw = float(_rand_scalar(generator))
    cumulative = 0.0
    for family in positive_families:
        cumulative += float(probs_by_family[family])
        if draw <= cumulative:
            return family
    return positive_families[-1]


def _sample_bool(generator: torch.Generator, *, p: float = 0.5) -> bool:
    return bool(_rand_scalar(generator) < p)


def sample_activation_plan(generator: torch.Generator) -> FixedLayoutActivationPlan:
    """Sample one activation plan using the shared fixed-layout schema."""

    if _rand_scalar(generator) < (1.0 / 3.0):
        choice = _PARAM_ACTIVATION_CHOICES[
            int(_randint_scalar(0, len(_PARAM_ACTIVATION_CHOICES), generator))
        ]
        if choice == "poly":
            return ParametricActivationPlan(
                kind=choice,
                poly_power=int(_randint_scalar(2, 6, generator)),
            )
        return ParametricActivationPlan(kind=choice)
    fixed = fixed_activation_names()
    name = fixed[int(_randint_scalar(0, len(fixed), generator))]
    return FixedActivationPlan(name=name)


def sample_matrix_plan(generator: torch.Generator) -> FixedLayoutMatrixPlan:
    """Sample one matrix-family plan."""

    kind = _MATRIX_KIND_CHOICES[int(_randint_scalar(0, len(_MATRIX_KIND_CHOICES), generator))]
    if kind == "gaussian":
        return GaussianMatrixPlan()
    if kind == "weights":
        return WeightsMatrixPlan()
    if kind == "singular_values":
        return SingularValuesMatrixPlan()
    if kind == "kernel":
        return KernelMatrixPlan()
    base_kind = _MATRIX_BASE_KIND_CHOICES[
        int(_randint_scalar(0, len(_MATRIX_BASE_KIND_CHOICES), generator))
    ]
    return ActivationMatrixPlan(
        base_kind=base_kind,
        activation=sample_activation_plan(generator),
    )


def sample_function_plan_for_family(
    generator: torch.Generator,
    *,
    family: MechanismFamily,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> FixedLayoutFunctionPlan:
    """Sample one typed function plan for an explicit family."""

    if family == "linear":
        return LinearFunctionPlan(matrix=sample_matrix_plan(generator))
    if family == "quadratic":
        return QuadraticFunctionPlan(matrix=sample_matrix_plan(generator))
    if family == "nn":
        n_layers = int(_randint_scalar(1, 4, generator))
        hidden_width = int(_log_uniform(generator, 1.0, 127.0, "cpu"))
        input_activation = sample_activation_plan(generator) if _sample_bool(generator) else None
        output_activation = sample_activation_plan(generator) if _sample_bool(generator) else None
        layer_count = max(1, n_layers)
        return NeuralNetFunctionPlan(
            n_layers=n_layers,
            hidden_width=max(1, hidden_width),
            input_activation=input_activation,
            output_activation=output_activation,
            layer_matrices=tuple(sample_matrix_plan(generator) for _ in range(layer_count)),
            hidden_activations=tuple(
                sample_activation_plan(generator) for _ in range(max(0, layer_count - 1))
            ),
        )
    if family == "tree":
        n_trees = int(_log_uniform(generator, 1.0, 32.0, "cpu"))
        n_trees = max(1, n_trees)
        return TreeFunctionPlan(
            n_trees=n_trees,
            depths=tuple(int(_randint_scalar(1, 8, generator)) for _ in range(n_trees)),
        )
    if family == "discretization":
        n_centers = int(_log_uniform(generator, 2.0, 128.0, "cpu"))
        return DiscretizationFunctionPlan(
            n_centers=max(2, n_centers),
            linear_matrix=sample_matrix_plan(generator),
        )
    if family == "gp":
        return GpFunctionPlan(branch_kind="ha" if _sample_bool(generator) else "projected")
    if family == "em":
        m_val = int(_log_uniform(generator, 2.0, float(max(16, 2 * out_dim)), "cpu"))
        return EmFunctionPlan(
            m_val=max(2, m_val),
            linear_matrix=sample_matrix_plan(generator),
        )
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
        lhs_family = eligible[int(_randint_scalar(0, len(eligible), generator))]
        rhs_family = eligible[int(_randint_scalar(0, len(eligible), generator))]
        return ProductFunctionPlan(
            lhs=sample_function_plan_for_family(
                generator,
                family=lhs_family,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
            ),
            rhs=sample_function_plan_for_family(
                generator,
                family=rhs_family,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
            ),
        )
    raise ValueError(f"Unsupported mechanism family in fixed-layout plan sampling: {family!r}")


def sample_function_plan(
    generator: torch.Generator,
    *,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> FixedLayoutFunctionPlan:
    """Sample one typed function plan using the shared family sampler."""

    family = sample_function_family(
        generator,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
    )
    return sample_function_plan_for_family(
        generator,
        family=family,
        out_dim=out_dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
    )


def sample_converter_plan(
    generator: torch.Generator,
    spec: ConverterSpecLike,
    *,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    method_override: str | None = None,
) -> FixedLayoutConverterPlan:
    """Sample one typed converter plan for a converter spec."""

    if spec.kind in {"num", "target_reg"}:
        return NumericConverterPlan(
            kind=cast(Literal["num", "target_reg"], spec.kind),
            warp_enabled=not _sample_bool(generator),
        )

    idx_joint = _randint_scalar(0, len(_JOINT_VARIANTS), generator)
    selected_method_raw, variant_raw = _JOINT_VARIANTS[int(idx_joint)]
    if method_override is None:
        selected_method = selected_method_raw
    else:
        normalized_method = method_override.strip().lower()
        if normalized_method not in {"neighbor", "softmax"}:
            raise ValueError(f"Unsupported categorical converter method: {normalized_method!r}.")
        selected_method = cast(FixedLayoutConverterMethod, normalized_method)
    variant = cast(FixedLayoutConverterVariant, variant_raw)
    if variant == "center_random_fn":
        return CategoricalConverterPlan(
            kind=cast(Literal["cat", "target_cls"], spec.kind),
            method=selected_method,
            variant=variant,
            function=sample_function_plan(
                generator,
                out_dim=max(1, int(spec.dim)),
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
            ),
        )
    return CategoricalConverterPlan(
        kind=cast(Literal["cat", "target_cls"], spec.kind),
        method=selected_method,
        variant=variant,
    )


def typed_converter_specs(
    converter_specs: Sequence[ConverterSpecLike],
) -> tuple[FixedLayoutConverterSpec, ...]:
    """Lift scalar converter specs into typed fixed-layout converter specs."""

    typed_specs: list[FixedLayoutConverterSpec] = []
    column_cursor = 0
    for spec in converter_specs:
        spec_dim = max(1, int(spec.dim))
        typed_specs.append(
            FixedLayoutConverterSpec(
                key=str(spec.key),
                kind=spec.kind,
                dim=int(spec.dim),
                cardinality=None if spec.cardinality is None else int(spec.cardinality),
                column_start=int(column_cursor),
                column_end=int(column_cursor + spec_dim),
            )
        )
        column_cursor += spec_dim
    return tuple(typed_specs)


def sample_latent_plan(
    converter_specs: Sequence[ConverterSpecLike],
    *,
    generator: torch.Generator,
    device: str,
) -> FixedLayoutLatentPlan:
    """Sample the shared latent-width plan for one node."""

    required_dim = int(sum(max(1, int(spec.dim)) for spec in converter_specs))
    extra_dim = max(1, int(_log_uniform(generator, 1.0, 32.0, device)))
    return FixedLayoutLatentPlan(
        required_dim=required_dim,
        extra_dim=extra_dim,
        total_dim=int(required_dim + extra_dim),
    )


def sample_root_source_plan(
    generator: torch.Generator,
    *,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> RandomPointsNodeSource:
    """Sample one root-source plan."""

    base_kind = _ROOT_BASE_KIND_CHOICES[
        int(_randint_scalar(0, len(_ROOT_BASE_KIND_CHOICES), generator))
    ]
    return RandomPointsNodeSource(
        base_kind=base_kind,
        function=sample_function_plan(
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        ),
    )


def sample_multi_source_plan(
    generator: torch.Generator,
    *,
    parent_count: int,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    aggregation_kind: AggregationKind | None = None,
) -> ConcatNodeSource | StackedNodeSource:
    """Sample one shared multi-parent source plan."""

    if parent_count <= 0:
        raise ValueError(f"parent_count must be > 0, got {parent_count}")
    combine_kind = "concat" if _sample_bool(generator) else "stack"
    if combine_kind == "concat":
        return ConcatNodeSource(
            function=sample_function_plan(
                generator,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
            )
        )
    resolved_aggregation_kind = aggregation_kind
    if resolved_aggregation_kind is None:
        resolved_aggregation_kind = _AGGREGATION_KIND_ORDER[
            int(_randint_scalar(0, len(_AGGREGATION_KIND_ORDER), generator))
        ]
    return StackedNodeSource(
        aggregation_kind=resolved_aggregation_kind,
        parent_functions=tuple(
            sample_function_plan(
                generator,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
            )
            for _ in range(parent_count)
        ),
    )


def sample_node_plan(
    *,
    node_index: int,
    parent_indices: Sequence[int],
    converter_specs: Sequence[ConverterSpecLike],
    generator: torch.Generator,
    device: str,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
) -> FixedLayoutNodePlan:
    """Sample one typed node execution plan."""

    latent = sample_latent_plan(
        converter_specs,
        generator=generator,
        device=device,
    )
    typed_specs = typed_converter_specs(converter_specs)
    converter_plans = tuple(
        sample_converter_plan(
            generator,
            spec,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
        for spec in converter_specs
    )
    source: ConcatNodeSource | StackedNodeSource | RandomPointsNodeSource
    if parent_indices:
        source = sample_multi_source_plan(
            generator,
            parent_count=len(parent_indices),
            out_dim=int(latent.total_dim),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
    else:
        source = sample_root_source_plan(
            generator,
            out_dim=int(latent.total_dim),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
    return FixedLayoutNodePlan(
        node_index=int(node_index),
        parent_indices=tuple(int(parent_index) for parent_index in parent_indices),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=latent,
        source=source,
    )


__all__ = [
    "_AGGREGATION_KIND_ORDER",
    "_JOINT_VARIANTS",
    "_ROOT_BASE_KIND_CHOICES",
    "sample_activation_plan",
    "sample_converter_plan",
    "sample_function_family",
    "sample_function_plan",
    "sample_function_plan_for_family",
    "sample_latent_plan",
    "sample_matrix_plan",
    "sample_multi_source_plan",
    "sample_node_plan",
    "sample_root_source_plan",
    "typed_converter_specs",
]
