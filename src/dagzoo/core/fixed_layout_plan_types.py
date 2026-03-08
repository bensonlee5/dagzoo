"""Internal typed fixed-layout execution-plan contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, TypeAlias

from dagzoo.core.layout_types import AggregationKind, ConverterKind

FixedLayoutExecutionContract = Literal["chunk_batched_v1"]
FixedLayoutRootBaseKind = Literal["normal", "uniform", "unit_ball", "normal_cov"]
FixedLayoutMatrixBaseKind = Literal["gaussian", "weights", "singular_values", "kernel"]
FixedLayoutActivationKind = Literal["relu_pow", "signed_pow", "inv_pow", "poly"]
FixedLayoutGpBranchKind = Literal["ha", "projected"]
FixedLayoutConverterMethod = Literal["neighbor", "softmax"]
FixedLayoutConverterVariant = Literal[
    "input",
    "index_repeat",
    "center",
    "center_random_fn",
    "softmax_points",
]

DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT: FixedLayoutExecutionContract = "chunk_batched_v1"


@dataclass(frozen=True, slots=True)
class FixedActivationPlan:
    name: str
    mode: Literal["fixed"] = "fixed"


@dataclass(frozen=True, slots=True)
class ParametricActivationPlan:
    kind: FixedLayoutActivationKind
    poly_power: int | None = None
    mode: Literal["parametric"] = "parametric"


FixedLayoutActivationPlan: TypeAlias = FixedActivationPlan | ParametricActivationPlan


@dataclass(frozen=True, slots=True)
class GaussianMatrixPlan:
    kind: Literal["gaussian"] = "gaussian"


@dataclass(frozen=True, slots=True)
class WeightsMatrixPlan:
    kind: Literal["weights"] = "weights"


@dataclass(frozen=True, slots=True)
class SingularValuesMatrixPlan:
    kind: Literal["singular_values"] = "singular_values"


@dataclass(frozen=True, slots=True)
class KernelMatrixPlan:
    kind: Literal["kernel"] = "kernel"


@dataclass(frozen=True, slots=True)
class ActivationMatrixPlan:
    base_kind: FixedLayoutMatrixBaseKind
    activation: FixedLayoutActivationPlan
    kind: Literal["activation"] = "activation"


FixedLayoutMatrixPlan: TypeAlias = (
    GaussianMatrixPlan
    | WeightsMatrixPlan
    | SingularValuesMatrixPlan
    | KernelMatrixPlan
    | ActivationMatrixPlan
)


@dataclass(frozen=True, slots=True)
class LinearFunctionPlan:
    matrix: FixedLayoutMatrixPlan
    family: Literal["linear"] = "linear"


@dataclass(frozen=True, slots=True)
class QuadraticFunctionPlan:
    matrix: FixedLayoutMatrixPlan
    family: Literal["quadratic"] = "quadratic"


@dataclass(frozen=True, slots=True)
class NeuralNetFunctionPlan:
    n_layers: int
    hidden_width: int
    input_activation: FixedLayoutActivationPlan | None
    output_activation: FixedLayoutActivationPlan | None
    layer_matrices: tuple[FixedLayoutMatrixPlan, ...]
    hidden_activations: tuple[FixedLayoutActivationPlan, ...]
    family: Literal["nn"] = "nn"


@dataclass(frozen=True, slots=True)
class TreeFunctionPlan:
    n_trees: int
    depths: tuple[int, ...]
    family: Literal["tree"] = "tree"


@dataclass(frozen=True, slots=True)
class DiscretizationFunctionPlan:
    n_centers: int
    linear_matrix: FixedLayoutMatrixPlan
    family: Literal["discretization"] = "discretization"


@dataclass(frozen=True, slots=True)
class GpFunctionPlan:
    branch_kind: FixedLayoutGpBranchKind
    family: Literal["gp"] = "gp"


@dataclass(frozen=True, slots=True)
class EmFunctionPlan:
    m_val: int
    linear_matrix: FixedLayoutMatrixPlan
    family: Literal["em"] = "em"


@dataclass(frozen=True, slots=True)
class ProductFunctionPlan:
    lhs: FixedLayoutFunctionPlan
    rhs: FixedLayoutFunctionPlan
    family: Literal["product"] = "product"


FixedLayoutFunctionPlan: TypeAlias = (
    LinearFunctionPlan
    | QuadraticFunctionPlan
    | NeuralNetFunctionPlan
    | TreeFunctionPlan
    | DiscretizationFunctionPlan
    | GpFunctionPlan
    | EmFunctionPlan
    | ProductFunctionPlan
)


@dataclass(frozen=True, slots=True)
class FixedLayoutLatentPlan:
    required_dim: int
    extra_dim: int
    total_dim: int


@dataclass(frozen=True, slots=True)
class FixedLayoutConverterSpec:
    key: str
    kind: ConverterKind
    dim: int
    cardinality: int | None
    column_start: int
    column_end: int


@dataclass(frozen=True, slots=True)
class NumericConverterPlan:
    kind: Literal["num", "target_reg"]
    warp_enabled: bool


@dataclass(frozen=True, slots=True)
class CategoricalConverterPlan:
    kind: Literal["cat", "target_cls"]
    method: FixedLayoutConverterMethod
    variant: FixedLayoutConverterVariant
    function: FixedLayoutFunctionPlan | None = None


FixedLayoutConverterPlan: TypeAlias = NumericConverterPlan | CategoricalConverterPlan


@dataclass(frozen=True, slots=True)
class NumericConverterGroup:
    spec_indices: tuple[int, ...]
    group_kind: Literal["numeric"] = "numeric"


@dataclass(frozen=True, slots=True)
class CategoricalConverterGroup:
    spec_indices: tuple[int, ...]
    group_kind: Literal["categorical"] = "categorical"


FixedLayoutConverterGroup: TypeAlias = NumericConverterGroup | CategoricalConverterGroup


@dataclass(frozen=True, slots=True)
class RandomPointsNodeSource:
    base_kind: FixedLayoutRootBaseKind
    function: FixedLayoutFunctionPlan
    source_kind: Literal["random_points"] = "random_points"


@dataclass(frozen=True, slots=True)
class ConcatNodeSource:
    function: FixedLayoutFunctionPlan
    source_kind: Literal["multi"] = "multi"
    combine_kind: Literal["concat"] = "concat"


@dataclass(frozen=True, slots=True)
class StackedNodeSource:
    aggregation_kind: AggregationKind
    parent_functions: tuple[FixedLayoutFunctionPlan, ...]
    source_kind: Literal["multi"] = "multi"
    combine_kind: Literal["stack"] = "stack"


FixedLayoutNodeSource: TypeAlias = RandomPointsNodeSource | ConcatNodeSource | StackedNodeSource


@dataclass(frozen=True, slots=True)
class FixedLayoutNodePlan:
    node_index: int
    parent_indices: tuple[int, ...]
    converter_specs: tuple[FixedLayoutConverterSpec, ...]
    converter_plans: tuple[FixedLayoutConverterPlan, ...]
    converter_groups: tuple[FixedLayoutConverterGroup, ...]
    latent: FixedLayoutLatentPlan
    source: FixedLayoutNodeSource


@dataclass(frozen=True, slots=True)
class FixedLayoutExecutionPlan:
    node_plans: tuple[FixedLayoutNodePlan, ...] = ()
    execution_contract: FixedLayoutExecutionContract = DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT


def fixed_layout_converter_groups(
    converter_specs: Sequence[FixedLayoutConverterSpec],
    converter_plans: Sequence[FixedLayoutConverterPlan],
) -> tuple[FixedLayoutConverterGroup, ...]:
    groups: dict[tuple[Any, ...], FixedLayoutConverterGroup] = {}
    ordered_keys: list[tuple[Any, ...]] = []
    for spec_index, (spec, plan) in enumerate(zip(converter_specs, converter_plans, strict=True)):
        key: tuple[Any, ...]
        if isinstance(plan, NumericConverterPlan):
            key = ("numeric",)
            group = groups.get(key)
            if group is None:
                ordered_keys.append(key)
                groups[key] = NumericConverterGroup(spec_indices=(int(spec_index),))
            else:
                groups[key] = NumericConverterGroup(
                    spec_indices=(*group.spec_indices, int(spec_index))
                )
            continue

        key = (
            "categorical",
            str(spec.kind),
            str(plan.method),
            str(plan.variant),
            int(spec.dim),
            None if spec.cardinality is None else int(spec.cardinality),
            _converter_function_signature(plan.function),
        )
        group = groups.get(key)
        if group is None:
            ordered_keys.append(key)
            groups[key] = CategoricalConverterGroup(spec_indices=(int(spec_index),))
        else:
            groups[key] = CategoricalConverterGroup(
                spec_indices=(*group.spec_indices, int(spec_index))
            )
    return tuple(groups[key] for key in ordered_keys)


def fixed_layout_execution_plan_payloads(
    execution_plan: FixedLayoutExecutionPlan,
) -> list[dict[str, Any]]:
    return [
        _node_plan_payload(node_plan, execution_contract=execution_plan.execution_contract)
        for node_plan in execution_plan.node_plans
    ]


def fixed_layout_signature_payloads(
    execution_plan: FixedLayoutExecutionPlan,
) -> list[dict[str, Any]]:
    return [_signature_node_plan_payload(node_plan) for node_plan in execution_plan.node_plans]


def coerce_fixed_layout_execution_plan(
    value: FixedLayoutExecutionPlan | Sequence[Mapping[str, Any]],
) -> FixedLayoutExecutionPlan:
    if isinstance(value, FixedLayoutExecutionPlan):
        return value

    node_plans: list[FixedLayoutNodePlan] = []
    execution_contract: str | None = None
    for raw_plan in value:
        plan_mapping = dict(raw_plan)
        raw_execution_contract = str(
            plan_mapping.get("execution_contract", DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT)
        )
        if execution_contract is None:
            execution_contract = raw_execution_contract
        elif raw_execution_contract != execution_contract:
            raise ValueError("Legacy fixed-layout payload uses inconsistent execution contracts.")
        converter_specs = _coerce_converter_specs(plan_mapping.get("converter_specs", []))
        converter_plans = tuple(
            _coerce_converter_plan(plan) for plan in list(plan_mapping.get("converter_plans", []))
        )
        if len(converter_specs) != len(converter_plans):
            raise ValueError("converter_specs and converter_plans must have matching lengths.")
        raw_groups = plan_mapping.get("converter_groups")
        if raw_groups is None:
            converter_groups = fixed_layout_converter_groups(converter_specs, converter_plans)
        else:
            converter_groups = tuple(_coerce_converter_group(group) for group in list(raw_groups))
        node_plans.append(
            FixedLayoutNodePlan(
                node_index=int(plan_mapping["node_index"]),
                parent_indices=tuple(
                    int(parent_index)
                    for parent_index in list(plan_mapping.get("parent_indices", []))
                ),
                converter_specs=converter_specs,
                converter_plans=converter_plans,
                converter_groups=converter_groups,
                latent=FixedLayoutLatentPlan(
                    required_dim=int(plan_mapping["latent"]["required_dim"]),
                    extra_dim=int(plan_mapping["latent"]["extra_dim"]),
                    total_dim=int(plan_mapping["latent"]["total_dim"]),
                ),
                source=_coerce_node_source(plan_mapping),
            )
        )
    return FixedLayoutExecutionPlan(
        node_plans=tuple(node_plans),
        execution_contract=str(execution_contract or DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT),  # type: ignore[arg-type]
    )


def _coerce_converter_specs(
    raw_specs: Any,
) -> tuple[FixedLayoutConverterSpec, ...]:
    cursor = 0
    specs: list[FixedLayoutConverterSpec] = []
    for raw_spec in list(raw_specs):
        spec_mapping = dict(raw_spec)
        dim = max(1, int(spec_mapping["dim"]))
        column_start = int(spec_mapping.get("column_start", cursor))
        column_end = int(spec_mapping.get("column_end", column_start + dim))
        cursor = column_end
        specs.append(
            FixedLayoutConverterSpec(
                key=str(spec_mapping["key"]),
                kind=str(spec_mapping["kind"]),  # type: ignore[arg-type]
                dim=int(spec_mapping["dim"]),
                cardinality=(
                    None
                    if spec_mapping.get("cardinality") is None
                    else int(spec_mapping["cardinality"])
                ),
                column_start=column_start,
                column_end=column_end,
            )
        )
    return tuple(specs)


def _coerce_converter_plan(raw_plan: Mapping[str, Any]) -> FixedLayoutConverterPlan:
    plan_mapping = dict(raw_plan)
    kind = str(plan_mapping["kind"])
    if kind in {"num", "target_reg"}:
        return NumericConverterPlan(
            kind=kind,  # type: ignore[arg-type]
            warp_enabled=bool(plan_mapping["warp_enabled"]),
        )
    function_payload = plan_mapping.get("function")
    variant = str(plan_mapping["variant"])
    if variant == "center_random_fn" and function_payload is None:
        raise ValueError("center_random_fn converter plan requires a nested function payload.")
    return CategoricalConverterPlan(
        kind=kind,  # type: ignore[arg-type]
        method=str(plan_mapping["method"]),  # type: ignore[arg-type]
        variant=variant,  # type: ignore[arg-type]
        function=(None if function_payload is None else _coerce_function_plan(function_payload)),
    )


def _coerce_converter_group(raw_group: Mapping[str, Any]) -> FixedLayoutConverterGroup:
    group_kind = str(raw_group["group_kind"])
    spec_indices = tuple(int(value) for value in list(raw_group["spec_indices"]))
    if group_kind == "numeric":
        return NumericConverterGroup(spec_indices=spec_indices)
    if group_kind == "categorical":
        return CategoricalConverterGroup(spec_indices=spec_indices)
    raise ValueError(f"Unsupported fixed-layout converter group kind: {group_kind!r}")


def _coerce_node_source(raw_plan: Mapping[str, Any]) -> FixedLayoutNodeSource:
    source_kind = str(raw_plan["source_kind"])
    if source_kind == "random_points":
        return RandomPointsNodeSource(
            base_kind=str(raw_plan["base_kind"]),  # type: ignore[arg-type]
            function=_coerce_function_plan(raw_plan["function"]),
        )
    if source_kind != "multi":
        raise ValueError(f"Unsupported fixed-layout source kind: {source_kind!r}")
    combine_kind = str(raw_plan["combine_kind"])
    if combine_kind == "concat":
        return ConcatNodeSource(function=_coerce_function_plan(raw_plan["function"]))
    if combine_kind == "stack":
        return StackedNodeSource(
            aggregation_kind=str(raw_plan["aggregation_kind"]),  # type: ignore[arg-type]
            parent_functions=tuple(
                _coerce_function_plan(plan) for plan in list(raw_plan["parent_functions"])
            ),
        )
    raise ValueError(f"Unsupported fixed-layout combine kind: {combine_kind!r}")


def _coerce_function_plan(raw_plan: Mapping[str, Any]) -> FixedLayoutFunctionPlan:
    family = str(raw_plan["family"])
    if family == "linear":
        return LinearFunctionPlan(matrix=_coerce_matrix_plan(raw_plan["matrix"]))
    if family == "quadratic":
        return QuadraticFunctionPlan(matrix=_coerce_matrix_plan(raw_plan["matrix"]))
    if family == "nn":
        return NeuralNetFunctionPlan(
            n_layers=int(raw_plan["n_layers"]),
            hidden_width=int(raw_plan["hidden_width"]),
            input_activation=(
                None
                if not bool(raw_plan.get("apply_input_activation"))
                else _coerce_activation_plan(raw_plan["input_activation"])
            ),
            output_activation=(
                None
                if not bool(raw_plan.get("apply_output_activation"))
                else _coerce_activation_plan(raw_plan["output_activation"])
            ),
            layer_matrices=tuple(
                _coerce_matrix_plan(plan) for plan in list(raw_plan["layer_matrices"])
            ),
            hidden_activations=tuple(
                _coerce_activation_plan(plan) for plan in list(raw_plan["hidden_activations"])
            ),
        )
    if family == "tree":
        return TreeFunctionPlan(
            n_trees=int(raw_plan["n_trees"]),
            depths=tuple(int(depth) for depth in list(raw_plan["depths"])),
        )
    if family == "discretization":
        return DiscretizationFunctionPlan(
            n_centers=int(raw_plan["n_centers"]),
            linear_matrix=_coerce_matrix_plan(raw_plan["linear_matrix"]),
        )
    if family == "gp":
        return GpFunctionPlan(branch_kind=str(raw_plan["branch_kind"]))  # type: ignore[arg-type]
    if family == "em":
        return EmFunctionPlan(
            m_val=int(raw_plan["m_val"]),
            linear_matrix=_coerce_matrix_plan(raw_plan["linear_matrix"]),
        )
    if family == "product":
        return ProductFunctionPlan(
            lhs=_coerce_function_plan(raw_plan["lhs"]),
            rhs=_coerce_function_plan(raw_plan["rhs"]),
        )
    raise ValueError(f"Unsupported fixed-layout function family: {family!r}")


def _coerce_matrix_plan(raw_plan: Mapping[str, Any]) -> FixedLayoutMatrixPlan:
    kind = str(raw_plan["kind"])
    if kind == "gaussian":
        return GaussianMatrixPlan()
    if kind == "weights":
        return WeightsMatrixPlan()
    if kind == "singular_values":
        return SingularValuesMatrixPlan()
    if kind == "kernel":
        return KernelMatrixPlan()
    if kind == "activation":
        return ActivationMatrixPlan(
            base_kind=str(raw_plan["base_kind"]),  # type: ignore[arg-type]
            activation=_coerce_activation_plan(raw_plan["activation"]),
        )
    raise ValueError(f"Unsupported fixed-layout matrix kind: {kind!r}")


def _coerce_activation_plan(raw_plan: Mapping[str, Any]) -> FixedLayoutActivationPlan:
    mode = str(raw_plan["mode"])
    if mode == "fixed":
        return FixedActivationPlan(name=str(raw_plan["name"]))
    if mode == "parametric":
        poly_power = raw_plan.get("poly_power")
        return ParametricActivationPlan(
            kind=str(raw_plan["kind"]),  # type: ignore[arg-type]
            poly_power=None if poly_power is None else int(poly_power),
        )
    raise ValueError(f"Unsupported fixed-layout activation mode: {mode!r}")


def _converter_function_signature(plan: FixedLayoutFunctionPlan | None) -> str | None:
    if plan is None:
        return None
    encoded = json.dumps(
        _function_plan_payload(plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=8).hexdigest()


def _node_plan_payload(
    node_plan: FixedLayoutNodePlan,
    *,
    execution_contract: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_index": int(node_plan.node_index),
        "parent_indices": [int(parent_index) for parent_index in node_plan.parent_indices],
        "converter_specs": [
            {
                "key": str(spec.key),
                "kind": str(spec.kind),
                "dim": int(spec.dim),
                "cardinality": None if spec.cardinality is None else int(spec.cardinality),
                "column_start": int(spec.column_start),
                "column_end": int(spec.column_end),
            }
            for spec in node_plan.converter_specs
        ],
        "converter_plans": [_converter_plan_payload(plan) for plan in node_plan.converter_plans],
        "latent": {
            "required_dim": int(node_plan.latent.required_dim),
            "extra_dim": int(node_plan.latent.extra_dim),
            "total_dim": int(node_plan.latent.total_dim),
        },
        "execution_contract": str(execution_contract),
        "converter_groups": _converter_group_payloads(node_plan),
    }
    source = node_plan.source
    if isinstance(source, RandomPointsNodeSource):
        payload["source_kind"] = "random_points"
        payload["base_kind"] = str(source.base_kind)
        payload["function"] = _function_plan_payload(source.function)
        return payload
    payload["source_kind"] = "multi"
    payload["combine_kind"] = str(source.combine_kind)
    if isinstance(source, ConcatNodeSource):
        payload["function"] = _function_plan_payload(source.function)
        return payload
    payload["aggregation_kind"] = str(source.aggregation_kind)
    payload["parent_functions"] = [_function_plan_payload(plan) for plan in source.parent_functions]
    return payload


def _signature_node_plan_payload(node_plan: FixedLayoutNodePlan) -> dict[str, Any]:
    payload = _node_plan_payload(
        node_plan,
        execution_contract=DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT,
    )
    payload.pop("converter_groups", None)
    payload.pop("execution_contract", None)
    stripped_specs: list[dict[str, Any]] = []
    for spec in list(payload["converter_specs"]):
        spec_payload = dict(spec)
        spec_payload.pop("column_start", None)
        spec_payload.pop("column_end", None)
        stripped_specs.append(spec_payload)
    payload["converter_specs"] = stripped_specs
    return payload


def _converter_plan_payload(plan: FixedLayoutConverterPlan) -> dict[str, Any]:
    if isinstance(plan, NumericConverterPlan):
        return {
            "kind": str(plan.kind),
            "warp_enabled": bool(plan.warp_enabled),
        }
    payload: dict[str, Any] = {
        "kind": str(plan.kind),
        "method": str(plan.method),
        "variant": str(plan.variant),
    }
    if plan.function is not None:
        payload["function"] = _function_plan_payload(plan.function)
    return payload


def _converter_group_payloads(node_plan: FixedLayoutNodePlan) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for group in node_plan.converter_groups:
        if isinstance(group, NumericConverterGroup):
            payloads.append(
                {
                    "group_kind": "numeric",
                    "group_key": "numeric",
                    "spec_indices": [int(value) for value in group.spec_indices],
                }
            )
            continue

        first_index = int(group.spec_indices[0])
        spec = node_plan.converter_specs[first_index]
        plan = node_plan.converter_plans[first_index]
        if not isinstance(plan, CategoricalConverterPlan):
            raise ValueError("Categorical converter group must reference categorical plans.")
        payloads.append(
            {
                "group_kind": "categorical",
                "group_key": json.dumps(
                    {
                        "kind": str(spec.kind),
                        "method": str(plan.method),
                        "variant": str(plan.variant),
                        "dim": int(spec.dim),
                        "cardinality": (
                            None if spec.cardinality is None else int(spec.cardinality)
                        ),
                        "function_signature": _converter_function_signature(plan.function),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "method": str(plan.method),
                "variant": str(plan.variant),
                "dim": int(spec.dim),
                "cardinality": None if spec.cardinality is None else int(spec.cardinality),
                "function_signature": _converter_function_signature(plan.function),
                "spec_indices": [int(value) for value in group.spec_indices],
            }
        )
    return payloads


def _function_plan_payload(plan: FixedLayoutFunctionPlan) -> dict[str, Any]:
    if isinstance(plan, LinearFunctionPlan):
        return {"family": "linear", "matrix": _matrix_plan_payload(plan.matrix)}
    if isinstance(plan, QuadraticFunctionPlan):
        return {"family": "quadratic", "matrix": _matrix_plan_payload(plan.matrix)}
    if isinstance(plan, NeuralNetFunctionPlan):
        payload: dict[str, Any] = {
            "family": "nn",
            "n_layers": int(plan.n_layers),
            "hidden_width": int(plan.hidden_width),
            "apply_input_activation": bool(plan.input_activation is not None),
            "apply_output_activation": bool(plan.output_activation is not None),
            "layer_matrices": [_matrix_plan_payload(matrix) for matrix in plan.layer_matrices],
            "hidden_activations": [
                _activation_plan_payload(activation) for activation in plan.hidden_activations
            ],
        }
        if plan.input_activation is not None:
            payload["input_activation"] = _activation_plan_payload(plan.input_activation)
        if plan.output_activation is not None:
            payload["output_activation"] = _activation_plan_payload(plan.output_activation)
        return payload
    if isinstance(plan, TreeFunctionPlan):
        return {
            "family": "tree",
            "n_trees": int(plan.n_trees),
            "depths": [int(depth) for depth in plan.depths],
        }
    if isinstance(plan, DiscretizationFunctionPlan):
        return {
            "family": "discretization",
            "n_centers": int(plan.n_centers),
            "linear_matrix": _matrix_plan_payload(plan.linear_matrix),
        }
    if isinstance(plan, GpFunctionPlan):
        return {"family": "gp", "branch_kind": str(plan.branch_kind)}
    if isinstance(plan, EmFunctionPlan):
        return {
            "family": "em",
            "m_val": int(plan.m_val),
            "linear_matrix": _matrix_plan_payload(plan.linear_matrix),
        }
    return {
        "family": "product",
        "lhs": _function_plan_payload(plan.lhs),
        "rhs": _function_plan_payload(plan.rhs),
    }


def _matrix_plan_payload(plan: FixedLayoutMatrixPlan) -> dict[str, Any]:
    if isinstance(plan, GaussianMatrixPlan):
        return {"kind": "gaussian"}
    if isinstance(plan, WeightsMatrixPlan):
        return {"kind": "weights"}
    if isinstance(plan, SingularValuesMatrixPlan):
        return {"kind": "singular_values"}
    if isinstance(plan, KernelMatrixPlan):
        return {"kind": "kernel"}
    return {
        "kind": "activation",
        "base_kind": str(plan.base_kind),
        "activation": _activation_plan_payload(plan.activation),
    }


def _activation_plan_payload(plan: FixedLayoutActivationPlan) -> dict[str, Any]:
    if isinstance(plan, FixedActivationPlan):
        return {"mode": "fixed", "name": str(plan.name)}
    payload: dict[str, Any] = {
        "mode": "parametric",
        "kind": str(plan.kind),
    }
    if plan.poly_power is not None:
        payload["poly_power"] = int(plan.poly_power)
    return payload


__all__ = [
    "ActivationMatrixPlan",
    "CategoricalConverterGroup",
    "CategoricalConverterPlan",
    "ConcatNodeSource",
    "DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT",
    "DiscretizationFunctionPlan",
    "EmFunctionPlan",
    "FixedActivationPlan",
    "FixedLayoutActivationPlan",
    "FixedLayoutConverterGroup",
    "FixedLayoutConverterPlan",
    "FixedLayoutConverterSpec",
    "FixedLayoutExecutionPlan",
    "FixedLayoutFunctionPlan",
    "FixedLayoutLatentPlan",
    "FixedLayoutMatrixPlan",
    "FixedLayoutNodePlan",
    "FixedLayoutNodeSource",
    "GaussianMatrixPlan",
    "GpFunctionPlan",
    "KernelMatrixPlan",
    "LinearFunctionPlan",
    "NeuralNetFunctionPlan",
    "NumericConverterGroup",
    "NumericConverterPlan",
    "ParametricActivationPlan",
    "ProductFunctionPlan",
    "QuadraticFunctionPlan",
    "RandomPointsNodeSource",
    "SingularValuesMatrixPlan",
    "StackedNodeSource",
    "TreeFunctionPlan",
    "WeightsMatrixPlan",
    "coerce_fixed_layout_execution_plan",
    "fixed_layout_converter_groups",
    "fixed_layout_execution_plan_payloads",
    "fixed_layout_signature_payloads",
]
