"""Effective-diversity audits for local overlap and dataset-scale impact."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys
import time
from typing import Any, Iterator

import torch
from sklearn.metrics import adjusted_rand_score

from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.core import fixed_layout_batched as fixed_layout_batched_module
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core import execution_semantics as execution_semantics_module
from dagzoo.core.layout_types import AggregationKind, MechanismFamily
from dagzoo.core.shift import MECHANISM_FAMILY_ORDER
from dagzoo.diagnostics.coverage import CoverageAggregationConfig, CoverageAggregator
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config
from dagzoo.functions import activations as activations_module
from dagzoo.functions import multi as multi_module
from dagzoo.functions import random_functions as random_functions_module
from dagzoo.functions.multi import _aggregate_parent_outputs
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.math_utils import log_uniform, sanitize_json, standardize
from dagzoo.types import DatasetBundle

_MECHANISM_FAMILIES: tuple[MechanismFamily, ...] = MECHANISM_FAMILY_ORDER
_AGGREGATION_SCALE_FACTORS: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0)
_AGGREGATION_KINDS: frozenset[AggregationKind] = frozenset({"sum", "product", "max", "logsumexp"})

_SCALE_CORE_METRICS: tuple[str, ...] = (
    "linearity_proxy",
    "nonlinearity_proxy",
    "wins_ratio_proxy",
    "pearson_abs_mean",
    "pearson_abs_max",
    "snr_proxy_db",
    "class_entropy",
    "majority_minority_ratio",
    "categorical_ratio",
    "cat_cardinality_mean",
    "graph_edge_density",
)

_SCALE_SUITE_DATASETS: dict[str, int] = {
    "smoke": 128,
    "standard": 10_000,
    "full": 20_000,
}


@dataclass(slots=True, frozen=True)
class AuditThresholds:
    exact_affine_rmse: float = 1e-6
    near_cosine: float = 0.95
    near_affine_rmse: float = 0.20


@dataclass(slots=True, frozen=True)
class HypothesisClaim:
    claim_id: str
    track: str
    left: str
    right: str
    claim_type: str
    confidence: str
    description: str


@dataclass(slots=True, frozen=True)
class AblationArm:
    arm_id: str
    description: str
    claim_ids: tuple[str, ...] = ()
    confidence: str = "high"
    activation_map: dict[str, str] = field(default_factory=dict)
    family_map: dict[MechanismFamily, MechanismFamily] = field(default_factory=dict)
    aggregation_map: dict[AggregationKind, AggregationKind] = field(default_factory=dict)


_HYPOTHESIS_CLAIMS: tuple[HypothesisClaim, ...] = (
    HypothesisClaim(
        claim_id="sigmoid_vs_tanh",
        track="activations",
        left="sigmoid",
        right="tanh",
        claim_type="near",
        confidence="high",
        description="Sigmoid/tanh are affine-related under random scaling + standardization envelope.",
    ),
    HypothesisClaim(
        claim_id="sign_vs_heaviside",
        track="activations",
        left="sign",
        right="heaviside",
        claim_type="exact",
        confidence="high",
        description="Binary sign/heaviside outputs collapse to the same standardized two-level support.",
    ),
    HypothesisClaim(
        claim_id="softplus_vs_logsigmoid",
        track="activations",
        left="softplus",
        right="logsigmoid",
        claim_type="near",
        confidence="high",
        description="logsigmoid(x) = -softplus(-x); upstream sign flips can collapse these behaviors.",
    ),
    HypothesisClaim(
        claim_id="relu6_vs_hardtanh",
        track="activations",
        left="relu6",
        right="hardtanh",
        claim_type="near",
        confidence="high",
        description="Both are clipped piecewise-linear responses under scale/shift envelopes.",
    ),
    HypothesisClaim(
        claim_id="selu_vs_elu",
        track="activations",
        left="selu",
        right="elu",
        claim_type="near",
        confidence="high",
        description="SELU is scaled ELU; post-activation standardization removes most scale effects.",
    ),
    HypothesisClaim(
        claim_id="linear_vs_nn",
        track="function_families",
        left="linear",
        right="nn",
        claim_type="subset",
        confidence="high",
        description="Linear is a strict subset of NN via one-layer/no-activation degenerate draws.",
    ),
    HypothesisClaim(
        claim_id="quadratic_vs_product",
        track="function_families",
        left="quadratic",
        right="product",
        claim_type="subset",
        confidence="high",
        description="Quadratic interactions overlap with a subset of product-family constructions.",
    ),
    HypothesisClaim(
        claim_id="gp_vs_nn",
        track="function_families",
        left="gp",
        right="nn",
        claim_type="approximate",
        confidence="approximate",
        description="RFF GP approximations overlap with shallow periodic NN constructions.",
    ),
    HypothesisClaim(
        claim_id="em_vs_discretization",
        track="function_families",
        left="em",
        right="discretization",
        claim_type="approximate",
        confidence="approximate",
        description="EM soft assignments approximate hard nearest-center discretization.",
    ),
    HypothesisClaim(
        claim_id="tree_vs_discretization",
        track="function_families",
        left="tree",
        right="discretization",
        claim_type="approximate",
        confidence="approximate",
        description="Both induce piecewise-constant regions with different partition geometry.",
    ),
    HypothesisClaim(
        claim_id="max_vs_logsumexp",
        track="aggregation",
        left="max",
        right="logsumexp",
        claim_type="approximate",
        confidence="approximate",
        description="LogSumExp is a smooth max surrogate and converges toward max at higher scales.",
    ),
    HypothesisClaim(
        claim_id="rank_vs_argsort",
        track="activations",
        left="rank",
        right="argsort",
        claim_type="descriptive",
        confidence="approximate",
        description="Rank and argsort are distinct order encodings with related integer supports.",
    ),
)

_BASE_ABLATION_ARMS: tuple[AblationArm, ...] = (
    AblationArm(
        arm_id="act_sigmoid_to_tanh",
        description="Replace sigmoid with tanh.",
        claim_ids=("sigmoid_vs_tanh",),
        confidence="high",
        activation_map={"sigmoid": "tanh"},
    ),
    AblationArm(
        arm_id="act_sign_to_heaviside",
        description="Replace sign with heaviside (runtime availability dependent).",
        claim_ids=("sign_vs_heaviside",),
        confidence="high",
        activation_map={"sign": "heaviside"},
    ),
    AblationArm(
        arm_id="act_logsigmoid_to_softplus",
        description="Replace logsigmoid with softplus.",
        claim_ids=("softplus_vs_logsigmoid",),
        confidence="high",
        activation_map={"logsigmoid": "softplus"},
    ),
    AblationArm(
        arm_id="act_relu6_to_hardtanh",
        description="Replace relu6 with hardtanh.",
        claim_ids=("relu6_vs_hardtanh",),
        confidence="high",
        activation_map={"relu6": "hardtanh"},
    ),
    AblationArm(
        arm_id="act_selu_to_elu",
        description="Replace selu with elu.",
        claim_ids=("selu_vs_elu",),
        confidence="high",
        activation_map={"selu": "elu"},
    ),
    AblationArm(
        arm_id="fam_linear_to_nn",
        description="Replace linear family with nn family.",
        claim_ids=("linear_vs_nn",),
        confidence="high",
        family_map={"linear": "nn"},
    ),
    AblationArm(
        arm_id="fam_product_to_quadratic",
        description="Replace product family with quadratic family.",
        claim_ids=("quadratic_vs_product",),
        confidence="high",
        family_map={"product": "quadratic"},
    ),
    AblationArm(
        arm_id="fam_gp_to_nn",
        description="Replace gp family with nn family.",
        claim_ids=("gp_vs_nn",),
        confidence="approximate",
        family_map={"gp": "nn"},
    ),
    AblationArm(
        arm_id="fam_em_to_discretization",
        description="Replace em family with discretization family.",
        claim_ids=("em_vs_discretization",),
        confidence="approximate",
        family_map={"em": "discretization"},
    ),
    AblationArm(
        arm_id="fam_tree_to_discretization",
        description="Replace tree family with discretization family.",
        claim_ids=("tree_vs_discretization",),
        confidence="approximate",
        family_map={"tree": "discretization"},
    ),
    AblationArm(
        arm_id="agg_logsumexp_to_max",
        description="Replace logsumexp aggregation with max aggregation.",
        claim_ids=("max_vs_logsumexp",),
        confidence="approximate",
        aggregation_map={"logsumexp": "max"},
    ),
    AblationArm(
        arm_id="act_argsort_to_rank",
        description="Replace argsort activation with rank activation.",
        claim_ids=("rank_vs_argsort",),
        confidence="approximate",
        activation_map={"argsort": "rank"},
    ),
)

_HIGH_CONFIDENCE_ARM_IDS: tuple[str, ...] = (
    "act_sigmoid_to_tanh",
    "act_sign_to_heaviside",
    "act_logsigmoid_to_softplus",
    "act_relu6_to_hardtanh",
    "act_selu_to_elu",
    "fam_linear_to_nn",
    "fam_product_to_quadratic",
)


def _torch_generator(seed: int) -> torch.Generator:
    """Return a deterministic CPU generator."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _pair_key(left: str, right: str) -> tuple[str, str]:
    """Return a sorted pair key."""

    return (left, right) if left <= right else (right, left)


def _pair_label(
    *,
    mean_cosine: float,
    min_cosine: float,
    mean_affine_rmse: float,
    thresholds: AuditThresholds,
) -> str:
    """Classify overlap level for one pair."""

    if mean_affine_rmse <= thresholds.exact_affine_rmse:
        return "exact_affine_equivalent"
    if mean_cosine >= thresholds.near_cosine and mean_affine_rmse <= thresholds.near_affine_rmse:
        return "near_equivalent"
    if min_cosine >= thresholds.near_cosine:
        return "near_equivalent"
    return "distinct"


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Return finite float32 flattened view."""

    return torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).flatten()


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute cosine similarity between flattened tensors."""

    left_f = _flatten(left)
    right_f = _flatten(right)
    denom = float(torch.norm(left_f).item() * torch.norm(right_f).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(left_f, right_f).item() / denom)


def _affine_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute normalized RMSE of best affine fit: right ~= a * left + b."""

    x = _flatten(left)
    y = _flatten(right)

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    x_var = float(torch.mean(x_centered * x_centered).item())

    if x_var <= 1e-12:
        pred = torch.full_like(y, fill_value=float(y_mean.item()))
    else:
        cov_xy = float(torch.mean(x_centered * y_centered).item())
        alpha = cov_xy / x_var
        beta = float(y_mean.item()) - (alpha * float(x_mean.item()))
        pred = (alpha * x) + beta

    rmse = float(torch.sqrt(torch.mean((y - pred) ** 2)).item())
    y_scale = float(torch.std(y, correction=0).clamp_min(1e-6).item())
    return rmse / y_scale


def _spearman_like(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute an argsort-based rank correlation proxy."""

    x = _flatten(left)
    y = _flatten(right)

    rank_x = torch.argsort(torch.argsort(x))
    rank_y = torch.argsort(torch.argsort(y))
    rx = rank_x.to(torch.float32)
    ry = rank_y.to(torch.float32)

    rx = rx - torch.mean(rx)
    ry = ry - torch.mean(ry)
    denom = float(torch.norm(rx).item() * torch.norm(ry).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(rx, ry).item() / denom)


def _numeric_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Return numeric overlap metrics for one pair."""

    diff = _flatten(left) - _flatten(right)
    return {
        "cosine_similarity": _cosine_similarity(left, right),
        "affine_rmse": _affine_rmse(left, right),
        "rank_correlation": _spearman_like(left, right),
        "max_abs_diff": float(torch.max(torch.abs(diff)).item()),
    }


def _label_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Return label/discrete agreement metrics for one pair."""

    left_i64 = left.to(torch.int64).flatten()
    right_i64 = right.to(torch.int64).flatten()
    agreement = float(torch.mean((left_i64 == right_i64).to(torch.float32)).item())
    ari = float(adjusted_rand_score(left_i64.cpu().tolist(), right_i64.cpu().tolist()))
    return {"agreement": agreement, "adjusted_rand_index": ari}


def _aggregate_pair_runs(
    pair_runs: dict[tuple[str, str], list[dict[str, float]]],
    *,
    thresholds: AuditThresholds,
) -> list[dict[str, Any]]:
    """Aggregate pair metrics across seeds and assign labels."""

    records: list[dict[str, Any]] = []
    for (left, right), metric_runs in pair_runs.items():
        if not metric_runs:
            continue

        metric_names = sorted(metric_runs[0].keys())
        aggregated: dict[str, dict[str, float]] = {}
        for name in metric_names:
            values = [float(run[name]) for run in metric_runs]
            aggregated[name] = {
                "mean": float(sum(values) / len(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }

        cosine_info = aggregated.get("cosine_similarity")
        affine_info = aggregated.get("affine_rmse")
        if cosine_info is not None and affine_info is not None:
            pair_label = _pair_label(
                mean_cosine=float(cosine_info["mean"]),
                min_cosine=float(cosine_info["min"]),
                mean_affine_rmse=float(affine_info["mean"]),
                thresholds=thresholds,
            )
        else:
            pair_label = "descriptive_only"

        records.append(
            {
                "left": left,
                "right": right,
                "label": pair_label,
                "metrics": aggregated,
            }
        )

    records.sort(
        key=lambda record: (
            float(record.get("metrics", {}).get("cosine_similarity", {}).get("mean", -1e9)),
            -float(record.get("metrics", {}).get("affine_rmse", {}).get("mean", 1e9)),
            record["left"],
            record["right"],
        ),
        reverse=True,
    )
    return records


def _runtime_activation_names() -> tuple[str, ...]:
    """Return activation names from runtime source-of-truth."""

    return tuple(activations_module.fixed_activation_names())


def _activation_output(name: str, x: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply one fixed activation using the runtime standardize/affine envelope."""

    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    generator = _torch_generator(seed)
    y = standardize(y)
    scale = log_uniform(generator, 1.0, 10.0, str(y.device))
    row_index = int(torch.randint(0, y.shape[0], (1,), generator=generator).item())
    anchor = y[row_index : row_index + 1]
    y = scale * (y - anchor)
    y = activations_module._fixed_activation(y, name)
    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    y = standardize(y)
    return y


def _family_output(
    family: MechanismFamily,
    x: torch.Tensor,
    *,
    seed: int,
    out_dim: int,
) -> torch.Tensor:
    """Apply one function family with fixed output width."""

    generator = _torch_generator(seed)
    y = apply_random_function(x, generator, out_dim=out_dim, function_type=family)
    y = torch.nan_to_num(y.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    y = standardize(y)
    return y


def _product_linear_linear_output(x: torch.Tensor, *, seed: int, out_dim: int) -> torch.Tensor:
    """Construct product(linear, linear) output for direct comparison against quadratic."""

    left = _family_output("linear", x, seed=seed, out_dim=out_dim)
    right = _family_output("linear", x, seed=seed + 19_991, out_dim=out_dim)
    out = left * right
    out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    out = torch.clamp(out, -1e6, 1e6)
    return standardize(out)


def _build_activation_track(
    *,
    seed: int,
    n_seeds: int,
    n_rows: int,
    n_cols: int,
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    """Build activation pair-overlap report."""

    activations = _runtime_activation_names()
    pair_runs: dict[tuple[str, str], list[dict[str, float]]] = {}
    for seed_offset in range(n_seeds):
        x_seed = seed + (13 * seed_offset)
        x = torch.randn(n_rows, n_cols, generator=_torch_generator(x_seed))
        activation_seed = seed + 10_000 + (101 * seed_offset)
        outputs = {name: _activation_output(name, x, seed=activation_seed) for name in activations}

        for idx, left in enumerate(activations):
            for right in activations[idx + 1 :]:
                key = _pair_key(left, right)
                pair_runs.setdefault(key, []).append(
                    _numeric_metrics(outputs[left], outputs[right])
                )

    pairs = _aggregate_pair_runs(pair_runs, thresholds=thresholds)
    return {
        "families": list(activations),
        "pair_count": int(len(pairs)),
        "pairs": pairs,
    }


def _build_family_track(
    *,
    seed: int,
    n_seeds: int,
    n_rows: int,
    n_cols: int,
    out_dim: int,
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    """Build mechanism-family overlap report."""

    pair_runs: dict[tuple[str, str], list[dict[str, float]]] = {}
    for seed_offset in range(n_seeds):
        x_seed = seed + (5_003 * seed_offset)
        x = torch.randn(n_rows, n_cols, generator=_torch_generator(x_seed))
        outputs = {
            family: _family_output(
                family,
                x,
                seed=seed + 20_000 + (313 * seed_offset) + idx,
                out_dim=out_dim,
            )
            for idx, family in enumerate(_MECHANISM_FAMILIES)
        }

        for idx, left in enumerate(_MECHANISM_FAMILIES):
            for right in _MECHANISM_FAMILIES[idx + 1 :]:
                key = _pair_key(left, right)
                pair_runs.setdefault(key, []).append(
                    _numeric_metrics(outputs[left], outputs[right])
                )

    constructed_pair_runs: dict[tuple[str, str], list[dict[str, float]]] = {}
    for seed_offset in range(n_seeds):
        x_seed = seed + (7_001 * seed_offset)
        x = torch.randn(n_rows, n_cols, generator=_torch_generator(x_seed))
        product_ll = _product_linear_linear_output(
            x,
            seed=seed + 33_000 + (97 * seed_offset),
            out_dim=out_dim,
        )
        quadratic = _family_output(
            "quadratic",
            x,
            seed=seed + 37_000 + (97 * seed_offset),
            out_dim=out_dim,
        )
        key = _pair_key("product_linear_linear", "quadratic")
        constructed_pair_runs.setdefault(key, []).append(_numeric_metrics(product_ll, quadratic))

    pairs = _aggregate_pair_runs(pair_runs, thresholds=thresholds)
    constructed_pairs = _aggregate_pair_runs(constructed_pair_runs, thresholds=thresholds)
    return {
        "families": list(_MECHANISM_FAMILIES),
        "pair_count": int(len(pairs)),
        "pairs": pairs,
        "constructed_pairs": constructed_pairs,
    }


def _build_aggregation_track(
    *,
    seed: int,
    n_seeds: int,
    n_rows: int,
    out_dim: int,
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    """Build max-vs-logsumexp overlap report across scale factors."""

    by_scale: list[dict[str, Any]] = []
    for scale in _AGGREGATION_SCALE_FACTORS:
        runs: list[dict[str, float]] = []
        for seed_offset in range(n_seeds):
            base = torch.randn(n_rows, 5, out_dim, generator=_torch_generator(seed + seed_offset))
            stacked = scale * base
            max_out = _aggregate_parent_outputs(stacked, aggregation_kind="max")
            lse_out = _aggregate_parent_outputs(stacked, aggregation_kind="logsumexp")
            max_out = standardize(max_out)
            lse_out = standardize(lse_out)
            runs.append(_numeric_metrics(max_out, lse_out))

        metrics = _aggregate_pair_runs(
            {_pair_key("max", "logsumexp"): runs},
            thresholds=thresholds,
        )
        by_scale.append(
            {
                "scale_factor": float(scale),
                "pair_metrics": metrics[0] if metrics else {},
            }
        )

    return {"scales": by_scale}


def _nn_linear_degenerate_probability(*, trials: int, seed: int) -> float:
    """Estimate probability that sampled NN path reduces to one affine layer."""

    generator = _torch_generator(seed)
    degenerate_count = 0
    for _ in range(trials):
        n_layers = int(torch.randint(1, 4, (1,), generator=generator).item())
        has_pre_activation = bool(torch.rand(1, generator=generator).item() < 0.5)
        has_post_activation = bool(torch.rand(1, generator=generator).item() < 0.5)
        if n_layers == 1 and (not has_pre_activation) and (not has_post_activation):
            degenerate_count += 1
    return float(degenerate_count / max(1, trials))


def _find_pair(
    pairs: list[dict[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, Any] | None:
    """Return pair record when available."""

    key = _pair_key(left, right)
    for pair in pairs:
        if _pair_key(str(pair["left"]), str(pair["right"])) == key:
            return pair
    return None


def _set_equivalence_metrics(n_cols: int) -> dict[str, float]:
    """Return set-equivalence metrics for argsort and rank output ranges."""

    argsort_values = set(range(n_cols))
    rank_values = set(range(1, n_cols + 1))
    overlap = len(argsort_values & rank_values)
    union = len(argsort_values | rank_values)
    jaccard = float(overlap / union) if union > 0 else 0.0
    return {
        "argsort_min": float(min(argsort_values)),
        "argsort_max": float(max(argsort_values)),
        "rank_min": float(min(rank_values)),
        "rank_max": float(max(rank_values)),
        "set_jaccard": float(jaccard),
    }


def _claim_status_from_pair(
    *,
    claim_type: str,
    pair_record: dict[str, Any] | None,
) -> str:
    """Resolve a claim status from a pair overlap record."""

    if pair_record is None:
        return "unavailable"
    label = str(pair_record.get("label", ""))
    if claim_type == "exact":
        return "supported" if label == "exact_affine_equivalent" else "not_supported"
    if claim_type == "near":
        if label in {"exact_affine_equivalent", "near_equivalent"}:
            return "supported"
        return "not_supported"
    return "informational"


def _evaluate_hypothesis_registry(
    *,
    activation_pairs: list[dict[str, Any]],
    family_pairs: list[dict[str, Any]],
    aggregation_track: dict[str, Any],
    runtime_activations: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Evaluate hypothesis registry against current runtime tracks."""

    activation_lookup = {
        _pair_key(str(item["left"]), str(item["right"])): item for item in activation_pairs
    }
    family_lookup = {
        _pair_key(str(item["left"]), str(item["right"])): item for item in family_pairs
    }

    aggregation_pair_records = [
        entry.get("pair_metrics", {}) for entry in aggregation_track.get("scales", [])
    ]
    aggregation_labels = {
        str(record.get("label"))
        for record in aggregation_pair_records
        if isinstance(record, dict) and record.get("label") is not None
    }

    families_available = set(_MECHANISM_FAMILIES)
    activations_available = set(runtime_activations)
    evaluated: list[dict[str, Any]] = []

    for claim in _HYPOTHESIS_CLAIMS:
        claim_payload: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "track": claim.track,
            "left": claim.left,
            "right": claim.right,
            "claim_type": claim.claim_type,
            "confidence": claim.confidence,
            "description": claim.description,
        }

        if claim.track == "activations":
            if claim.left not in activations_available or claim.right not in activations_available:
                claim_payload["status"] = "not_applicable_runtime_absent"
                claim_payload["pair_record"] = None
                evaluated.append(claim_payload)
                continue
            pair_record = activation_lookup.get(_pair_key(claim.left, claim.right))
            claim_payload["pair_record"] = pair_record
            claim_payload["status"] = _claim_status_from_pair(
                claim_type=claim.claim_type,
                pair_record=pair_record,
            )
            evaluated.append(claim_payload)
            continue

        if claim.track == "function_families":
            if claim.left not in families_available or claim.right not in families_available:
                claim_payload["status"] = "not_applicable_runtime_absent"
                claim_payload["pair_record"] = None
                evaluated.append(claim_payload)
                continue
            pair_record = family_lookup.get(_pair_key(claim.left, claim.right))
            claim_payload["pair_record"] = pair_record
            claim_payload["status"] = _claim_status_from_pair(
                claim_type=claim.claim_type,
                pair_record=pair_record,
            )
            evaluated.append(claim_payload)
            continue

        if claim.track == "aggregation":
            status = "informational"
            if claim.claim_type == "approximate":
                if (
                    "near_equivalent" in aggregation_labels
                    or "exact_affine_equivalent" in aggregation_labels
                ):
                    status = "supported"
                else:
                    status = "not_supported"
            claim_payload["status"] = status
            claim_payload["pair_record"] = None
            claim_payload["aggregation_labels_by_scale"] = [
                {
                    "scale_factor": entry.get("scale_factor"),
                    "label": (entry.get("pair_metrics") or {}).get("label"),
                }
                for entry in aggregation_track.get("scales", [])
            ]
            evaluated.append(claim_payload)
            continue

        claim_payload["status"] = "unavailable"
        claim_payload["pair_record"] = None
        evaluated.append(claim_payload)

    return evaluated


def generate_effective_diversity_report(
    *,
    seed: int = 2_026_0304,
    n_seeds: int = 8,
    n_rows: int = 2_048,
    n_cols: int = 16,
    out_dim: int = 16,
    nn_degenerate_trials: int = 50_000,
    thresholds: AuditThresholds = AuditThresholds(),
) -> dict[str, Any]:
    """Generate a deterministic local effective-diversity audit report."""

    activation_track = _build_activation_track(
        seed=seed,
        n_seeds=n_seeds,
        n_rows=n_rows,
        n_cols=n_cols,
        thresholds=thresholds,
    )
    family_track = _build_family_track(
        seed=seed,
        n_seeds=n_seeds,
        n_rows=n_rows,
        n_cols=n_cols,
        out_dim=out_dim,
        thresholds=thresholds,
    )
    aggregation_track = _build_aggregation_track(
        seed=seed,
        n_seeds=n_seeds,
        n_rows=n_rows,
        out_dim=out_dim,
        thresholds=thresholds,
    )

    runtime_activations = tuple(activation_track["families"])
    activation_pairs = activation_track["pairs"]
    family_pairs = family_track["pairs"]
    constructed_pairs = family_track["constructed_pairs"]
    logsumexp_profile = {
        f"scale_{entry['scale_factor']}": entry["pair_metrics"]
        for entry in aggregation_track["scales"]
    }

    x_probe = torch.randn(n_rows, n_cols, generator=_torch_generator(seed + 888))
    argsort_probe = _activation_output("argsort", x_probe, seed=seed + 1_111)
    rank_probe = _activation_output("rank", x_probe, seed=seed + 1_112)
    ordinal_label_metrics = _label_metrics(argsort_probe.round(), rank_probe.round())

    hypothesis_checks = {
        "sigmoid_vs_tanh": _find_pair(activation_pairs, left="sigmoid", right="tanh"),
        "sign_vs_heaviside": _find_pair(activation_pairs, left="sign", right="heaviside"),
        "softplus_vs_logsigmoid": _find_pair(
            activation_pairs,
            left="softplus",
            right="logsigmoid",
        ),
        "relu6_vs_hardtanh": _find_pair(activation_pairs, left="relu6", right="hardtanh"),
        "elu_vs_selu": _find_pair(activation_pairs, left="elu", right="selu"),
        "nn_vs_linear": _find_pair(family_pairs, left="nn", right="linear"),
        "quadratic_vs_product": _find_pair(family_pairs, left="quadratic", right="product"),
        "gp_vs_nn": _find_pair(family_pairs, left="gp", right="nn"),
        "product_linear_linear_vs_quadratic": _find_pair(
            constructed_pairs,
            left="product_linear_linear",
            right="quadratic",
        ),
        "discretization_vs_em": _find_pair(family_pairs, left="discretization", right="em"),
        "tree_vs_discretization": _find_pair(family_pairs, left="tree", right="discretization"),
        "max_vs_logsumexp_profile": logsumexp_profile,
        "nn_linear_degenerate_probability": {
            "analytic": float(1.0 / 12.0),
            "empirical": float(
                _nn_linear_degenerate_probability(
                    trials=nn_degenerate_trials,
                    seed=seed + 90_123,
                )
            ),
            "trials": int(nn_degenerate_trials),
        },
        "argsort_vs_rank": {
            "set_equivalence": _set_equivalence_metrics(n_cols),
            "label_metrics": ordinal_label_metrics,
            "pair_record": _find_pair(activation_pairs, left="argsort", right="rank"),
        },
    }

    hypothesis_registry = _evaluate_hypothesis_registry(
        activation_pairs=activation_pairs,
        family_pairs=family_pairs,
        aggregation_track=aggregation_track,
        runtime_activations=runtime_activations,
    )

    return {
        "schema_name": "dagzoo_effective_diversity_audit",
        "schema_version": 2,
        "config": {
            "seed": int(seed),
            "n_seeds": int(n_seeds),
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "out_dim": int(out_dim),
            "nn_degenerate_trials": int(nn_degenerate_trials),
            "thresholds": {
                "exact_affine_rmse": float(thresholds.exact_affine_rmse),
                "near_cosine": float(thresholds.near_cosine),
                "near_affine_rmse": float(thresholds.near_affine_rmse),
            },
            "runtime_activation_names": list(runtime_activations),
        },
        "tracks": {
            "activations": activation_track,
            "function_families": family_track,
            "aggregation": aggregation_track,
        },
        "hypothesis_checks": hypothesis_checks,
        "hypothesis_registry": hypothesis_registry,
    }


def _format_pair_line(pair: dict[str, Any]) -> str:
    """Return concise markdown line for one pair."""

    metrics = pair.get("metrics", {})
    cosine = metrics.get("cosine_similarity", {}).get("mean")
    affine_rmse = metrics.get("affine_rmse", {}).get("mean")
    label = pair.get("label")
    cosine_text = "n/a" if cosine is None else f"{float(cosine):.4f}"
    rmse_text = "n/a" if affine_rmse is None else f"{float(affine_rmse):.6f}"
    return (
        f"- `{pair['left']}` vs `{pair['right']}`: label={label}, "
        f"cosine_mean={cosine_text}, affine_rmse_mean={rmse_text}"
    )


def format_effective_diversity_markdown(report: dict[str, Any]) -> str:
    """Render a concise markdown summary for human review."""

    cfg = report["config"]
    tracks = report["tracks"]
    activation_pairs = tracks["activations"]["pairs"]
    family_pairs = tracks["function_families"]["pairs"]
    constructed_pairs = tracks["function_families"]["constructed_pairs"]
    aggregation_scales = tracks["aggregation"]["scales"]
    registry = report.get("hypothesis_registry", [])

    top_activation = activation_pairs[:8]
    top_family = family_pairs[:8]
    top_constructed = constructed_pairs[:5]

    lines = [
        "# Effective Diversity Audit",
        "",
        "No generator behavior/config/schema was changed by this audit.",
        "",
        "## Run Config",
        f"- seed: `{cfg['seed']}`",
        f"- n_seeds: `{cfg['n_seeds']}`",
        f"- n_rows: `{cfg['n_rows']}`",
        f"- n_cols: `{cfg['n_cols']}`",
        f"- out_dim: `{cfg['out_dim']}`",
        f"- runtime activations: `{len(cfg.get('runtime_activation_names', []))}`",
        "",
        "## Activation Pair Overlap (Top by Cosine)",
    ]
    lines.extend(_format_pair_line(pair) for pair in top_activation)
    lines.append("")
    lines.append("## Function Family Overlap (Top by Cosine)")
    lines.extend(_format_pair_line(pair) for pair in top_family)
    if top_constructed:
        lines.append("")
        lines.append("## Constructed Family Checks")
        lines.extend(_format_pair_line(pair) for pair in top_constructed)

    lines.append("")
    lines.append("## Aggregation: `max` vs `logsumexp`")
    for entry in aggregation_scales:
        pair = entry.get("pair_metrics", {})
        label = pair.get("label", "unknown")
        cosine = pair.get("metrics", {}).get("cosine_similarity", {}).get("mean", float("nan"))
        rmse = pair.get("metrics", {}).get("affine_rmse", {}).get("mean", float("nan"))
        lines.append(
            f"- scale={entry['scale_factor']}: label={label}, "
            f"cosine_mean={float(cosine):.4f}, affine_rmse_mean={float(rmse):.6f}"
        )

    lines.append("")
    lines.append("## Hypothesis Registry")
    for claim in registry:
        lines.append(
            f"- `{claim['claim_id']}`: status={claim.get('status')}, "
            f"track={claim.get('track')}, type={claim.get('claim_type')}"
        )

    return "\n".join(lines)


def write_effective_diversity_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write local-overlap JSON + Markdown artifacts and return paths."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "equivalence_report.json"
    md_path = out_path / "equivalence_report.md"
    json_path.write_text(
        f"{json.dumps(sanitize_json(report), indent=2, sort_keys=True, allow_nan=False)}\n",
        encoding="utf-8",
    )
    md_path.write_text(format_effective_diversity_markdown(report), encoding="utf-8")
    return json_path, md_path


def _arm_by_id() -> dict[str, AblationArm]:
    """Return indexed ablation arm mapping."""

    return {arm.arm_id: arm for arm in _BASE_ABLATION_ARMS}


def _build_combined_arm(*, selected_arms: tuple[AblationArm, ...], arm_set: str) -> AblationArm:
    """Build merged combined arm for the selected one-at-a-time set."""

    activation_map: dict[str, str] = {}
    family_map: dict[MechanismFamily, MechanismFamily] = {}
    aggregation_map: dict[AggregationKind, AggregationKind] = {}

    for arm in selected_arms:
        for source, target in arm.activation_map.items():
            existing = activation_map.get(source)
            if existing is not None and existing != target:
                raise ValueError(
                    f"Conflicting combined activation mapping for {source!r}: {existing!r} vs {target!r}."
                )
            activation_map[source] = target
        for source, target in arm.family_map.items():
            existing = family_map.get(source)
            if existing is not None and existing != target:
                raise ValueError(
                    f"Conflicting combined family mapping for {source!r}: {existing!r} vs {target!r}."
                )
            family_map[source] = target
        for source, target in arm.aggregation_map.items():
            existing = aggregation_map.get(source)
            if existing is not None and existing != target:
                raise ValueError(
                    f"Conflicting combined aggregation mapping for {source!r}: {existing!r} vs {target!r}."
                )
            aggregation_map[source] = target

    combined_id = (
        "combined_high_confidence" if arm_set == "high_confidence" else "combined_all_claims"
    )
    combined_description = (
        "Combined high-confidence redundancy mappings."
        if arm_set == "high_confidence"
        else "Combined all redundancy mappings (including approximate claims)."
    )
    combined_claim_ids: list[str] = []
    for arm in selected_arms:
        combined_claim_ids.extend(list(arm.claim_ids))

    return AblationArm(
        arm_id=combined_id,
        description=combined_description,
        claim_ids=tuple(dict.fromkeys(combined_claim_ids)),
        confidence="high" if arm_set == "high_confidence" else "approximate",
        activation_map=activation_map,
        family_map=family_map,
        aggregation_map=aggregation_map,
    )


def _resolve_scale_arms(
    arm_set: str,
    *,
    runtime_activations: tuple[str, ...],
) -> tuple[AblationArm, ...]:
    """Resolve one-at-a-time plus combined ablation arms."""

    index = _arm_by_id()
    if arm_set == "high_confidence":
        selected = tuple(index[arm_id] for arm_id in _HIGH_CONFIDENCE_ARM_IDS)
    elif arm_set == "all_claims":
        selected = tuple(_BASE_ABLATION_ARMS)
    else:
        raise ValueError(f"Unsupported arm_set: {arm_set!r}")

    applicable_for_combined = tuple(
        arm
        for arm in selected
        if _arm_skip_reason(arm, runtime_activations=runtime_activations) is None
    )
    if not applicable_for_combined:
        return selected

    combined = _build_combined_arm(selected_arms=applicable_for_combined, arm_set=arm_set)
    return (*selected, combined)


def _arm_skip_reason(arm: AblationArm, *, runtime_activations: tuple[str, ...]) -> str | None:
    """Return skip reason when an ablation arm is not executable on current runtime."""

    activation_set = set(runtime_activations)
    for source, target in arm.activation_map.items():
        if source not in activation_set:
            return f"activation_source_missing:{source}"
        if target not in activation_set:
            return f"activation_target_missing:{target}"

    family_set = set(_MECHANISM_FAMILIES)
    for source, target in arm.family_map.items():
        if source not in family_set:
            return f"family_source_missing:{source}"
        if target not in family_set:
            return f"family_target_missing:{target}"

    for source, target in arm.aggregation_map.items():
        if source not in _AGGREGATION_KINDS:
            return f"aggregation_source_missing:{source}"
        if target not in _AGGREGATION_KINDS:
            return f"aggregation_target_missing:{target}"

    return None


def _coerce_float(value: object) -> float | None:
    """Parse finite float payload when possible."""

    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _mean_or_none(values: list[float]) -> float | None:
    """Return finite list mean or ``None``."""

    if not values:
        return None
    return float(sum(values) / len(values))


@dataclass(slots=True)
class _FilterTelemetry:
    """Streaming filter/postprocess-adjacent telemetry from emitted bundle metadata."""

    bundles: int = 0
    attempts_total: int = 0
    retries_total: int = 0
    filter_enabled_count: int = 0
    accepted_true_count: int = 0
    wins_ratio_values: list[float] = field(default_factory=list)
    threshold_effective_values: list[float] = field(default_factory=list)
    threshold_delta_values: list[float] = field(default_factory=list)
    n_valid_oob_values: list[float] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)

    def update(self, bundle: DatasetBundle) -> None:
        """Ingest one emitted bundle metadata payload."""

        self.bundles += 1
        metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}

        raw_attempt_used = metadata.get("attempt_used")
        attempt_used = 0
        if isinstance(raw_attempt_used, int):
            attempt_used = max(0, int(raw_attempt_used))
        self.retries_total += attempt_used
        self.attempts_total += attempt_used + 1

        filter_meta = metadata.get("filter")
        if not isinstance(filter_meta, dict):
            return

        if bool(filter_meta.get("enabled")):
            self.filter_enabled_count += 1

        if filter_meta.get("accepted") is True:
            self.accepted_true_count += 1

        wins_ratio = _coerce_float(filter_meta.get("wins_ratio"))
        if wins_ratio is not None:
            self.wins_ratio_values.append(wins_ratio)

        threshold_effective = _coerce_float(filter_meta.get("threshold_effective"))
        if threshold_effective is not None:
            self.threshold_effective_values.append(threshold_effective)

        threshold_delta = _coerce_float(filter_meta.get("threshold_delta"))
        if threshold_delta is not None:
            self.threshold_delta_values.append(threshold_delta)

        n_valid_oob = _coerce_float(filter_meta.get("n_valid_oob"))
        if n_valid_oob is not None:
            self.n_valid_oob_values.append(n_valid_oob)

        reason = filter_meta.get("reason")
        if isinstance(reason, str) and reason:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def build_summary(self) -> dict[str, Any]:
        """Finalize telemetry summary payload."""

        return {
            "num_bundles": int(self.bundles),
            "mean_attempts_per_emitted_bundle": (
                float(self.attempts_total / self.bundles) if self.bundles > 0 else None
            ),
            "mean_retries_per_emitted_bundle": (
                float(self.retries_total / self.bundles) if self.bundles > 0 else None
            ),
            "filter_enabled_fraction": (
                float(self.filter_enabled_count / self.bundles) if self.bundles > 0 else 0.0
            ),
            "accepted_true_fraction": (
                float(self.accepted_true_count / self.bundles) if self.bundles > 0 else 0.0
            ),
            "wins_ratio_mean": _mean_or_none(self.wins_ratio_values),
            "threshold_effective_mean": _mean_or_none(self.threshold_effective_values),
            "threshold_delta_mean": _mean_or_none(self.threshold_delta_values),
            "n_valid_oob_mean": _mean_or_none(self.n_valid_oob_values),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


def _coverage_config_for_scale(base_config: GeneratorConfig) -> CoverageAggregationConfig:
    """Create diagnostics aggregation config with quantiles needed for impact deltas."""

    cfg = build_diagnostics_aggregation_config(base_config.diagnostics)
    required_quantiles = {0.25, 0.5, 0.75}
    quantiles = tuple(sorted(set(float(q) for q in cfg.quantiles) | required_quantiles))
    return CoverageAggregationConfig(
        include_spearman=bool(cfg.include_spearman),
        histogram_bins=max(1, int(cfg.histogram_bins)),
        quantiles=quantiles,
        underrepresented_threshold=float(cfg.underrepresented_threshold),
        max_values_per_metric=cfg.max_values_per_metric,
        target_bands=dict(cfg.target_bands),
    )


def _adjust_family_mix(
    function_family_mix: dict[MechanismFamily, float] | None,
    *,
    source_family: MechanismFamily,
    target_family: MechanismFamily,
) -> dict[MechanismFamily, float] | None:
    """Ensure mapped explicit function family remains allowed under mix constraints."""

    if function_family_mix is None:
        return None
    if source_family == target_family:
        return function_family_mix
    if target_family in function_family_mix:
        return function_family_mix

    adjusted = dict(function_family_mix)
    source_weight = float(adjusted.get(source_family, 0.0))
    if source_weight > 0.0:
        adjusted[target_family] = source_weight
    else:
        adjusted[target_family] = max(1e-6, float(adjusted.get(target_family, 0.0)))
    return adjusted


@contextmanager
def _runtime_override_context(arm: AblationArm) -> Iterator[None]:
    """Temporarily patch runtime function dispatch for one ablation arm."""

    if not arm.activation_map and not arm.family_map and not arm.aggregation_map:
        yield
        return

    patches: list[tuple[Any, str, Any]] = []

    def _patch_attr(module: Any, attr: str, value: Any) -> None:
        patches.append((module, attr, getattr(module, attr)))
        setattr(module, attr, value)

    if arm.activation_map:
        original_fixed_activation = activations_module._fixed_activation

        def _mapped_fixed_activation(x: torch.Tensor, name: str) -> torch.Tensor:
            mapped_name = arm.activation_map.get(name, name)
            return original_fixed_activation(x, mapped_name)

        _patch_attr(activations_module, "_fixed_activation", _mapped_fixed_activation)

    if arm.family_map:
        original_apply_random_function = random_functions_module.apply_random_function
        original_sample_function_family = random_functions_module._sample_function_family

        def _mapped_sample_function_family(
            generator: torch.Generator | None = None,
            *,
            keyed_rng: Any = None,
            mechanism_logit_tilt: float,
            function_family_mix: dict[MechanismFamily, float] | None = None,
            device: str | None = None,
        ) -> MechanismFamily:
            sampled = original_sample_function_family(
                generator,
                keyed_rng=keyed_rng,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=device,
            )
            return arm.family_map.get(sampled, sampled)

        def _mapped_apply_random_function(
            x: torch.Tensor,
            generator: torch.Generator,
            *,
            out_dim: int | None = None,
            function_type: MechanismFamily | None = None,
            mechanism_logit_tilt: float = 0.0,
            function_family_mix: dict[MechanismFamily, float] | None = None,
            noise_sigma_multiplier: float = 1.0,
            noise_spec: Any = None,
            _keyed_root: Any = None,
        ) -> torch.Tensor:
            mapped_type = function_type
            adjusted_mix = function_family_mix
            if function_type is not None:
                mapped_type = arm.family_map.get(function_type, function_type)
                adjusted_mix = _adjust_family_mix(
                    function_family_mix,
                    source_family=function_type,
                    target_family=mapped_type,
                )
            return original_apply_random_function(
                x,
                generator,
                out_dim=out_dim,
                function_type=mapped_type,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=adjusted_mix,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                _keyed_root=_keyed_root,
            )

        _patch_attr(
            random_functions_module, "_sample_function_family", _mapped_sample_function_family
        )
        _patch_attr(
            execution_semantics_module,
            "sample_function_family",
            _mapped_sample_function_family,
        )
        _patch_attr(random_functions_module, "apply_random_function", _mapped_apply_random_function)
        _patch_attr(multi_module, "apply_random_function", _mapped_apply_random_function)

    if arm.aggregation_map:
        original_aggregate = multi_module._aggregate_parent_outputs
        original_incremental_aggregate = multi_module._aggregate_incrementally
        original_batch_aggregate = fixed_layout_batched_module._aggregate_parent_outputs_batch
        original_batch_incremental_aggregate = (
            fixed_layout_batched_module._aggregate_batch_incrementally
        )

        def _mapped_aggregate_parent_outputs(
            stacked: torch.Tensor,
            *,
            aggregation_kind: AggregationKind,
        ) -> torch.Tensor:
            mapped_kind = arm.aggregation_map.get(aggregation_kind, aggregation_kind)
            return original_aggregate(stacked, aggregation_kind=mapped_kind)

        def _mapped_aggregate_incrementally(
            aggregate: torch.Tensor,
            transformed_output: torch.Tensor,
            *,
            aggregation_kind: AggregationKind,
        ) -> torch.Tensor:
            mapped_kind = arm.aggregation_map.get(aggregation_kind, aggregation_kind)
            return original_incremental_aggregate(
                aggregate,
                transformed_output,
                aggregation_kind=mapped_kind,
            )

        def _mapped_aggregate_parent_outputs_batch(
            stacked: torch.Tensor,
            *,
            aggregation_kind: AggregationKind,
        ) -> torch.Tensor:
            mapped_kind = arm.aggregation_map.get(aggregation_kind, aggregation_kind)
            return original_batch_aggregate(stacked, aggregation_kind=mapped_kind)

        def _mapped_aggregate_batch_incrementally(
            aggregate: torch.Tensor,
            transformed_output: torch.Tensor,
            *,
            aggregation_kind: AggregationKind,
        ) -> torch.Tensor:
            mapped_kind = arm.aggregation_map.get(aggregation_kind, aggregation_kind)
            return original_batch_incremental_aggregate(
                aggregate,
                transformed_output,
                aggregation_kind=mapped_kind,
            )

        _patch_attr(multi_module, "_aggregate_parent_outputs", _mapped_aggregate_parent_outputs)
        _patch_attr(multi_module, "_aggregate_incrementally", _mapped_aggregate_incrementally)
        _patch_attr(
            fixed_layout_batched_module,
            "_aggregate_parent_outputs_batch",
            _mapped_aggregate_parent_outputs_batch,
        )
        _patch_attr(
            fixed_layout_batched_module,
            "_aggregate_batch_incrementally",
            _mapped_aggregate_batch_incrementally,
        )

    try:
        yield
    finally:
        for module, attr, original in reversed(patches):
            setattr(module, attr, original)


def _run_scale_arm(
    *,
    base_config: GeneratorConfig,
    arm: AblationArm,
    n_datasets_per_arm: int,
    seed: int,
    device: str | None,
    runtime_activations: tuple[str, ...],
) -> dict[str, Any]:
    """Run one scale-impact arm and return diagnostics/filter summaries."""

    skip_reason = _arm_skip_reason(arm, runtime_activations=runtime_activations)
    if skip_reason is not None:
        return {
            "arm_id": arm.arm_id,
            "description": arm.description,
            "claim_ids": list(arm.claim_ids),
            "confidence": arm.confidence,
            "status": "skipped",
            "skip_reason": skip_reason,
            "n_datasets_requested": int(n_datasets_per_arm),
            "n_datasets_emitted": 0,
            "diagnostics_summary": None,
            "filter_summary": None,
        }

    arm_config = clone_generator_config(base_config, revalidate=False)
    coverage_aggregator = CoverageAggregator(_coverage_config_for_scale(arm_config))
    filter_telemetry = _FilterTelemetry()
    emitted = 0
    t0 = time.monotonic()
    log_interval = max(1, n_datasets_per_arm // 20)  # ~5% increments

    with _runtime_override_context(arm):
        for bundle in generate_batch_iter(
            arm_config,
            num_datasets=n_datasets_per_arm,
            seed=seed,
            device=device,
        ):
            coverage_aggregator.update_bundle(bundle)
            filter_telemetry.update(bundle)
            emitted += 1
            if emitted % log_interval == 0 or emitted == n_datasets_per_arm:
                elapsed = time.monotonic() - t0
                rate = emitted / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{arm.arm_id}] {emitted}/{n_datasets_per_arm} datasets "
                    f"({100 * emitted / n_datasets_per_arm:.0f}%) "
                    f"[{elapsed:.1f}s, {rate:.1f} ds/s]",
                    file=sys.stderr,
                    flush=True,
                )

    arm_elapsed = time.monotonic() - t0
    print(
        f"  [{arm.arm_id}] done — {emitted} datasets in {arm_elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    return {
        "arm_id": arm.arm_id,
        "description": arm.description,
        "claim_ids": list(arm.claim_ids),
        "confidence": arm.confidence,
        "status": "executed",
        "n_datasets_requested": int(n_datasets_per_arm),
        "n_datasets_emitted": int(emitted),
        "diagnostics_summary": coverage_aggregator.build_summary(),
        "filter_summary": filter_telemetry.build_summary(),
    }


def _metric_quantile(summary: dict[str, Any], metric: str, key: str) -> float | None:
    """Read one metric quantile from coverage summary."""

    metric_payload = (summary.get("metrics") or {}).get(metric)
    if not isinstance(metric_payload, dict):
        return None
    quantiles = metric_payload.get("quantiles")
    if not isinstance(quantiles, dict):
        return None
    return _coerce_float(quantiles.get(key))


def _metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    """Read one metric mean from coverage summary."""

    metric_payload = (summary.get("metrics") or {}).get(metric)
    if not isinstance(metric_payload, dict):
        return None
    return _coerce_float(metric_payload.get("mean"))


def _metric_shift_pct(
    *,
    baseline_summary: dict[str, Any],
    arm_summary: dict[str, Any],
    metric: str,
) -> float | None:
    """Compute weighted metric shift percentage for one metric."""

    base_mean = _metric_mean(baseline_summary, metric)
    arm_mean = _metric_mean(arm_summary, metric)
    base_p50 = _metric_quantile(baseline_summary, metric, "p50")
    arm_p50 = _metric_quantile(arm_summary, metric, "p50")
    base_p25 = _metric_quantile(baseline_summary, metric, "p25")
    base_p75 = _metric_quantile(baseline_summary, metric, "p75")

    if (
        base_mean is None
        or arm_mean is None
        or base_p50 is None
        or arm_p50 is None
        or base_p25 is None
        or base_p75 is None
    ):
        return None

    iqr = abs(float(base_p75) - float(base_p25))
    scale = max(abs(float(base_mean)), iqr, 1e-6)
    weighted = (0.6 * abs(float(arm_mean) - float(base_mean)) / scale) + (
        0.4 * abs(float(arm_p50) - float(base_p50)) / scale
    )
    return float(weighted * 100.0)


def _compare_arm_to_baseline(
    *,
    arm_result: dict[str, Any],
    baseline_result: dict[str, Any],
    meaningful_threshold_pct: float,
) -> dict[str, Any]:
    """Compare one arm with baseline and compute composite diversity shift score."""

    if arm_result.get("status") != "executed":
        return {
            "arm_id": arm_result.get("arm_id"),
            "status": "skipped",
            "skip_reason": arm_result.get("skip_reason"),
            "metric_shift_pct": {},
            "composite_shift_pct": None,
            "meaningful_threshold_pct": float(meaningful_threshold_pct),
            "is_meaningful": None,
        }

    baseline_summary = baseline_result.get("diagnostics_summary")
    arm_summary = arm_result.get("diagnostics_summary")
    if not isinstance(baseline_summary, dict) or not isinstance(arm_summary, dict):
        return {
            "arm_id": arm_result.get("arm_id"),
            "status": "insufficient_metrics",
            "metric_shift_pct": {},
            "composite_shift_pct": None,
            "meaningful_threshold_pct": float(meaningful_threshold_pct),
            "is_meaningful": None,
        }

    metric_shift_pct: dict[str, float] = {}
    for metric in _SCALE_CORE_METRICS:
        shift = _metric_shift_pct(
            baseline_summary=baseline_summary,
            arm_summary=arm_summary,
            metric=metric,
        )
        if shift is not None:
            metric_shift_pct[metric] = float(shift)

    if not metric_shift_pct:
        return {
            "arm_id": arm_result.get("arm_id"),
            "status": "insufficient_metrics",
            "metric_shift_pct": {},
            "composite_shift_pct": None,
            "meaningful_threshold_pct": float(meaningful_threshold_pct),
            "is_meaningful": None,
        }

    composite = float(median(metric_shift_pct.values()))
    return {
        "arm_id": arm_result.get("arm_id"),
        "status": "evaluated",
        "metric_shift_pct": metric_shift_pct,
        "composite_shift_pct": composite,
        "meaningful_threshold_pct": float(meaningful_threshold_pct),
        "is_meaningful": bool(composite >= meaningful_threshold_pct),
    }


def generate_effective_diversity_scale_report(
    *,
    base_config: GeneratorConfig,
    arm_set: str = "high_confidence",
    suite: str = "standard",
    num_datasets_per_arm: int | None = None,
    seed: int = 2_026_0304,
    meaningful_threshold_pct: float = 5.0,
    device: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run dataset-scale overlap impact audit through full generation/postprocess/filter path."""

    normalized_suite = str(suite).strip().lower()
    if normalized_suite not in _SCALE_SUITE_DATASETS:
        raise ValueError(f"Unsupported diversity-audit suite: {suite!r}")

    datasets_per_arm = (
        int(num_datasets_per_arm)
        if num_datasets_per_arm is not None
        else int(_SCALE_SUITE_DATASETS[normalized_suite])
    )
    if datasets_per_arm <= 0:
        raise ValueError(f"num_datasets_per_arm must be > 0, got {datasets_per_arm}")

    runtime_activations = _runtime_activation_names()
    selected_arms = _resolve_scale_arms(arm_set, runtime_activations=runtime_activations)
    total_arms = len(selected_arms) + 1  # +1 for baseline

    print(
        f"\n=== Scale audit: {total_arms} arms x {datasets_per_arm} datasets "
        f"(suite={normalized_suite}, arm_set={arm_set}) ===",
        file=sys.stderr,
        flush=True,
    )

    baseline_arm = AblationArm(
        arm_id="baseline",
        description="Unmodified runtime baseline.",
        confidence="reference",
    )
    print(f"\n[1/{total_arms}] Running baseline arm...", file=sys.stderr, flush=True)
    scale_t0 = time.monotonic()
    baseline_result = _run_scale_arm(
        base_config=base_config,
        arm=baseline_arm,
        n_datasets_per_arm=datasets_per_arm,
        seed=seed,
        device=device,
        runtime_activations=runtime_activations,
    )

    arm_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for i, arm in enumerate(selected_arms, start=2):
        status = (
            "skipped"
            if _arm_skip_reason(arm, runtime_activations=runtime_activations)
            else "running"
        )
        print(
            f"\n[{i}/{total_arms}] {status}: {arm.arm_id} ({arm.confidence})",
            file=sys.stderr,
            flush=True,
        )
        arm_result = _run_scale_arm(
            base_config=base_config,
            arm=arm,
            n_datasets_per_arm=datasets_per_arm,
            seed=seed,
            device=device,
            runtime_activations=runtime_activations,
        )
        arm_results.append(arm_result)
        comparison = _compare_arm_to_baseline(
            arm_result=arm_result,
            baseline_result=baseline_result,
            meaningful_threshold_pct=float(meaningful_threshold_pct),
        )
        comparisons.append(comparison)

        # -- per-arm telemetry --
        arm_id = comparison.get("arm_id", "?")
        cmp_status = comparison.get("status")
        if cmp_status == "evaluated":
            composite = comparison.get("composite_shift_pct")
            meaningful = comparison.get("is_meaningful")
            shifts = comparison.get("metric_shift_pct", {})
            shift_parts = ", ".join(f"{k}={v:+.2f}%" for k, v in sorted(shifts.items()))
            print(
                f"  -> {arm_id}: composite={composite:+.2f}% "
                f"meaningful={meaningful} ({shift_parts})",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"  -> {arm_id}: {cmp_status}",
                file=sys.stderr,
                flush=True,
            )

        if out_dir is not None:
            arm_json_path = out_dir / f"arm_{arm_id}.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            arm_json_path.write_text(
                json.dumps(
                    sanitize_json({"arm_result": arm_result, "comparison": comparison}),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )

    scale_elapsed = time.monotonic() - scale_t0
    print(
        f"\n=== Scale audit complete: {scale_elapsed:.1f}s total ===",
        file=sys.stderr,
        flush=True,
    )

    evaluated = [item for item in comparisons if item.get("status") == "evaluated"]
    meaningful = [item for item in evaluated if bool(item.get("is_meaningful"))]

    return {
        "schema_name": "dagzoo_effective_diversity_scale_impact",
        "schema_version": 1,
        "config": {
            "suite": normalized_suite,
            "arm_set": arm_set,
            "seed": int(seed),
            "num_datasets_per_arm": int(datasets_per_arm),
            "meaningful_threshold_pct": float(meaningful_threshold_pct),
            "device": device,
            "runtime_activation_names": list(runtime_activations),
            "base_config": base_config.to_dict(),
        },
        "baseline": baseline_result,
        "arms": arm_results,
        "comparisons": comparisons,
        "summary": {
            "num_arms_total": int(len(arm_results)),
            "num_arms_evaluated": int(len(evaluated)),
            "num_arms_meaningful": int(len(meaningful)),
            "meaningful_arm_ids": [str(item.get("arm_id")) for item in meaningful],
            "core_metrics": list(_SCALE_CORE_METRICS),
        },
    }


def _config_fingerprint(payload: dict[str, Any]) -> str:
    """Return deterministic SHA1 fingerprint for a JSON payload."""

    normalized = json.dumps(sanitize_json(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def build_effective_diversity_baseline_payload(scale_report: dict[str, Any]) -> dict[str, Any]:
    """Extract compact baseline payload from a scale-impact report."""

    config = scale_report.get("config", {})
    comparisons = scale_report.get("comparisons", [])

    arms: dict[str, dict[str, float]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        if comparison.get("status") != "evaluated":
            continue
        arm_id = str(comparison.get("arm_id"))
        composite = _coerce_float(comparison.get("composite_shift_pct"))
        if composite is None:
            continue
        arms[arm_id] = {"composite_shift_pct": float(composite)}

    return {
        "version": 1,
        "schema_name": "dagzoo_effective_diversity_scale_baseline",
        "suite": config.get("suite"),
        "arm_set": config.get("arm_set"),
        "meaningful_threshold_pct": _coerce_float(config.get("meaningful_threshold_pct")),
        "metrics": ["composite_shift_pct"],
        "config_fingerprint": _config_fingerprint(config if isinstance(config, dict) else {}),
        "arms": arms,
    }


def write_effective_diversity_baseline_payload(
    payload: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write effective-diversity scale baseline payload to disk."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"{json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False)}\n",
        encoding="utf-8",
    )
    return out_path


def load_effective_diversity_baseline_payload(path: str | Path) -> dict[str, Any]:
    """Load effective-diversity baseline JSON payload."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Effective-diversity baseline payload must be a JSON object.")
    return payload


def compare_scale_report_to_baseline(
    scale_report: dict[str, Any],
    baseline_payload: dict[str, Any],
    *,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> dict[str, Any]:
    """Compare scale-impact report against a saved baseline."""

    current_config = scale_report.get("config", {})
    if not isinstance(current_config, dict):
        current_config = {}
    current_config_fingerprint = _config_fingerprint(current_config)
    baseline_config_fingerprint = baseline_payload.get("config_fingerprint")

    compatibility_issues: list[dict[str, Any]] = []
    if baseline_payload.get("schema_name") != "dagzoo_effective_diversity_scale_baseline":
        compatibility_issues.append(
            {
                "code": "baseline_schema_mismatch",
                "expected": "dagzoo_effective_diversity_scale_baseline",
                "actual": baseline_payload.get("schema_name"),
                "severity": "fail",
            }
        )
    if baseline_payload.get("version") != 1:
        compatibility_issues.append(
            {
                "code": "baseline_version_mismatch",
                "expected": 1,
                "actual": baseline_payload.get("version"),
                "severity": "fail",
            }
        )
    if baseline_payload.get("suite") != current_config.get("suite"):
        compatibility_issues.append(
            {
                "code": "suite_mismatch",
                "expected": current_config.get("suite"),
                "actual": baseline_payload.get("suite"),
                "severity": "fail",
            }
        )
    if baseline_payload.get("arm_set") != current_config.get("arm_set"):
        compatibility_issues.append(
            {
                "code": "arm_set_mismatch",
                "expected": current_config.get("arm_set"),
                "actual": baseline_payload.get("arm_set"),
                "severity": "fail",
            }
        )
    if baseline_config_fingerprint != current_config_fingerprint:
        compatibility_issues.append(
            {
                "code": "config_fingerprint_mismatch",
                "expected": current_config_fingerprint,
                "actual": baseline_config_fingerprint,
                "severity": "fail",
            }
        )

    arms_baseline = baseline_payload.get("arms", {})
    if not isinstance(arms_baseline, dict):
        compatibility_issues.append(
            {
                "code": "baseline_arms_not_object",
                "expected": "object",
                "actual": type(arms_baseline).__name__,
                "severity": "fail",
            }
        )
        arms_baseline = {}

    for comparison in scale_report.get("comparisons", []):
        if not isinstance(comparison, dict):
            continue
        if comparison.get("status") != "evaluated":
            continue
        arm_id = str(comparison.get("arm_id"))
        baseline_entry = arms_baseline.get(arm_id)
        if not isinstance(baseline_entry, dict):
            compatibility_issues.append(
                {
                    "code": "missing_baseline_arm",
                    "arm_id": arm_id,
                    "severity": "fail",
                }
            )
            continue
        baseline = _coerce_float(baseline_entry.get("composite_shift_pct"))
        if baseline is None:
            compatibility_issues.append(
                {
                    "code": "invalid_baseline_metric",
                    "arm_id": arm_id,
                    "metric": "composite_shift_pct",
                    "actual": baseline_entry.get("composite_shift_pct"),
                    "severity": "fail",
                }
            )

    if compatibility_issues:
        return {
            "status": "fail",
            "warn_threshold_pct": float(warn_threshold_pct),
            "fail_threshold_pct": float(fail_threshold_pct),
            "issues": compatibility_issues,
            "compatibility_issues": compatibility_issues,
            "delta_issues": [],
            "baseline_config_fingerprint": baseline_config_fingerprint,
            "current_config_fingerprint": current_config_fingerprint,
        }

    delta_issues: list[dict[str, Any]] = []
    for comparison in scale_report.get("comparisons", []):
        if not isinstance(comparison, dict):
            continue
        if comparison.get("status") != "evaluated":
            continue

        arm_id = str(comparison.get("arm_id"))
        current = _coerce_float(comparison.get("composite_shift_pct"))
        baseline_entry = arms_baseline.get(arm_id)
        if not isinstance(baseline_entry, dict):
            continue
        baseline = _coerce_float(baseline_entry.get("composite_shift_pct"))
        if current is None or baseline is None:
            continue

        delta = float(current - baseline)
        if delta < float(warn_threshold_pct):
            continue

        severity = "fail" if delta >= float(fail_threshold_pct) else "warn"
        delta_issues.append(
            {
                "arm_id": arm_id,
                "metric": "composite_shift_pct",
                "current": float(current),
                "baseline": float(baseline),
                "delta_pct": float(delta),
                "severity": severity,
            }
        )

    status = "pass"
    if any(issue["severity"] == "fail" for issue in delta_issues):
        status = "fail"
    elif delta_issues:
        status = "warn"

    delta_issues.sort(key=lambda item: (item["severity"] != "fail", -float(item["delta_pct"])))

    return {
        "status": status,
        "warn_threshold_pct": float(warn_threshold_pct),
        "fail_threshold_pct": float(fail_threshold_pct),
        "issues": delta_issues,
        "compatibility_issues": [],
        "delta_issues": delta_issues,
        "baseline_config_fingerprint": baseline_config_fingerprint,
        "current_config_fingerprint": current_config_fingerprint,
    }


def run_effective_diversity_audit(
    *,
    base_config: GeneratorConfig | None = None,
    phase: str = "both",
    arm_set: str = "high_confidence",
    suite: str = "standard",
    num_datasets_per_arm: int | None = None,
    device: str | None = None,
    seed: int = 2_026_0304,
    n_seeds: int = 8,
    n_rows: int = 2_048,
    n_cols: int = 16,
    out_dim: int = 16,
    nn_degenerate_trials: int = 50_000,
    thresholds: AuditThresholds = AuditThresholds(),
    meaningful_threshold_pct: float = 5.0,
    baseline_payload: dict[str, Any] | None = None,
    warn_threshold_pct: float = 2.5,
    fail_threshold_pct: float = 5.0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run local and/or dataset-scale effective-diversity audits."""

    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"both", "local", "scale"}:
        raise ValueError(f"Unsupported phase: {phase!r}")

    resolved_config = (
        clone_generator_config(base_config, revalidate=False)
        if base_config is not None
        else GeneratorConfig()
    )

    local_report: dict[str, Any] | None = None
    scale_report: dict[str, Any] | None = None

    print(
        f"Effective-diversity audit starting (phase={normalized_phase})",
        file=sys.stderr,
        flush=True,
    )

    if normalized_phase in {"both", "local"}:
        print("\n--- Local overlap phase ---", file=sys.stderr, flush=True)
        local_report = generate_effective_diversity_report(
            seed=seed,
            n_seeds=n_seeds,
            n_rows=n_rows,
            n_cols=n_cols,
            out_dim=out_dim,
            nn_degenerate_trials=nn_degenerate_trials,
            thresholds=thresholds,
        )

    if normalized_phase in {"both", "scale"}:
        print("\n--- Dataset-scale impact phase ---", file=sys.stderr, flush=True)
        scale_report = generate_effective_diversity_scale_report(
            base_config=resolved_config,
            arm_set=arm_set,
            suite=suite,
            num_datasets_per_arm=num_datasets_per_arm,
            seed=seed,
            meaningful_threshold_pct=meaningful_threshold_pct,
            device=device,
            out_dir=out_dir,
        )

    regression: dict[str, Any] | None = None
    if scale_report is not None and baseline_payload is not None:
        regression = compare_scale_report_to_baseline(
            scale_report,
            baseline_payload,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        )

    return {
        "schema_name": "dagzoo_effective_diversity_run",
        "schema_version": 1,
        "phase": normalized_phase,
        "local_report": local_report,
        "scale_report": scale_report,
        "regression": regression,
    }


def format_effective_diversity_scale_markdown(report: dict[str, Any]) -> str:
    """Render dataset-scale impact summary markdown."""

    cfg = report.get("config", {})
    lines = [
        "# Effective Diversity Scale Impact",
        "",
        f"- suite: `{cfg.get('suite')}`",
        f"- arm_set: `{cfg.get('arm_set')}`",
        f"- num_datasets_per_arm: `{cfg.get('num_datasets_per_arm')}`",
        f"- meaningful_threshold_pct: `{cfg.get('meaningful_threshold_pct')}`",
        "",
        "## Arm Comparisons",
    ]

    comparisons = report.get("comparisons", [])
    for item in comparisons:
        arm_id = item.get("arm_id")
        status = item.get("status")
        if status == "evaluated":
            lines.append(
                f"- `{arm_id}`: composite_shift_pct={float(item['composite_shift_pct']):.3f}, "
                f"meaningful={bool(item.get('is_meaningful'))}"
            )
        else:
            lines.append(f"- `{arm_id}`: status={status} ({item.get('skip_reason', 'n/a')})")

    summary = report.get("summary", {})
    lines.extend(
        [
            "",
            "## Summary",
            f"- num_arms_total: `{summary.get('num_arms_total')}`",
            f"- num_arms_evaluated: `{summary.get('num_arms_evaluated')}`",
            f"- num_arms_meaningful: `{summary.get('num_arms_meaningful')}`",
            f"- meaningful_arm_ids: `{summary.get('meaningful_arm_ids')}`",
        ]
    )

    return "\n".join(lines)


def write_effective_diversity_scale_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write dataset-scale JSON + Markdown artifacts and return paths."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "impact_summary.json"
    md_path = out_path / "impact_summary.md"

    json_path.write_text(
        f"{json.dumps(sanitize_json(report), indent=2, sort_keys=True, allow_nan=False)}\n",
        encoding="utf-8",
    )
    md_path.write_text(format_effective_diversity_scale_markdown(report), encoding="utf-8")
    return json_path, md_path


def format_effective_diversity_run_markdown(report: dict[str, Any]) -> str:
    """Render top-level run summary markdown."""

    lines = [
        "# Effective Diversity Run Summary",
        "",
        f"- phase: `{report.get('phase')}`",
    ]

    local_report = report.get("local_report")
    scale_report = report.get("scale_report")
    regression = report.get("regression")

    if isinstance(local_report, dict):
        lines.append(
            "- local report: "
            f"schema={local_report.get('schema_name')} v{local_report.get('schema_version')}"
        )
    if isinstance(scale_report, dict):
        lines.append(
            "- scale report: "
            f"schema={scale_report.get('schema_name')} v{scale_report.get('schema_version')}"
        )

    if isinstance(regression, dict):
        lines.append(f"- regression status: `{regression.get('status')}`")
        lines.append(f"- regression issues: `{len(regression.get('issues', []))}`")

    return "\n".join(lines)


def write_effective_diversity_run_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write run artifacts for executed phases and return path map."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    local_report = report.get("local_report")
    scale_report = report.get("scale_report")

    if isinstance(local_report, dict):
        local_json, local_md = write_effective_diversity_artifacts(local_report, out_dir=out_path)
        paths["equivalence_json"] = local_json
        paths["equivalence_markdown"] = local_md

    if isinstance(scale_report, dict):
        impact_json, impact_md = write_effective_diversity_scale_artifacts(
            scale_report, out_dir=out_path
        )
        paths["impact_json"] = impact_json
        paths["impact_markdown"] = impact_md

    run_json_path = out_path / "run_summary.json"
    run_md_path = out_path / "run_summary.md"
    run_json_path.write_text(
        f"{json.dumps(sanitize_json(report), indent=2, sort_keys=True, allow_nan=False)}\n",
        encoding="utf-8",
    )
    run_md_path.write_text(format_effective_diversity_run_markdown(report), encoding="utf-8")
    paths["run_json"] = run_json_path
    paths["run_markdown"] = run_md_path
    return paths


__all__ = [
    "AblationArm",
    "AuditThresholds",
    "HypothesisClaim",
    "build_effective_diversity_baseline_payload",
    "compare_scale_report_to_baseline",
    "format_effective_diversity_markdown",
    "format_effective_diversity_run_markdown",
    "format_effective_diversity_scale_markdown",
    "generate_effective_diversity_report",
    "generate_effective_diversity_scale_report",
    "load_effective_diversity_baseline_payload",
    "run_effective_diversity_audit",
    "write_effective_diversity_artifacts",
    "write_effective_diversity_baseline_payload",
    "write_effective_diversity_run_artifacts",
    "write_effective_diversity_scale_artifacts",
]
