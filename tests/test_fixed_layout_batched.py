"""Tests for internal fixed-layout batched helpers."""

from unittest.mock import patch

import pytest
import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.fixed_layout import _sample_fixed_layout
from dagzoo.core.fixed_layout_batched import (
    FixedLayoutBatchRng,
    _apply_activation_plan,
    _lp_distances_to_centers,
    _nearest_lp_center_indices,
    _sample_random_matrix_from_plan_batch,
    generate_fixed_layout_graph_batch,
    generate_fixed_layout_label_batch,
)
from dagzoo.core.fixed_layout_plan_types import (
    ActivationMatrixPlan,
    FixedActivationPlan,
    GaussianMatrixPlan,
    KernelMatrixPlan,
    ParametricActivationPlan,
    SingularValuesMatrixPlan,
    WeightsMatrixPlan,
    coerce_fixed_layout_execution_plan,
    fixed_layout_execution_plan_payloads,
)
from dagzoo.functions.activations import _fixed_activation


@pytest.mark.parametrize(
    "name",
    ["relu_sq", "softmax", "onehot_argmax", "argsort", "rank"],
)
def test_apply_activation_plan_fixed_variants_match_flat_reference(name: str) -> None:
    x = torch.randn(2, 5, 4, generator=torch.Generator(device="cpu").manual_seed(7))
    rng = FixedLayoutBatchRng(seed=11, batch_size=2, device="cpu")
    out = _apply_activation_plan(
        x,
        rng,
        FixedActivationPlan(name=name),
        with_standardize=False,
    )
    expected = _fixed_activation(x.reshape(-1, x.shape[-1]), name).reshape_as(x)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("kind", ["relu_pow", "signed_pow", "inv_pow"])
def test_apply_activation_plan_parametric_variants_broadcast_across_matrix_count(
    kind: str,
) -> None:
    x = torch.tensor(
        [
            [
                [[-1.5, -0.5, 0.25], [1.0, 2.0, 3.0]],
                [[-2.0, -1.0, 0.5], [0.75, 1.5, 2.5]],
                [[-3.0, -1.5, 0.75], [1.25, 2.5, 4.0]],
            ],
            [
                [[-1.25, -0.25, 0.4], [1.5, 2.5, 3.5]],
                [[-2.5, -0.75, 0.6], [1.0, 1.75, 2.75]],
                [[-3.5, -1.25, 0.8], [0.5, 1.25, 2.25]],
            ],
        ],
        dtype=torch.float32,
    )
    q = torch.tensor(
        [
            [0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0],
        ],
        dtype=torch.float32,
    )
    rng = FixedLayoutBatchRng(seed=17, batch_size=2, device="cpu")
    with patch.object(
        FixedLayoutBatchRng,
        "log_uniform",
        autospec=True,
        return_value=q,
    ) as mocked_log_uniform:
        out = _apply_activation_plan(
            x,
            rng,
            ParametricActivationPlan(kind=kind),
            with_standardize=False,
        )

    mocked_log_uniform.assert_called_once_with(rng, (2, 3), low=0.1, high=10.0)
    q_view = q.unsqueeze(-1).unsqueeze(-1)
    if kind == "relu_pow":
        expected = torch.pow(torch.clamp(x, min=0.0), q_view)
    elif kind == "signed_pow":
        expected = torch.sign(x) * torch.pow(torch.abs(x), q_view)
    else:
        expected = torch.pow(torch.abs(x) + 1e-3, -q_view)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "plan",
    [
        GaussianMatrixPlan(),
        WeightsMatrixPlan(),
        SingularValuesMatrixPlan(),
        KernelMatrixPlan(),
        ActivationMatrixPlan(base_kind="gaussian", activation=FixedActivationPlan(name="relu")),
    ],
)
def test_sample_random_matrix_from_plan_batch_supports_matrix_count(
    plan: object,
) -> None:
    rng = FixedLayoutBatchRng(seed=13, batch_size=3, device="cpu")
    matrices = _sample_random_matrix_from_plan_batch(
        plan,
        out_dim=4,
        in_dim=3,
        rng=rng,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        matrix_count=2,
    )
    assert matrices.shape == (3, 2, 4, 3)
    assert torch.all(torch.isfinite(matrices))


def test_sample_random_matrix_from_plan_batch_supports_parametric_activation_with_matrix_count() -> (
    None
):
    rng = FixedLayoutBatchRng(seed=19, batch_size=2, device="cpu")
    matrices = _sample_random_matrix_from_plan_batch(
        ActivationMatrixPlan(
            base_kind="gaussian",
            activation=ParametricActivationPlan(kind="relu_pow"),
        ),
        out_dim=4,
        in_dim=3,
        rng=rng,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        matrix_count=5,
    )
    assert matrices.shape == (2, 5, 4, 3)
    assert torch.all(torch.isfinite(matrices))


def test_nearest_lp_center_indices_matches_dense_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(23)
    x = torch.randn(2, 3, 7, 4, generator=generator)
    centers = torch.randn(2, 3, 5, 4, generator=generator)
    p = torch.tensor([[0.5, 1.5, 2.0], [3.0, 4.0, 1.25]], dtype=torch.float32)

    out = _nearest_lp_center_indices(x, centers, p=p)
    dense = torch.pow(
        torch.abs(x.unsqueeze(3) - centers.unsqueeze(2)),
        p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
    ).sum(dim=4)
    expected = torch.argmin(dense, dim=3)

    torch.testing.assert_close(out, expected)


def test_lp_distances_to_centers_matches_dense_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(29)
    x = torch.randn(2, 11, 3, generator=generator)
    centers = torch.randn(2, 6, 3, generator=generator)
    p = torch.tensor([1.25, 3.0], dtype=torch.float32)

    out = _lp_distances_to_centers(x, centers, p=p, take_root=True)
    expected = torch.pow(
        torch.pow(
            torch.abs(x.unsqueeze(2) - centers.unsqueeze(1)),
            p.view(-1, 1, 1, 1),
        ).sum(dim=3),
        1.0 / p.view(-1, 1, 1),
    )

    torch.testing.assert_close(out, expected)


def test_generate_fixed_layout_label_batch_matches_graph_batch_targets() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 6
    cfg.dataset.n_test = 4
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 4
    cfg.dataset.n_classes_min = 3
    cfg.dataset.n_classes_max = 3
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 3
    plan = _sample_fixed_layout(cfg, seed=123, device="cpu")
    dataset_seeds = [901, 902]

    x_batch, y_batch, _aux = generate_fixed_layout_graph_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    label_batch, _aux_only = generate_fixed_layout_label_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert x_batch.shape[0] == label_batch.shape[0] == len(dataset_seeds)
    torch.testing.assert_close(label_batch, y_batch)


def test_coerce_fixed_layout_execution_plan_backfills_legacy_ranges_and_groups() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 6
    cfg.dataset.n_test = 4
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 4
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 3
    plan = _sample_fixed_layout(cfg, seed=456, device="cpu")

    legacy_payload = fixed_layout_execution_plan_payloads(plan.execution_plan)
    assert legacy_payload
    first_payload = legacy_payload[0]
    first_payload.pop("converter_groups", None)
    first_payload.pop("execution_contract", None)
    for spec in first_payload["converter_specs"]:
        spec.pop("column_start", None)
        spec.pop("column_end", None)

    coerced = coerce_fixed_layout_execution_plan(legacy_payload)

    assert coerced == plan.execution_plan
