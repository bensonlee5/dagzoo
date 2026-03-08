"""Tests for internal fixed-layout batched helpers."""

from unittest.mock import patch

import pytest
import torch

from dagzoo.core.fixed_layout_batched import (
    FixedLayoutBatchRng,
    _apply_activation_plan,
    _sample_random_matrix_from_plan_batch,
)
from dagzoo.functions.activations import _fixed_activation


@pytest.mark.parametrize(
    "name",
    ["relu_sq", "softmax", "onehot_argmax", "argsort", "rank"],
)
def test_apply_activation_plan_fixed_variants_match_flat_reference(name: str) -> None:
    x = torch.randn(2, 5, 4, generator=torch.Generator(device="cpu").manual_seed(7))
    rng = FixedLayoutBatchRng(seed=11, batch_size=2, device="cpu")
    out = _apply_activation_plan(x, rng, {"mode": "fixed", "name": name}, with_standardize=False)
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
            {"mode": "parametric", "kind": kind},
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
        {"kind": "gaussian"},
        {"kind": "weights"},
        {"kind": "singular_values"},
        {"kind": "kernel"},
        {
            "kind": "activation",
            "base_kind": "gaussian",
            "activation": {"mode": "fixed", "name": "relu"},
        },
    ],
)
def test_sample_random_matrix_from_plan_batch_supports_matrix_count(
    plan: dict[str, object],
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
        {
            "kind": "activation",
            "base_kind": "gaussian",
            "activation": {"mode": "parametric", "kind": "relu_pow"},
        },
        out_dim=4,
        in_dim=3,
        rng=rng,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        matrix_count=5,
    )
    assert matrices.shape == (2, 5, 4, 3)
    assert torch.all(torch.isfinite(matrices))
