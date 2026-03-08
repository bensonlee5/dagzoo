"""Tests for internal fixed-layout batched helpers."""

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
