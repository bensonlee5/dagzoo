"""Tests for deterministic MCAR/MAR/MNAR missingness mask sampling."""

from __future__ import annotations

import pytest
import torch

from dagzoo.config import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    DatasetConfig,
    MissingnessMechanism,
)
from dagzoo.rng import KeyedRng
from dagzoo.sampling import sample_missingness_mask


def _feature_matrix(n_rows: int = 512, n_cols: int = 12) -> torch.Tensor:
    """Create a deterministic, non-trivial feature matrix for sampler tests."""

    grid = torch.linspace(-3.0, 3.0, steps=n_rows, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(1, n_cols + 1, dtype=torch.float32).unsqueeze(0)
    return torch.sin(grid * freqs * 0.7) + torch.cos(grid * (freqs + 0.5) * 0.4)


def _cfg(mechanism: MissingnessMechanism, *, missing_rate: float = 0.35) -> DatasetConfig:
    return DatasetConfig(
        missing_rate=missing_rate,
        missing_mechanism=mechanism,
        missing_mar_observed_fraction=0.5,
        missing_mar_logit_scale=1.5,
        missing_mnar_logit_scale=1.5,
    )


def test_missingness_mask_shape_and_dtype() -> None:
    x = _feature_matrix(64, 7)
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=0.2)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(7), device="cpu")
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_NONE, MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR],
)
def test_missingness_rate_zero_returns_all_false(mechanism: str) -> None:
    x = _feature_matrix(32, 5)
    cfg = _cfg(mechanism, missing_rate=0.0)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(1), device="cpu")
    assert not torch.any(mask)


def test_mcar_rate_one_returns_all_true() -> None:
    x = _feature_matrix(32, 5)
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=1.0)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(1), device="cpu")
    assert torch.all(mask)


def test_mcar_empirical_rate_close_to_target() -> None:
    x = _feature_matrix(2000, 10)
    target_rate = 0.30
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=target_rate)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(11), device="cpu")
    observed_rate = float(mask.float().mean().item())
    assert abs(observed_rate - target_rate) < 0.03


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR],
)
def test_sampler_is_deterministic_for_fixed_seed(mechanism: str) -> None:
    x = _feature_matrix(257, 13)
    cfg = _cfg(mechanism, missing_rate=0.35)
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(123), device="cpu")
    mask_b = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(123), device="cpu")
    assert torch.equal(mask_a, mask_b)


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR],
)
def test_sampler_changes_when_seed_changes(mechanism: str) -> None:
    x = _feature_matrix(257, 13)
    cfg = _cfg(mechanism, missing_rate=0.35)
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(123), device="cpu")
    mask_b = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(124), device="cpu")
    assert not torch.equal(mask_a, mask_b)


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR],
)
def test_mar_and_mnar_empirical_rate_close_to_target(mechanism: str) -> None:
    x = _feature_matrix(2048, 16)
    target_rate = 0.25
    cfg = _cfg(mechanism, missing_rate=target_rate)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(19), device="cpu")
    observed_rate = float(mask.float().mean().item())
    assert abs(observed_rate - target_rate) < 0.05


def test_mar_mask_depends_on_observed_feature_values() -> None:
    x = _feature_matrix(512, 12)
    perturb = torch.linspace(-1.5, 1.5, steps=x.shape[0], dtype=torch.float32).unsqueeze(1)
    scales = torch.arange(1, x.shape[1] + 1, dtype=torch.float32).unsqueeze(0) / float(x.shape[1])
    x_shifted = x + perturb * scales

    cfg = _cfg(MISSINGNESS_MECHANISM_MAR, missing_rate=0.35)
    seed = 909
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    mask_b = sample_missingness_mask(
        x_shifted, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu"
    )
    assert not torch.equal(mask_a, mask_b)


def test_mnar_mask_depends_on_feature_self_values() -> None:
    x = _feature_matrix(512, 12)
    x_mutated = x.clone()
    x_mutated[:, 0] = x_mutated[:, 0] ** 3 + 0.5 * x_mutated[:, 0]

    cfg = _cfg(MISSINGNESS_MECHANISM_MNAR, missing_rate=0.35)
    seed = 1201
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    mask_b = sample_missingness_mask(
        x_mutated, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu"
    )

    assert int(torch.count_nonzero(mask_a ^ mask_b).item()) > 0
    assert int(torch.count_nonzero(mask_a[:, 0] ^ mask_b[:, 0]).item()) > 0


def test_sampler_rejects_non_2d_input() -> None:
    x = torch.ones(10)
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=0.2)
    with pytest.raises(ValueError, match="expects a 2D tensor"):
        sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(1), device="cpu")
