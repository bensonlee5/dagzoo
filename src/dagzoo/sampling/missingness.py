"""Deterministic missingness mask samplers (MCAR/MAR/MNAR)."""

from __future__ import annotations

import math

import torch

from dagzoo.config import (
    DatasetConfig,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    normalize_missing_mechanism,
)
from dagzoo.rng import SeedManager

_MIN_STD = 1e-6
_CALIBRATION_ITERS = 48
_CALIBRATION_BOUND = 30.0


def _standardize_columns(x: torch.Tensor) -> torch.Tensor:
    """Column-wise z-score normalization with safe variance floor."""

    means = x.mean(dim=0, keepdim=True)
    stds = torch.clamp(x.std(dim=0, correction=0, keepdim=True), min=_MIN_STD)
    return (x - means) / stds


def _calibrate_intercept(base_logits: torch.Tensor, target_rate: float) -> float:
    """Find intercept so mean(sigmoid(base_logits + intercept)) ~= target_rate."""

    low = -_CALIBRATION_BOUND
    high = _CALIBRATION_BOUND
    for _ in range(_CALIBRATION_ITERS):
        mid = (low + high) / 2.0
        mean_prob = float(torch.sigmoid(base_logits + mid).mean().item())
        if mean_prob < target_rate:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _calibrated_probabilities(base_logits: torch.Tensor, target_rate: float) -> torch.Tensor:
    """Return per-cell probabilities calibrated to a global target mean rate."""

    if target_rate <= 0.0:
        return torch.zeros_like(base_logits)
    if target_rate >= 1.0:
        return torch.ones_like(base_logits)
    intercept = _calibrate_intercept(base_logits, target_rate)
    return torch.sigmoid(base_logits + intercept)


def _sample_mask_from_probabilities(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw a boolean mask from per-cell Bernoulli probabilities."""

    draws = torch.rand(
        probabilities.shape,
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    return draws < probabilities


def _sample_mcar_mask(
    x: torch.Tensor,
    *,
    missing_rate: float,
    seed_manager: SeedManager,
    device: str,
) -> torch.Tensor:
    """MCAR sampler: each cell is independently missing with fixed probability."""

    probs = torch.full_like(x, fill_value=float(missing_rate), dtype=torch.float32)
    draw_generator = seed_manager.torch_rng("missingness", "mcar", "draws", device=device)
    return _sample_mask_from_probabilities(probs, generator=draw_generator)


def _sample_mar_mask(
    x: torch.Tensor,
    *,
    dataset_cfg: DatasetConfig,
    missing_rate: float,
    seed_manager: SeedManager,
    device: str,
) -> torch.Tensor:
    """MAR sampler: missingness depends on a sampled observed feature subset."""

    x_standardized = _standardize_columns(x)
    n_cols = int(x_standardized.shape[1])
    observed_count = max(
        1,
        min(n_cols, int(math.ceil(float(dataset_cfg.missing_mar_observed_fraction) * n_cols))),
    )

    obs_generator = seed_manager.torch_rng("missingness", "mar", "observed_idx", device=device)
    observed_perm = torch.randperm(n_cols, device=x_standardized.device, generator=obs_generator)
    observed_mask = torch.zeros(n_cols, dtype=torch.bool, device=x_standardized.device)
    observed_mask[observed_perm[:observed_count]] = True

    weight_generator = seed_manager.torch_rng("missingness", "mar", "weights", device=device)
    weights = torch.randn(
        (n_cols, n_cols),
        dtype=x_standardized.dtype,
        device=x_standardized.device,
        generator=weight_generator,
    )
    weights = weights * observed_mask.to(dtype=x_standardized.dtype).unsqueeze(1)
    weights.fill_diagonal_(0.0)

    raw_scores = x_standardized @ weights
    raw_scores = raw_scores / math.sqrt(float(max(1, observed_count)))
    base_logits = raw_scores * float(dataset_cfg.missing_mar_logit_scale)
    probs = _calibrated_probabilities(base_logits, missing_rate)

    draw_generator = seed_manager.torch_rng("missingness", "mar", "draws", device=device)
    return _sample_mask_from_probabilities(probs, generator=draw_generator)


def _sample_mnar_mask(
    x: torch.Tensor,
    *,
    dataset_cfg: DatasetConfig,
    missing_rate: float,
    seed_manager: SeedManager,
    device: str,
) -> torch.Tensor:
    """MNAR sampler: missingness depends on each feature's own standardized value."""

    x_standardized = _standardize_columns(x)
    n_cols = int(x_standardized.shape[1])
    weight_generator = seed_manager.torch_rng("missingness", "mnar", "weights", device=device)
    feature_weights = torch.randn(
        (1, n_cols),
        dtype=x_standardized.dtype,
        device=x_standardized.device,
        generator=weight_generator,
    )
    base_logits = (x_standardized * feature_weights) * float(dataset_cfg.missing_mnar_logit_scale)
    probs = _calibrated_probabilities(base_logits, missing_rate)

    draw_generator = seed_manager.torch_rng("missingness", "mnar", "draws", device=device)
    return _sample_mask_from_probabilities(probs, generator=draw_generator)


def sample_missingness_mask(
    x: torch.Tensor,
    *,
    dataset_cfg: DatasetConfig,
    seed_manager: SeedManager,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample a deterministic missingness mask for one feature matrix.

    The returned tensor is boolean with the same shape as `x`, where `True`
    indicates missing entries to be injected by downstream integration.
    """

    if x.ndim != 2:
        raise ValueError(f"Missingness sampler expects a 2D tensor, got shape {tuple(x.shape)!r}.")

    missing_rate = float(dataset_cfg.missing_rate)
    if missing_rate <= 0.0 or x.numel() == 0:
        return torch.zeros_like(x, dtype=torch.bool)
    if missing_rate >= 1.0:
        return torch.ones_like(x, dtype=torch.bool)

    mechanism = normalize_missing_mechanism(dataset_cfg.missing_mechanism)
    if mechanism == MISSINGNESS_MECHANISM_NONE:
        return torch.zeros_like(x, dtype=torch.bool)

    work = x.to(device=device, dtype=torch.float32)
    work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)

    if mechanism == MISSINGNESS_MECHANISM_MCAR:
        mask = _sample_mcar_mask(
            work, missing_rate=missing_rate, seed_manager=seed_manager, device=device
        )
    elif mechanism == MISSINGNESS_MECHANISM_MAR:
        mask = _sample_mar_mask(
            work,
            dataset_cfg=dataset_cfg,
            missing_rate=missing_rate,
            seed_manager=seed_manager,
            device=device,
        )
    elif mechanism == MISSINGNESS_MECHANISM_MNAR:
        mask = _sample_mnar_mask(
            work,
            dataset_cfg=dataset_cfg,
            missing_rate=missing_rate,
            seed_manager=seed_manager,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported missingness mechanism: {mechanism!r}.")

    return mask.to(device=x.device)
