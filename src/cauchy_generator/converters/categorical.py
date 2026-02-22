"""Categorical converter implementations (Appendix E.6)."""

from __future__ import annotations

import numpy as np
import torch

from cauchy_generator.functions.random_functions import (
    apply_random_function,
    apply_random_function_torch,
)
from cauchy_generator.math_utils import (
    log_uniform as _log_uniform,
    log_uniform_torch as _log_uniform_torch,
    softmax as _softmax,
    standardize as _standardize,
    standardize_torch as _standardize_torch,
)


def _sample_categories_from_probs(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample categorical indices from row-wise probability vectors."""

    cdf = np.cumsum(probs, axis=1)
    u = rng.random(size=(probs.shape[0], 1))
    return np.sum(u > cdf, axis=1).astype(np.int64)


_JOINT_VARIANTS = [
    ("neighbor", "input"),
    ("neighbor", "index_repeat"),
    ("neighbor", "center"),
    ("neighbor", "center_random_fn"),
    ("softmax", "input"),
    ("softmax", "index_repeat"),
    ("softmax", "softmax_points"),
]


def apply_categorical_converter_torch(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    n_categories: int,
    method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Categorical converter in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    device = str(y.device)
    c = max(2, int(n_categories))

    # Sample joint variant
    idx_joint = torch.randint(0, len(_JOINT_VARIANTS), (1,), generator=generator).item()
    selected_method, variant = _JOINT_VARIANTS[int(idx_joint)]
    if method is not None:
        selected_method = method

    centers = None
    if selected_method == "neighbor":
        n_centers = min(c, y.shape[0])
        idx = torch.randperm(y.shape[0], generator=generator, device=y.device)[:n_centers]
        centers = y[idx]
        p = _log_uniform_torch(generator, 0.5, 4.0, device)
        # Pairwise distance (N, 1, D) - (1, M, D)
        dist = torch.pow(torch.abs(y.unsqueeze(1) - centers.unsqueeze(0)), p).sum(dim=2)
        labels = torch.argmin(dist, dim=1)
        if n_centers < c:
            labels = labels % c
    else:
        if y.shape[1] != c:
            proj = torch.randn(y.shape[1], c, generator=generator, device=y.device)
            logits_in = y @ proj
        else:
            logits_in = y
        x_std = _standardize_torch(logits_in)
        a = _log_uniform_torch(generator, 0.1, 10.0, device)
        w = torch.empty(c, device=device).uniform_(0, 1, generator=generator)
        b = torch.log(w + 1e-4)
        logits = a * x_std + b.unsqueeze(0)
        probs = torch.softmax(logits, dim=1)
        labels = torch.multinomial(probs, 1, generator=generator).squeeze(1)

    d = y.shape[1]
    if variant == "input":
        out = y
    elif variant == "index_repeat":
        out = labels.unsqueeze(1).repeat(1, d).to(torch.float32)
    elif variant == "center":
        if centers is not None:
            out = centers[labels % centers.shape[0]]
            if out.shape[1] != d:
                if out.shape[1] > d:
                    out = out[:, :d]
                else:
                    out = torch.nn.functional.pad(out, (0, d - out.shape[1]))
        else:
            out = y
    elif variant == "center_random_fn":
        if centers is not None:
            cvec = centers[labels % centers.shape[0]]
            out = apply_random_function_torch(cvec, generator, out_dim=d)
        else:
            out = apply_random_function_torch(y, generator, out_dim=d)
    elif variant == "softmax_points":
        z = torch.randn(c, d, generator=generator, device=device)
        out = z[labels % c]
    else:
        out = y

    return out, labels


def apply_categorical_converter(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    n_categories: int,
    method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Categorical converter returning (x', class_index).

    Per E.6 the 7 joint (method, output-variant) combinations are sampled
    uniformly rather than selecting method and variant independently.
    """

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    c = max(2, int(n_categories))

    # Joint (method, variant) sampling per E.6
    joint_idx = int(rng.integers(0, len(_JOINT_VARIANTS)))
    selected_method, variant = _JOINT_VARIANTS[joint_idx]
    if method is not None:
        selected_method = method

    centers = None
    if selected_method == "neighbor":
        n_centers = min(c, arr.shape[0])
        idx = rng.choice(np.arange(arr.shape[0]), size=n_centers, replace=False)
        centers = arr[idx]
        p = _log_uniform(rng, 0.5, 4.0)
        dist = np.power(np.abs(arr[:, None, :] - centers[None, :, :]), p).sum(axis=2)
        labels = np.argmin(dist, axis=1).astype(np.int64)
        if n_centers < c:
            labels = labels % c
    else:
        if arr.shape[1] != c:
            proj = rng.normal(size=(arr.shape[1], c)).astype(np.float32)
            logits_in = arr @ proj
        else:
            logits_in = arr
        x_std = _standardize(logits_in)
        a = _log_uniform(rng, 0.1, 10.0)
        w = rng.random(size=(c,)).astype(np.float32)
        b = np.log(w + 1e-4)
        logits = a * x_std + b[None, :]
        probs = _softmax(logits)
        labels = _sample_categories_from_probs(probs, rng)

    d = arr.shape[1]
    if variant == "input":
        out = arr
    elif variant == "index_repeat":
        out = np.repeat(labels[:, None].astype(np.float32), d, axis=1)
    elif variant == "center":
        if centers is not None:
            out = centers[labels % centers.shape[0]]
            if out.shape[1] != d:
                out = (
                    out[:, :d] if out.shape[1] > d else np.pad(out, ((0, 0), (0, d - out.shape[1])))
                )
        else:
            out = arr
    elif variant == "center_random_fn":
        if centers is not None:
            cvec = centers[labels % centers.shape[0]]
            out = apply_random_function(cvec, rng, out_dim=d)
        else:
            out = apply_random_function(arr, rng, out_dim=d)
    elif variant == "softmax_points":
        z = rng.normal(size=(c, d)).astype(np.float32)
        out = z[labels % c]
    else:
        out = arr

    return out.astype(np.float32), labels.astype(np.int64)
