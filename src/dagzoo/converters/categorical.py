"""Categorical converter implementations."""

from __future__ import annotations

import torch

from dagzoo.core.layout_types import MechanismFamily
from dagzoo.functions._rng_helpers import randint_scalar
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.math_utils import (
    log_uniform as _log_uniform,
    standardize as _standardize,
)


_JOINT_VARIANTS = [
    ("neighbor", "input"),
    ("neighbor", "index_repeat"),
    ("neighbor", "center"),
    ("neighbor", "center_random_fn"),
    ("softmax", "input"),
    ("softmax", "index_repeat"),
    ("softmax", "softmax_points"),
]


def apply_categorical_converter(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    n_categories: int,
    method: str | None = None,
    function_family_mix: dict[MechanismFamily, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Categorical converter in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    device = str(y.device)
    c = max(2, int(n_categories))

    # Sample joint variant
    idx_joint = randint_scalar(0, len(_JOINT_VARIANTS), generator)
    selected_method, variant = _JOINT_VARIANTS[int(idx_joint)]
    if method is not None:
        selected_method = str(method).strip().lower()
    if selected_method not in {"neighbor", "softmax"}:
        raise ValueError(f"Unsupported categorical converter method: {selected_method!r}.")

    centers = None
    if selected_method == "neighbor":
        n_centers = min(c, y.shape[0])
        idx = torch.randperm(y.shape[0], generator=generator, device=y.device)[:n_centers]
        centers = y[idx]
        p = _log_uniform(generator, 0.5, 4.0, device)
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
        x_std = _standardize(logits_in)
        a = _log_uniform(generator, 0.1, 10.0, device)
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
            out = apply_random_function(
                cvec,
                generator,
                out_dim=d,
                function_family_mix=function_family_mix,
            )
        else:
            out = apply_random_function(
                y,
                generator,
                out_dim=d,
                function_family_mix=function_family_mix,
            )
    elif variant == "softmax_points":
        z = torch.randn(c, d, generator=generator, device=device)
        out = z[labels % c]
    else:
        out = y

    if out.shape[1] != d:
        if out.shape[1] > d:
            out = out[:, :d]
        else:
            out = torch.nn.functional.pad(out, (0, d - out.shape[1]))
    out = torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)

    labels = torch.remainder(labels.to(torch.int64), c)
    return out, labels
