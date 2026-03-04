"""Random function families."""

from __future__ import annotations

import math

import torch

from dagzoo.core.layout_types import MechanismFamily
from dagzoo.core.shift import MECHANISM_FAMILY_ORDER, mechanism_family_probabilities
from dagzoo.core.trees import compute_odt_leaf_indices, sample_odt_splits
from dagzoo.functions.activations import apply_random_activation
from dagzoo.linalg.random_matrices import sample_random_matrix
from dagzoo.math_utils import (
    log_uniform as _log_uniform,
    standardize as _standardize_base,
)
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec
from dagzoo.sampling.random_weights import sample_random_weights


def _standardize(x: torch.Tensor) -> torch.Tensor:
    """Standardize columns in torch after clipping non-finite/extreme values."""
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = torch.clamp(x, -1e6, 1e6)
    return _standardize_base(x)


def _apply_linear(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply a sampled random linear map in torch."""
    m = sample_random_matrix(
        out_dim,
        x.shape[1],
        generator,
        str(x.device),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return x @ m.t()


def _apply_quadratic(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply quadratic forms in torch."""
    d_cap = min(x.shape[1], 20)
    if x.shape[1] > d_cap:
        idx = torch.randperm(x.shape[1], generator=generator, device=x.device)[:d_cap]
        x_sub = x[:, idx]
    else:
        x_sub = x
    x_aug = torch.cat([x_sub, torch.ones(x_sub.shape[0], 1, device=x.device)], dim=1)

    y = torch.empty(x_aug.shape[0], out_dim, device=x.device)
    for i in range(out_dim):
        m = sample_random_matrix(
            x_aug.shape[1],
            x_aug.shape[1],
            generator,
            str(x.device),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        y[:, i] = torch.sum((x_aug @ m) * x_aug, dim=1)
    return y


def _apply_nn(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply a shallow random NN in torch."""
    n_layers = torch.randint(1, 4, (1,), generator=generator).item()
    hidden = int(_log_uniform(generator, 1.0, 127.0, str(x.device)))
    widths = [x.shape[1]]
    for _ in range(max(0, int(n_layers) - 1)):
        widths.append(max(1, hidden))
    widths.append(out_dim)

    y = x
    if torch.rand(1, generator=generator).item() < 0.5:
        y = apply_random_activation(y, generator)

    for din, dout in zip(widths[:-1], widths[1:], strict=True):
        m = sample_random_matrix(
            dout,
            din,
            generator,
            str(x.device),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        y = y @ m.t()
        if dout != out_dim:
            y = apply_random_activation(y, generator)

    if torch.rand(1, generator=generator).item() < 0.5:
        y = apply_random_activation(y, generator)
    return y


def _apply_tree(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply an ensemble of random trees in torch."""
    n_trees = int(_log_uniform(generator, 1.0, 32.0, str(x.device)))
    y = torch.zeros(x.shape[0], out_dim, device=x.device)

    std = torch.std(x, dim=0)
    total = torch.sum(std).item()
    if not math.isfinite(total) or total <= 1e-12:
        probs = torch.ones_like(std) / max(1, len(std))
    else:
        probs = torch.clamp(std, min=0.0)
        probs /= torch.clamp(torch.sum(probs), min=1e-12)
        if not torch.all(torch.isfinite(probs)) or torch.any(probs < 0):
            probs = torch.ones_like(std) / max(1, len(std))

    for _ in range(max(1, n_trees)):
        depth = int(torch.randint(1, 8, (1,), generator=generator).item())
        split_dims, thresholds = sample_odt_splits(x, depth, generator, feature_probs=probs)
        leaf_idx = compute_odt_leaf_indices(x, split_dims, thresholds)

        n_leaves = 2**depth
        leaf_vals = sample_noise_from_spec(
            (n_leaves, out_dim),
            generator=generator,
            device=str(x.device),
            noise_spec=noise_spec,
        )
        y += leaf_vals[leaf_idx]

    y /= float(max(1, n_trees))
    return y


def _apply_discretization(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply discretization function in torch."""
    n_centers = int(_log_uniform(generator, 2.0, 128.0, str(x.device)))
    n_centers = min(max(2, n_centers), x.shape[0])
    center_idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:n_centers]
    centers = x[center_idx]

    p = _log_uniform(generator, 0.5, 4.0, str(x.device))
    dist = torch.pow(torch.abs(x.unsqueeze(1) - centers.unsqueeze(0)), p).sum(dim=2)
    nearest = torch.argmin(dist, dim=1)
    y = centers[nearest]
    return _apply_linear(
        y,
        out_dim,
        generator,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _sample_radial_ha(
    n: int,
    generator: torch.Generator,
    device: str,
    *,
    a: float,
) -> torch.Tensor:
    """Sample radial magnitudes in torch."""
    u = torch.empty(n, device=device).uniform_(0.0, 1.0, generator=generator)
    return torch.pow(1.0 - u, 1.0 / (1.0 - a)) - 1.0


def _apply_gp(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply GP approximation in torch."""
    din = x.shape[1]
    p = 256
    device = str(x.device)
    a = _log_uniform(generator, 2.0, 20.0, device)

    if torch.rand(1, generator=generator).item() < 0.5:
        r = _sample_radial_ha(p * din, generator, device, a=a).view(p, din)
        s = torch.where(
            torch.empty(p, din, device=device).uniform_(0, 1, generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        omega = r * s
        x_proj = x
    else:
        z = sample_noise_from_spec(
            (p, din),
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        )
        z /= torch.clamp(torch.norm(z, dim=1, keepdim=True), min=1e-6)
        r = _sample_radial_ha(p, generator, device, a=a)
        omega = z * r.unsqueeze(1)

        w = sample_random_weights(
            din,
            generator,
            device,
            sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        alpha = _log_uniform(generator, 0.5, 10.0, device)
        a_mat = sample_noise_from_spec(
            (din, din),
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        )
        m = alpha * (w.unsqueeze(1) * a_mat)
        x_proj = x @ m.t()

    b = torch.empty(p, device=device).uniform_(0.0, 2.0 * math.pi, generator=generator)
    phi = torch.cos(x_proj @ omega.t() + b)
    z_out = sample_noise_from_spec(
        (out_dim, p),
        generator=generator,
        device=device,
        noise_spec=noise_spec,
    )
    return (phi @ z_out.t()) / math.sqrt(float(p))


def _apply_em(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply EM assignment function in torch."""
    m_val = int(_log_uniform(generator, 2.0, float(max(16, 2 * out_dim)), str(x.device)))
    m_val = max(2, m_val)

    base_idx = torch.randint(0, x.shape[0], (m_val,), generator=generator, device=x.device)
    centers = x[base_idx] + sample_noise_from_spec(
        (m_val, x.shape[1]),
        generator=generator,
        device=str(x.device),
        noise_spec=noise_spec,
    )
    sigma = torch.exp(
        sample_noise_from_spec(
            (m_val,),
            generator=generator,
            device=str(x.device),
            noise_spec=noise_spec,
            scale_multiplier=0.1,
        )
    )
    p_val = _log_uniform(generator, 1.0, 4.0, str(x.device))
    q_val = _log_uniform(generator, 1.0, 2.0, str(x.device))

    dist_p = torch.pow(torch.abs(x.unsqueeze(1) - centers.unsqueeze(0)), p_val).sum(dim=2) ** (
        1.0 / p_val
    )
    logits = -0.5 * torch.log(2.0 * math.pi * sigma**2) - torch.pow(
        dist_p / torch.clamp(sigma, min=1e-6), q_val
    )
    probs = torch.softmax(logits, dim=1)
    return _apply_linear(
        probs,
        out_dim,
        generator,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _apply_product(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
    *,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply product function in torch."""
    allowed: tuple[MechanismFamily, ...] = (
        "tree",
        "discretization",
        "gp",
        "linear",
        "quadratic",
    )
    eligible = list(allowed)
    if function_family_mix is not None:
        eligible = [
            family for family in allowed if float(function_family_mix.get(family, 0.0)) > 0.0
        ]
    if not eligible:
        raise ValueError(
            "mechanism.function_family_mix enables 'product' but disables all product component "
            "families (tree, discretization, gp, linear, quadratic)."
        )
    idx_f = torch.randint(0, len(eligible), (1,), generator=generator).item()
    idx_g = torch.randint(0, len(eligible), (1,), generator=generator).item()

    fx = apply_random_function(
        x,
        generator,
        out_dim=out_dim,
        function_type=eligible[int(idx_f)],
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    gx = apply_random_function(
        x,
        generator,
        out_dim=out_dim,
        function_type=eligible[int(idx_g)],
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return fx * gx


def _sample_function_family(
    generator: torch.Generator,
    *,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None = None,
) -> MechanismFamily:
    """Sample one function family with optional logit tilt."""

    if mechanism_logit_tilt <= 0.0 and function_family_mix is None:
        idx = torch.randint(0, len(MECHANISM_FAMILY_ORDER), (1,), generator=generator).item()
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
    draw = float(torch.rand(1, generator=generator).item())
    cumulative = 0.0
    for family in positive_families:
        cumulative += float(probs_by_family[family])
        if draw <= cumulative:
            return family
    return positive_families[-1]


def apply_random_function(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    out_dim: int | None = None,
    function_type: MechanismFamily | None = None,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply one sampled random function family to `x` in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    y = _standardize(y)

    dout = out_dim if out_dim is not None else y.shape[1]

    if function_type is None:
        function_type = _sample_function_family(
            generator,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
    elif function_family_mix is not None and function_type not in function_family_mix:
        raise ValueError(
            f"Mechanism family '{function_type}' is not enabled by mechanism.function_family_mix."
        )

    if function_type == "nn":
        return _apply_nn(
            y,
            dout,
            generator,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if function_type == "tree":
        return _apply_tree(y, dout, generator, noise_spec=noise_spec)
    if function_type == "discretization":
        return _apply_discretization(
            y,
            dout,
            generator,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if function_type == "gp":
        return _apply_gp(
            y,
            dout,
            generator,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if function_type == "linear":
        return _apply_linear(
            y,
            dout,
            generator,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if function_type == "quadratic":
        return _apply_quadratic(
            y,
            dout,
            generator,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if function_type == "em":
        return _apply_em(
            y,
            dout,
            generator,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if function_type == "product":
        return _apply_product(
            y,
            dout,
            generator,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    raise ValueError(f"Unknown random function family: {function_type}")
