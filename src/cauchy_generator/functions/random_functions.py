"""Random function families (Appendix E.8)."""

from __future__ import annotations

import numpy as np
import torch

from cauchy_generator.core.trees import compute_odt_leaf_indices, sample_odt_splits
from cauchy_generator.functions.activations import (
    apply_random_activation,
    apply_random_activation_torch,
)
from cauchy_generator.linalg.random_matrices import (
    sample_random_matrix,
    sample_random_matrix_torch,
)
from cauchy_generator.math_utils import (
    log_uniform as _log_uniform,
    log_uniform_torch as _log_uniform_torch,
    softmax as _softmax,
    standardize as _standardize_base,
    standardize_torch as _standardize_torch_base,
)
from cauchy_generator.sampling.random_weights import (
    sample_random_weights,
    sample_random_weights_torch,
)

Array = np.ndarray


def _standardize(x: Array) -> Array:
    """Standardize columns after clipping non-finite/extreme values."""
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = np.clip(x, -1e6, 1e6)
    return _standardize_base(x).astype(np.float32)


def _standardize_torch(x: torch.Tensor) -> torch.Tensor:
    """Standardize columns in torch after clipping non-finite/extreme values."""
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = torch.clamp(x, -1e6, 1e6)
    return _standardize_torch_base(x)


def _softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax in torch."""
    return torch.softmax(x, dim=1)


def _apply_linear(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Apply a sampled random linear map."""

    m = sample_random_matrix(out_dim, x.shape[1], rng)
    return (x @ m.T).astype(np.float32)


def _apply_linear_torch(x: torch.Tensor, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Apply a sampled random linear map in torch."""
    m = sample_random_matrix_torch(out_dim, x.shape[1], generator, str(x.device))
    return x @ m.t()


def _apply_quadratic(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Apply quadratic forms with dimension capping for efficiency."""

    d_cap = min(x.shape[1], 20)
    if x.shape[1] > d_cap:
        idx = rng.choice(np.arange(x.shape[1]), size=d_cap, replace=False)
        x_sub = x[:, idx]
    else:
        x_sub = x
    x_aug = np.concatenate([x_sub, np.ones((x_sub.shape[0], 1), dtype=np.float32)], axis=1)

    y = np.empty((x_aug.shape[0], out_dim), dtype=np.float32)
    for i in range(out_dim):
        m = sample_random_matrix(x_aug.shape[1], x_aug.shape[1], rng)
        y[:, i] = np.einsum("ni,ij,nj->n", x_aug, m, x_aug)
    return y


def _apply_quadratic_torch(
    x: torch.Tensor, out_dim: int, generator: torch.Generator
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
        m = sample_random_matrix_torch(x_aug.shape[1], x_aug.shape[1], generator, str(x.device))
        y[:, i] = torch.sum((x_aug @ m) * x_aug, dim=1)
    return y


def _apply_nn(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Apply a shallow random feed-forward network."""

    n_layers = int(rng.integers(1, 4))
    hidden = int(np.exp(rng.uniform(np.log(1.0), np.log(127.0))))
    widths = [x.shape[1]]
    for _ in range(max(0, n_layers - 1)):
        widths.append(max(1, hidden))
    widths.append(out_dim)

    y = x
    if rng.random() < 0.5:
        y = apply_random_activation(y, rng)
    for din, dout in zip(widths[:-1], widths[1:], strict=True):
        m = sample_random_matrix(dout, din, rng)
        y = (y @ m.T).astype(np.float32)
        if dout != out_dim:
            y = apply_random_activation(y, rng)
    if rng.random() < 0.5:
        y = apply_random_activation(y, rng)
    return y.astype(np.float32)


def _apply_nn_torch(x: torch.Tensor, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Apply a shallow random NN in torch."""
    n_layers = torch.randint(1, 4, (1,), generator=generator).item()
    hidden = int(_log_uniform_torch(generator, 1.0, 127.0, str(x.device)))
    widths = [x.shape[1]]
    for _ in range(max(0, int(n_layers) - 1)):
        widths.append(max(1, hidden))
    widths.append(out_dim)

    y = x
    if torch.rand(1, generator=generator).item() < 0.5:
        y = apply_random_activation_torch(y, generator)

    for din, dout in zip(widths[:-1], widths[1:], strict=True):
        m = sample_random_matrix_torch(dout, din, generator, str(x.device))
        y = y @ m.t()
        if dout != out_dim:
            y = apply_random_activation_torch(y, generator)

    if torch.rand(1, generator=generator).item() < 0.5:
        y = apply_random_activation_torch(y, generator)
    return y


def _apply_tree(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Apply an ensemble of random oblivious-style trees."""

    n_trees = int(np.exp(rng.uniform(np.log(1.0), np.log(32.0))))
    y = np.zeros((x.shape[0], out_dim), dtype=np.float32)

    std = np.std(x, axis=0)
    total = float(np.sum(std))
    if (not np.isfinite(total)) or total <= 1e-12:
        probs = np.ones_like(std) / max(1, len(std))
    else:
        probs = std / total
        probs = probs / np.clip(np.sum(probs), 1e-12, None)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            probs = np.ones_like(std) / max(1, len(std))

    for _ in range(max(1, n_trees)):
        depth = int(rng.integers(1, 8))
        split_dims = rng.choice(np.arange(x.shape[1]), size=depth, replace=True, p=probs)
        thresholds = np.array(
            [x[int(rng.integers(0, x.shape[0])), d] for d in split_dims],
            dtype=np.float32,
        )

        bits = (x[:, split_dims] > thresholds[None, :]).astype(np.int32)
        powers = (1 << np.arange(depth)).astype(np.int32)
        leaf_idx = np.sum(bits * powers[None, :], axis=1)

        n_leaves = 2**depth
        leaf_vals = rng.normal(size=(n_leaves, out_dim)).astype(np.float32)
        y += leaf_vals[leaf_idx]

    y /= float(max(1, n_trees))
    return y.astype(np.float32)


def _apply_tree_torch(x: torch.Tensor, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Apply an ensemble of random trees in torch."""
    n_trees = int(_log_uniform_torch(generator, 1.0, 32.0, str(x.device)))
    y = torch.zeros(x.shape[0], out_dim, device=x.device)

    std = torch.std(x, dim=0)
    total = torch.sum(std).item()
    if not np.isfinite(total) or total <= 1e-12:
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
        leaf_vals = torch.randn(n_leaves, out_dim, generator=generator, device=x.device)
        y += leaf_vals[leaf_idx]

    y /= float(max(1, n_trees))
    return y


def _apply_discretization(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Map samples to nearest random centers and linearly project."""

    n_centers = int(np.exp(rng.uniform(np.log(2.0), np.log(128.0))))
    n_centers = min(max(2, n_centers), x.shape[0])
    center_idx = rng.choice(np.arange(x.shape[0]), size=n_centers, replace=False)
    centers = x[center_idx]

    p = _log_uniform(rng, 0.5, 4.0)
    dist = np.power(np.abs(x[:, None, :] - centers[None, :, :]), p).sum(axis=2)
    nearest = np.argmin(dist, axis=1)
    y = centers[nearest]
    return _apply_linear(y, out_dim, rng)


def _apply_discretization_torch(
    x: torch.Tensor, out_dim: int, generator: torch.Generator
) -> torch.Tensor:
    """Apply discretization function in torch."""
    n_centers = int(_log_uniform_torch(generator, 2.0, 128.0, str(x.device)))
    n_centers = min(max(2, n_centers), x.shape[0])
    center_idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:n_centers]
    centers = x[center_idx]

    p = _log_uniform_torch(generator, 0.5, 4.0, str(x.device))
    dist = torch.pow(torch.abs(x.unsqueeze(1) - centers.unsqueeze(0)), p).sum(dim=2)
    nearest = torch.argmin(dist, dim=1)
    y = centers[nearest]
    return _apply_linear_torch(y, out_dim, generator)


def _sample_radial_ha(
    n: int,
    rng: np.random.Generator,
    *,
    a: float,
) -> Array:
    """Sample radial magnitudes from the Ha inverse-CDF family."""

    u = rng.uniform(0.0, 1.0, size=n)
    return np.power(1.0 - u, 1.0 / (1.0 - a)) - 1.0


def _sample_radial_ha_torch(
    n: int,
    generator: torch.Generator,
    device: str,
    *,
    a: float,
) -> torch.Tensor:
    """Sample radial magnitudes in torch."""
    u = torch.empty(n, device=device).uniform_(0.0, 1.0, generator=generator)
    return torch.pow(1.0 - u, 1.0 / (1.0 - a)) - 1.0


def _apply_gp(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Approximate random GP features via random Fourier features."""

    din = x.shape[1]
    p = 256
    a = _log_uniform(rng, 2.0, 20.0)

    axis_aligned = rng.random() < 0.5
    if axis_aligned:
        r = _sample_radial_ha(p * din, rng, a=a).reshape(p, din)
        s = rng.choice([-1.0, 1.0], size=(p, din))
        omega = (r * s).astype(np.float32)
        x_proj = x
    else:
        z = rng.normal(size=(p, din)).astype(np.float32)
        z /= np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-6, None)
        r = _sample_radial_ha(p, rng, a=a).astype(np.float32)
        omega = z * r[:, None]

        w = sample_random_weights(din, rng).astype(np.float32)
        alpha = _log_uniform(rng, 0.5, 10.0)
        a_mat = rng.normal(size=(din, din)).astype(np.float32)
        m = alpha * ((w[:, None] * a_mat).astype(np.float32))
        x_proj = x @ m.T

    b = rng.uniform(0.0, 2.0 * np.pi, size=(p,)).astype(np.float32)
    phi = np.cos(x_proj @ omega.T + b[None, :]).astype(np.float32)
    z_out = rng.normal(size=(out_dim, p)).astype(np.float32)
    y = (phi @ z_out.T) / np.sqrt(float(p))
    return y.astype(np.float32)


def _apply_gp_torch(x: torch.Tensor, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Apply GP approximation in torch."""
    din = x.shape[1]
    p = 256
    device = str(x.device)
    a = _log_uniform_torch(generator, 2.0, 20.0, device)

    if torch.rand(1, generator=generator).item() < 0.5:
        r = _sample_radial_ha_torch(p * din, generator, device, a=a).view(p, din)
        s = torch.where(
            torch.empty(p, din, device=device).uniform_(0, 1, generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        omega = r * s
        x_proj = x
    else:
        z = torch.randn(p, din, generator=generator, device=device)
        z /= torch.clamp(torch.norm(z, dim=1, keepdim=True), min=1e-6)
        r = _sample_radial_ha_torch(p, generator, device, a=a)
        omega = z * r.unsqueeze(1)

        w = sample_random_weights_torch(din, generator, device)
        alpha = _log_uniform_torch(generator, 0.5, 10.0, device)
        a_mat = torch.randn(din, din, generator=generator, device=device)
        m = alpha * (w.unsqueeze(1) * a_mat)
        x_proj = x @ m.t()

    b = torch.empty(p, device=device).uniform_(0.0, 2.0 * np.pi, generator=generator)
    phi = torch.cos(x_proj @ omega.t() + b)
    z_out = torch.randn(out_dim, p, generator=generator, device=device)
    return (phi @ z_out.t()) / np.sqrt(float(p))


def _apply_em(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Compute randomized EM-like assignment logits and project."""

    m = int(np.exp(rng.uniform(np.log(2.0), np.log(float(max(16, 2 * out_dim))))))
    m = max(2, m)
    base_idx = rng.choice(np.arange(x.shape[0]), size=m, replace=True)
    centers = x[base_idx] + rng.normal(size=(m, x.shape[1])).astype(np.float32)
    sigma = np.exp(0.1 * rng.normal(size=(m,))).astype(np.float32)
    p = _log_uniform(rng, 1.0, 4.0)
    q = _log_uniform(rng, 1.0, 2.0)

    dist_p = np.power(np.abs(x[:, None, :] - centers[None, :, :]), p).sum(axis=2) ** (1.0 / p)
    logits = -0.5 * np.log(2.0 * np.pi * sigma[None, :] ** 2) - np.power(
        dist_p / np.clip(sigma[None, :], 1e-6, None), q
    )
    probs = _softmax(logits.astype(np.float32))
    return _apply_linear(probs, out_dim, rng)


def _apply_em_torch(x: torch.Tensor, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Apply EM assignment function in torch."""
    m_val = int(_log_uniform_torch(generator, 2.0, float(max(16, 2 * out_dim)), str(x.device)))
    m_val = max(2, m_val)

    base_idx = torch.randint(0, x.shape[0], (m_val,), generator=generator, device=x.device)
    centers = x[base_idx] + torch.randn(m_val, x.shape[1], generator=generator, device=x.device)
    sigma = torch.exp(0.1 * torch.randn(m_val, generator=generator, device=x.device))
    p_val = _log_uniform_torch(generator, 1.0, 4.0, str(x.device))
    q_val = _log_uniform_torch(generator, 1.0, 2.0, str(x.device))

    dist_p = torch.pow(torch.abs(x.unsqueeze(1) - centers.unsqueeze(0)), p_val).sum(dim=2) ** (
        1.0 / p_val
    )
    logits = -0.5 * torch.log(2.0 * np.pi * sigma**2) - torch.pow(
        dist_p / torch.clamp(sigma, min=1e-6), q_val
    )
    probs = _softmax_torch(logits)
    return _apply_linear_torch(probs, out_dim, generator)


def _apply_product(x: Array, out_dim: int, rng: np.random.Generator) -> Array:
    """Multiply outputs of two independently sampled function families."""

    allowed = ["tree", "discretization", "gp", "linear", "quadratic"]
    f_type = str(rng.choice(allowed))
    g_type = str(rng.choice(allowed))
    fx = apply_random_function(x, rng, out_dim=out_dim, function_type=f_type)
    gx = apply_random_function(x, rng, out_dim=out_dim, function_type=g_type)
    return (fx * gx).astype(np.float32)


def _apply_product_torch(x: torch.Tensor, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Apply product function in torch."""
    allowed = ["tree", "discretization", "gp", "linear", "quadratic"]
    idx_f = torch.randint(0, len(allowed), (1,), generator=generator).item()
    idx_g = torch.randint(0, len(allowed), (1,), generator=generator).item()

    fx = apply_random_function_torch(
        x, generator, out_dim=out_dim, function_type=allowed[int(idx_f)]
    )
    gx = apply_random_function_torch(
        x, generator, out_dim=out_dim, function_type=allowed[int(idx_g)]
    )
    return fx * gx


def apply_random_function_torch(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    out_dim: int | None = None,
    function_type: str | None = None,
) -> torch.Tensor:
    """Apply one sampled random function family to `x` in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    y = _standardize_torch(y)

    dout = out_dim if out_dim is not None else y.shape[1]

    if function_type is None:
        families = [
            "nn",
            "tree",
            "discretization",
            "gp",
            "linear",
            "quadratic",
            "em",
            "product",
        ]
        idx = torch.randint(0, len(families), (1,), generator=generator).item()
        function_type = families[int(idx)]

    if function_type == "nn":
        return _apply_nn_torch(y, dout, generator)
    if function_type == "tree":
        return _apply_tree_torch(y, dout, generator)
    if function_type == "discretization":
        return _apply_discretization_torch(y, dout, generator)
    if function_type == "gp":
        return _apply_gp_torch(y, dout, generator)
    if function_type == "linear":
        return _apply_linear_torch(y, dout, generator)
    if function_type == "quadratic":
        return _apply_quadratic_torch(y, dout, generator)
    if function_type == "em":
        return _apply_em_torch(y, dout, generator)
    if function_type == "product":
        return _apply_product_torch(y, dout, generator)

    raise ValueError(f"Unknown random function family: {function_type}")


def apply_random_function(
    x: Array,
    rng: np.random.Generator,
    *,
    out_dim: int | None = None,
    function_type: str | None = None,
) -> Array:
    """Apply one sampled random function family to `x`."""

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = _standardize(arr)

    dout = int(out_dim if out_dim is not None else arr.shape[1])
    family = function_type or str(
        rng.choice(
            [
                "nn",
                "tree",
                "discretization",
                "gp",
                "linear",
                "quadratic",
                "em",
                "product",
            ]
        )
    )

    if family == "nn":
        y = _apply_nn(arr, dout, rng)
    elif family == "tree":
        y = _apply_tree(arr, dout, rng)
    elif family == "discretization":
        y = _apply_discretization(arr, dout, rng)
    elif family == "gp":
        y = _apply_gp(arr, dout, rng)
    elif family == "linear":
        y = _apply_linear(arr, dout, rng)
    elif family == "quadratic":
        y = _apply_quadratic(arr, dout, rng)
    elif family == "em":
        y = _apply_em(arr, dout, rng)
    elif family == "product":
        y = _apply_product(arr, dout, rng)
    else:
        raise ValueError(f"Unknown random function family: {family}")

    return y.astype(np.float32)
