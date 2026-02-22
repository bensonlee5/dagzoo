"""Random activations (Appendix E.9)."""

from __future__ import annotations

import numpy as np
import torch

from cauchy_generator.math_utils import (
    log_uniform as _log_uniform,
    log_uniform_torch as _log_uniform_torch,
    standardize as _standardize,
    standardize_torch as _standardize_torch,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax with max-shift stabilization."""

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)


def _fixed_activation_torch(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply one fixed activation variant by name in torch."""
    if name == "tanh":
        return torch.tanh(x)
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    if name == "elu":
        return torch.nn.functional.elu(x)
    if name == "identity":
        return x
    if name == "silu":
        return torch.nn.functional.silu(x)
    if name == "relu":
        return torch.relu(x)
    if name == "softplus":
        return torch.nn.functional.softplus(x)
    if name == "sign":
        return torch.sign(x)
    if name == "gauss":
        return torch.exp(-(x**2))
    if name == "exp":
        return torch.exp(torch.clamp(x, max=20.0))
    if name == "sin":
        return torch.sin(x)
    if name == "square":
        return x * x
    if name == "abs":
        return torch.abs(x)
    if name == "softmax":
        return torch.softmax(x, dim=1)
    if name == "logsigmoid":
        return torch.nn.functional.logsigmoid(x)
    if name == "logabs":
        return torch.log(torch.clamp(torch.abs(x), min=1e-6))
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "round":
        return torch.round(x)
    if name == "mod1":
        return torch.remainder(x, 1.0)
    if name == "selu":
        return torch.nn.functional.selu(x)
    if name == "relu6":
        return torch.nn.functional.relu6(x)
    if name == "hardtanh":
        return torch.nn.functional.hardtanh(x)
    if name == "heaviside":
        return torch.heaviside(x, values=torch.tensor(0.0, device=x.device))
    if name == "indicator_01":
        return ((x >= 0) & (x <= 1)).to(x.dtype)
    if name == "onehot_argmax":
        indices = torch.argmax(x, dim=1)
        return torch.nn.functional.one_hot(indices, num_classes=x.shape[1]).to(x.dtype)
    if name == "argsort":
        return torch.argsort(x, dim=1).to(x.dtype)
    if name == "rank":
        return (torch.argsort(torch.argsort(x, dim=1), dim=1) + 1).to(x.dtype)
    return x


def _param_activation_torch(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply one randomly parameterized activation family in torch."""
    choices = ["relu_pow", "signed_pow", "inv_pow", "poly"]
    idx = torch.randint(0, len(choices), (1,), generator=generator).item()
    choice = choices[int(idx)]

    if choice == "relu_pow":
        q = _log_uniform_torch(generator, 0.1, 10.0, str(x.device))
        return torch.pow(torch.clamp(x, min=0.0), q)
    if choice == "signed_pow":
        q = _log_uniform_torch(generator, 0.1, 10.0, str(x.device))
        return torch.sign(x) * torch.pow(torch.abs(x), q)
    if choice == "inv_pow":
        q = _log_uniform_torch(generator, 0.1, 10.0, str(x.device))
        return torch.pow(torch.abs(x) + 1e-3, -q)

    m = torch.randint(2, 6, (1,), generator=generator).item()
    return torch.pow(x, float(m))


def apply_random_activation_torch(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    with_standardize: bool = True,
    parametric_prob: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Apply one random activation from Appendix E.9-style families in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    device = str(y.device)

    if with_standardize:
        y = _standardize_torch(y)
        a = _log_uniform_torch(generator, 1.0, 10.0, device)
        row_idx = torch.randint(0, y.shape[0], (1,), generator=generator).item()
        b = y[int(row_idx) : int(row_idx) + 1]
        y = a * (y - b)

    if torch.rand(1, generator=generator).item() < parametric_prob:
        y = _param_activation_torch(y, generator)
    else:
        fixed = [
            "tanh",
            "leaky_relu",
            "elu",
            "identity",
            "silu",
            "relu",
            "softplus",
            "sign",
            "gauss",
            "exp",
            "sin",
            "square",
            "abs",
            "softmax",
            "logsigmoid",
            "logabs",
            "sigmoid",
            "round",
            "mod1",
            "selu",
            "relu6",
            "hardtanh",
            "heaviside",
            "indicator_01",
            "onehot_argmax",
            "argsort",
            "rank",
        ]
        idx = torch.randint(0, len(fixed), (1,), generator=generator).item()
        y = _fixed_activation_torch(y, fixed[int(idx)])

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)

    if with_standardize:
        y = _standardize_torch(y)
    return y.to(torch.float32)


def _fixed_activation(x: np.ndarray, name: str) -> np.ndarray:
    """Apply one fixed activation variant by name."""

    if name == "tanh":
        return np.tanh(x)
    if name == "leaky_relu":
        return np.where(x >= 0.0, x, 0.01 * x)
    if name == "elu":
        return np.where(x >= 0.0, x, np.exp(x) - 1.0)
    if name == "identity":
        return x
    if name == "silu":
        z = np.clip(x, -50.0, 50.0)
        return z / (1.0 + np.exp(-z))
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "softplus":
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    if name == "sign":
        return np.sign(x)
    if name == "gauss":
        return np.exp(-(x**2))
    if name == "exp":
        return np.exp(np.clip(x, -20.0, 20.0))
    if name == "sin":
        return np.sin(x)
    if name == "square":
        return x * x
    if name == "abs":
        return np.abs(x)
    if name == "softmax":
        return _softmax(x)
    if name == "logsigmoid":
        z = np.clip(x, -50.0, 50.0)
        return -np.log1p(np.exp(-z))
    if name == "logabs":
        return np.log(np.maximum(np.abs(x), 1e-6))
    if name == "sigmoid":
        z = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))
    if name == "round":
        return np.round(x)
    if name == "mod1":
        return np.mod(x, 1.0)
    if name == "selu":
        _alpha = 1.6732632
        _lambda = 1.0507010
        return _lambda * np.where(x >= 0, x, _alpha * (np.exp(np.clip(x, -20.0, 20.0)) - 1.0))
    if name == "relu6":
        return np.clip(x, 0.0, 6.0)
    if name == "hardtanh":
        return np.clip(x, -1.0, 1.0)
    if name == "heaviside":
        return np.where(x > 0, 1.0, 0.0)
    if name == "indicator_01":
        return np.where((x >= 0) & (x <= 1), 1.0, 0.0)
    if name == "onehot_argmax":
        out = np.zeros_like(x)
        out[np.arange(x.shape[0]), np.argmax(x, axis=1)] = 1.0
        return out
    if name == "argsort":
        return np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float32)
    if name == "rank":
        return (np.argsort(np.argsort(x, axis=1), axis=1) + 1).astype(np.float32)
    return x


def _param_activation(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one randomly parameterized activation family."""

    choice = rng.choice(["relu_pow", "signed_pow", "inv_pow", "poly"])
    if choice == "relu_pow":
        q = _log_uniform(rng, 0.1, 10.0)
        return np.power(np.maximum(x, 0.0), q)
    if choice == "signed_pow":
        q = _log_uniform(rng, 0.1, 10.0)
        return np.sign(x) * np.power(np.abs(x), q)
    if choice == "inv_pow":
        q = _log_uniform(rng, 0.1, 10.0)
        return np.power(np.abs(x) + 1e-3, -q)
    m = int(rng.integers(2, 6))
    return np.power(x, m)


def apply_random_activation(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    with_standardize: bool = True,
    parametric_prob: float = 1.0 / 3.0,
) -> np.ndarray:
    """Apply one random activation from Appendix E.9-style families."""

    y = np.asarray(x, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]

    if with_standardize:
        y = _standardize(y)
        a = _log_uniform(rng, 1.0, 10.0)
        row_idx = int(rng.integers(0, y.shape[0]))
        b = y[row_idx : row_idx + 1]
        y = a * (y - b)

    if rng.random() < parametric_prob:
        y = _param_activation(y, rng)
    else:
        fixed = [
            "tanh",
            "leaky_relu",
            "elu",
            "identity",
            "silu",
            "relu",
            "softplus",
            "sign",
            "gauss",
            "exp",
            "sin",
            "square",
            "abs",
            "softmax",
            "logsigmoid",
            "logabs",
            "sigmoid",
            "round",
            "mod1",
            "selu",
            "relu6",
            "hardtanh",
            "heaviside",
            "indicator_01",
            "onehot_argmax",
            "argsort",
            "rank",
        ]
        y = _fixed_activation(y, str(rng.choice(fixed)))

    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.clip(y, -1e6, 1e6)

    if with_standardize:
        y = _standardize(y)
    return y.astype(np.float32)
