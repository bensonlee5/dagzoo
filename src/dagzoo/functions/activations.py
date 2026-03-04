"""Random activations."""

from __future__ import annotations

import torch

from dagzoo.functions._rng_helpers import rand_scalar, randint_scalar
from dagzoo.math_utils import (
    log_uniform as _log_uniform,
    standardize as _standardize,
)

_FIXED_ACTIVATION_NAMES: tuple[str, ...] = (
    "tanh",
    "leaky_relu",
    "elu",
    "identity",
    "silu",
    "swiglu",
    "relu",
    "relu_sq",
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
    "indicator_01",
    "onehot_argmax",
    "argsort",
    "rank",
)


def fixed_activation_names() -> tuple[str, ...]:
    """Return the canonical fixed-activation sampling order."""

    return _FIXED_ACTIVATION_NAMES


def _fixed_activation(x: torch.Tensor, name: str) -> torch.Tensor:
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
    if name == "swiglu":
        return x * torch.nn.functional.silu(x)
    if name == "relu":
        return torch.relu(x)
    if name == "relu_sq":
        return torch.square(torch.relu(x))
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


def _param_activation(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply one randomly parameterized activation family in torch."""
    choices = ["relu_pow", "signed_pow", "inv_pow", "poly"]
    idx = randint_scalar(0, len(choices), generator)
    choice = choices[int(idx)]

    if choice == "relu_pow":
        q = _log_uniform(generator, 0.1, 10.0, str(x.device))
        return torch.pow(torch.clamp(x, min=0.0), q)
    if choice == "signed_pow":
        q = _log_uniform(generator, 0.1, 10.0, str(x.device))
        return torch.sign(x) * torch.pow(torch.abs(x), q)
    if choice == "inv_pow":
        q = _log_uniform(generator, 0.1, 10.0, str(x.device))
        return torch.pow(torch.abs(x) + 1e-3, -q)

    m = randint_scalar(2, 6, generator)
    return torch.pow(x, float(m))


def apply_random_activation(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    with_standardize: bool = True,
    parametric_prob: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Apply one random activation from supported families in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    device = str(y.device)

    if with_standardize:
        y = _standardize(y)
        a = _log_uniform(generator, 1.0, 10.0, device)
        row_idx = randint_scalar(0, y.shape[0], generator)
        b = y[int(row_idx) : int(row_idx) + 1]
        y = a * (y - b)

    if rand_scalar(generator) < parametric_prob:
        y = _param_activation(y, generator)
    else:
        fixed = fixed_activation_names()
        idx = randint_scalar(0, len(fixed), generator)
        y = _fixed_activation(y, fixed[int(idx)])

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)

    if with_standardize:
        y = _standardize(y)
    return y.to(torch.float32)
