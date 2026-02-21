import torch

from cauchy_generator.functions.random_functions import _apply_tree_torch


def test_apply_tree_torch_survives_nan_feature() -> None:
    """_apply_tree_torch should produce output even when one feature column is all-NaN."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    x = torch.randn(64, 4, generator=g)
    x[:, 2] = float("nan")  # one column is entirely NaN

    # Before fix: RuntimeError from torch.multinomial on NaN probs
    # After fix: falls back to uniform probs, returns without raising
    y = _apply_tree_torch(x, out_dim=2, generator=g)
    assert y.shape == (64, 2)


def test_apply_tree_torch_constant_features() -> None:
    """_apply_tree_torch should handle input where all features are constant."""
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    x = torch.ones(32, 3)

    y = _apply_tree_torch(x, out_dim=1, generator=g)
    assert y.shape == (32, 1)
    assert torch.all(torch.isfinite(y))
