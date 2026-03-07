import pytest
import torch

from dagzoo.functions.random_functions import (
    MechanismFamily,
    _apply_quadratic,
    _sample_function_family,
    _apply_tree,
    apply_random_function,
)
from dagzoo.linalg.random_matrices import sample_random_matrix
from conftest import make_generator as _make_generator


def test_apply_tree_survives_nan_feature() -> None:
    """_apply_tree should produce output even when one feature column is all-NaN."""
    g = _make_generator(42)
    x = torch.randn(64, 4, generator=g)
    x[:, 2] = float("nan")

    y = _apply_tree(x, out_dim=2, generator=g)
    assert y.shape == (64, 2)


def test_apply_tree_constant_features() -> None:
    """_apply_tree should handle input where all features are constant."""
    g = _make_generator(7)
    x = torch.ones(32, 3)

    y = _apply_tree(x, out_dim=1, generator=g)
    assert y.shape == (32, 1)
    assert torch.all(torch.isfinite(y))


def test_output_shape() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function(x, g, out_dim=5)
    assert y.shape == (64, 5)


def test_deterministic() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function(x.clone(), _make_generator(0), out_dim=3)
    y2 = apply_random_function(x.clone(), _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_deterministic_with_shift_tilt_and_noise_multiplier() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function(
        x.clone(),
        _make_generator(8),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        noise_sigma_multiplier=1.35,
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(8),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        noise_sigma_multiplier=1.35,
    )
    torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize(
    "family",
    ["nn", "tree", "discretization", "gp", "linear", "quadratic", "em", "product"],
)
def test_each_family(family: MechanismFamily) -> None:
    g = _make_generator(10)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function(x, _make_generator(10), out_dim=3, function_type=family)
    assert y.shape == (64, 3)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("family", ["discretization", "quadratic"])
def test_explicit_family_is_deterministic(family: MechanismFamily) -> None:
    x = torch.randn(64, 4, generator=_make_generator(14))
    y1 = apply_random_function(
        x.clone(),
        _make_generator(15),
        out_dim=3,
        function_type=family,
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(15),
        out_dim=3,
        function_type=family,
    )
    torch.testing.assert_close(y1, y2)


def _reference_apply_quadratic(
    x: torch.Tensor,
    out_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    d_cap = min(x.shape[1], 20)
    if x.shape[1] > d_cap:
        idx = torch.randperm(x.shape[1], generator=generator, device=x.device)[:d_cap]
        x_sub = x[:, idx]
    else:
        x_sub = x
    x_aug = torch.cat([x_sub, torch.ones(x_sub.shape[0], 1, device=x.device)], dim=1)

    y = torch.empty(x_aug.shape[0], out_dim, device=x.device)
    for i in range(out_dim):
        m = sample_random_matrix(x_aug.shape[1], x_aug.shape[1], generator, str(x.device))
        y[:, i] = torch.sum((x_aug @ m) * x_aug, dim=1)
    return y


def test_quadratic_preserves_reference_fixed_seed_outputs() -> None:
    x = torch.randn(64, 24, generator=_make_generator(16))
    actual_generator = _make_generator(17)
    reference_generator = _make_generator(17)

    actual = _apply_quadratic(
        x.clone(),
        out_dim=5,
        generator=actual_generator,
    )
    expected = _reference_apply_quadratic(
        x.clone(),
        out_dim=5,
        generator=reference_generator,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_invalid_type_raises() -> None:
    g = _make_generator()
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="Unknown random function family"):
        apply_random_function(x, g, out_dim=2, function_type="bogus")  # type: ignore[arg-type]


def test_sampled_family_respects_family_mix_allowlist() -> None:
    g = _make_generator(3)
    mix = {"linear": 1.0}
    for _ in range(16):
        sampled = _sample_function_family(
            g,
            mechanism_logit_tilt=0.9,
            function_family_mix=mix,  # type: ignore[arg-type]
        )
        assert sampled == "linear"


def test_sampled_family_skips_zero_probability_families_on_zero_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _make_generator(33)
    mix = {"linear": 1.0}
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    sampled = _sample_function_family(
        g,
        mechanism_logit_tilt=0.9,
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    assert sampled == "linear"


def test_disallowed_explicit_family_raises_when_mix_is_present() -> None:
    g = _make_generator(4)
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="is not enabled by mechanism.function_family_mix"):
        apply_random_function(
            x,
            g,
            out_dim=2,
            function_type="gp",
            function_family_mix={"linear": 1.0},  # type: ignore[arg-type]
        )


def test_product_family_without_enabled_component_raises_when_mix_is_present() -> None:
    g = _make_generator(5)
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="enables 'product' but disables all product component"):
        apply_random_function(
            x,
            g,
            out_dim=2,
            function_type="product",
            function_family_mix={"product": 1.0},  # type: ignore[arg-type]
        )


def test_deterministic_with_family_mix() -> None:
    x = torch.randn(32, 4)
    mix = {"nn": 0.5, "linear": 0.5}
    y1 = apply_random_function(
        x.clone(),
        _make_generator(6),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(6),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    torch.testing.assert_close(y1, y2)
