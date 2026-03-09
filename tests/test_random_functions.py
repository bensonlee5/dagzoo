import pytest
import torch

from dagzoo.core.execution_semantics import sample_function_family
from dagzoo.functions.random_functions import (
    MechanismFamily,
    _sample_function_family,
    apply_random_function,
)
from conftest import make_generator as _make_generator


def test_tree_family_survives_nan_feature() -> None:
    g = _make_generator(42)
    x = torch.randn(64, 4, generator=g)
    x[:, 2] = float("nan")

    y = apply_random_function(x, _make_generator(42), out_dim=2, function_type="tree")
    assert y.shape == (64, 2)
    assert torch.all(torch.isfinite(y))

    x = torch.ones(32, 3)
    y = apply_random_function(x, _make_generator(7), out_dim=1, function_type="tree")
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


@pytest.mark.parametrize("family", ["linear", "quadratic", "discretization", "gp", "em", "product"])
def test_non_finite_inputs_are_sanitized(family: MechanismFamily) -> None:
    x = torch.tensor(
        [
            [0.0, float("nan"), 1.0],
            [float("inf"), -1.0, 2.0],
            [-float("inf"), 0.5, 3.0],
            [1.5, -2.0, 4.0],
            [2.0, 3.0, -5.0],
            [-3.0, 1.0, 6.0],
        ],
        dtype=torch.float32,
    )

    y = apply_random_function(x, _make_generator(9), out_dim=2, function_type=family)
    assert y.shape == (6, 2)
    assert torch.all(torch.isfinite(y))


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


def test_invalid_type_raises() -> None:
    g = _make_generator()
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="Unknown random function family"):
        apply_random_function(x, g, out_dim=2, function_type="bogus")  # type: ignore[arg-type]


def test_sampled_family_respects_family_mix_allowlist() -> None:
    g = _make_generator(3)
    mix = {"linear": 1.0}
    for _ in range(16):
        sampled = sample_function_family(
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
    sampled = sample_function_family(
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


def test_random_functions_module_retains_family_sampling_patch_point() -> None:
    g = _make_generator(21)
    sampled = _sample_function_family(
        g,
        mechanism_logit_tilt=0.0,
        function_family_mix={"linear": 1.0},  # type: ignore[arg-type]
    )
    assert sampled == "linear"
