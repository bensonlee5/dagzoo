import pytest
import torch

from dagzoo.sampling.noise import (
    NoiseSamplingSpec,
    sample_mixture_component_family,
    sample_noise,
    sample_noise_from_spec,
)
from conftest import make_generator as _make_generator
import dagzoo.sampling.noise as noise_mod


@pytest.mark.parametrize(
    ("family", "kwargs"),
    [
        ("gaussian", {}),
        ("laplace", {}),
        ("student_t", {"student_t_df": 6.0}),
        ("mixture", {"mixture_weights": {"gaussian": 0.6, "laplace": 0.2, "student_t": 0.2}}),
    ],
)
def test_sample_noise_shape_and_finite_outputs(family: str, kwargs: dict[str, object]) -> None:
    samples = sample_noise(
        (128, 4), generator=_make_generator(7), device="cpu", family=family, **kwargs
    )
    assert samples.shape == (128, 4)
    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize(
    ("family", "kwargs"),
    [
        ("gaussian", {}),
        ("laplace", {}),
        ("student_t", {"student_t_df": 7.0}),
        ("mixture", {"mixture_weights": {"gaussian": 1.0, "student_t": 1.0}}),
    ],
)
def test_sample_noise_is_deterministic_for_fixed_generator_seed(
    family: str, kwargs: dict[str, object]
) -> None:
    a = sample_noise((64,), generator=_make_generator(123), device="cpu", family=family, **kwargs)
    b = sample_noise((64,), generator=_make_generator(123), device="cpu", family=family, **kwargs)
    torch.testing.assert_close(a, b)


def test_sample_noise_mixture_is_order_independent_for_same_seed() -> None:
    a = sample_noise(
        (64,),
        generator=_make_generator(123),
        device="cpu",
        family="mixture",
        mixture_weights={"gaussian": 0.7, "laplace": 0.3},
    )
    b = sample_noise(
        (64,),
        generator=_make_generator(123),
        device="cpu",
        family="mixture",
        mixture_weights={"laplace": 0.3, "gaussian": 0.7},
    )
    torch.testing.assert_close(a, b)


def test_sample_noise_mixture_supports_nondefault_global_dtype() -> None:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        samples = sample_noise(
            (64,),
            generator=_make_generator(123),
            device="cpu",
            family="mixture",
            mixture_weights={"gaussian": 0.7, "laplace": 0.3},
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    assert samples.dtype == torch.float64
    assert torch.all(torch.isfinite(samples))


def test_sample_noise_scale_multiplier_is_applied() -> None:
    base = sample_noise(
        (64,), generator=_make_generator(314), device="cpu", family="gaussian", scale=1.0
    )
    scaled = sample_noise(
        (64,), generator=_make_generator(314), device="cpu", family="gaussian", scale=2.5
    )
    torch.testing.assert_close(scaled, base * 2.5)


def test_sample_noise_from_spec_applies_scale_multiplier() -> None:
    spec = NoiseSamplingSpec(family="gaussian", scale=1.5)
    base = sample_noise_from_spec(
        (64,),
        generator=_make_generator(314),
        device="cpu",
        noise_spec=spec,
        scale_multiplier=1.0,
    )
    scaled = sample_noise_from_spec(
        (64,),
        generator=_make_generator(314),
        device="cpu",
        noise_spec=spec,
        scale_multiplier=2.0,
    )
    torch.testing.assert_close(scaled, base * 2.0)


def test_sample_mixture_component_family_is_seed_deterministic() -> None:
    a = sample_mixture_component_family(
        generator=_make_generator(777),
        device="cpu",
        mixture_weights={"gaussian": 0.7, "laplace": 0.2, "student_t": 0.1},
    )
    b = sample_mixture_component_family(
        generator=_make_generator(777),
        device="cpu",
        mixture_weights={"gaussian": 0.7, "laplace": 0.2, "student_t": 0.1},
    )
    assert a == b
    assert a in {"gaussian", "laplace", "student_t"}


def test_sample_noise_rejects_nonpositive_shape_dims() -> None:
    with pytest.raises(ValueError, match="shape dimensions must be > 0"):
        sample_noise((16, 0), generator=_make_generator(1), device="cpu", family="gaussian")


def test_sample_noise_rejects_noninteger_shape_dims() -> None:
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        sample_noise((16, 3.9), generator=_make_generator(1), device="cpu", family="gaussian")


def test_sample_noise_rejects_boolean_shape_dims() -> None:
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        sample_noise((16, True), generator=_make_generator(1), device="cpu", family="gaussian")


def test_sample_noise_rejects_invalid_mixture_weights_usage() -> None:
    with pytest.raises(ValueError, match="only allowed when family is 'mixture'"):
        sample_noise(
            (32,),
            generator=_make_generator(2),
            device="cpu",
            family="gaussian",
            mixture_weights={"gaussian": 1.0},
        )


def test_sample_noise_rejects_nonpositive_student_t_df() -> None:
    with pytest.raises(ValueError, match="student_t_df must be a finite value > 2"):
        sample_noise(
            (32,), generator=_make_generator(2), device="cpu", family="student_t", student_t_df=2.0
        )


def test_sample_noise_rejects_mixture_with_nonpositive_total_weight() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        sample_noise(
            (32,),
            generator=_make_generator(2),
            device="cpu",
            family="mixture",
            mixture_weights={"gaussian": 0.0, "laplace": 0.0},
        )


def test_sample_noise_rejects_boolean_mixture_weights() -> None:
    with pytest.raises(ValueError, match="must be a finite value >= 0"):
        sample_noise(
            (32,),
            generator=_make_generator(2),
            device="cpu",
            family="mixture",
            mixture_weights={"gaussian": True},
        )


def test_sample_noise_handles_large_mixture_weights_without_multinomial_failure() -> None:
    samples = sample_noise(
        (32,),
        generator=_make_generator(2),
        device="cpu",
        family="mixture",
        mixture_weights={"gaussian": 1e308, "laplace": 1e308},
    )
    assert samples.shape == (32,)
    assert torch.all(torch.isfinite(samples))


def test_sample_noise_laplace_clamps_uniform_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    def _rand_zeros(*args, **kwargs) -> torch.Tensor:
        shape = args[0]
        device = kwargs.get("device", "cpu")
        return torch.zeros(shape, device=device, dtype=torch.float32)

    monkeypatch.setattr(noise_mod.torch, "rand", _rand_zeros)
    samples = sample_noise((64,), generator=_make_generator(5), device="cpu", family="laplace")
    assert torch.all(torch.isfinite(samples))


def test_sample_noise_student_t_does_not_call_global_manual_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(11)

    def _fail_manual_seed(*_args, **_kwargs):
        raise AssertionError("torch.manual_seed should not be called in student_t sampling")

    monkeypatch.setattr(noise_mod.torch, "manual_seed", _fail_manual_seed)
    samples = sample_noise((64,), generator=generator, device="cpu", family="student_t")
    assert torch.all(torch.isfinite(samples))


def test_sample_noise_mixture_student_t_does_not_call_global_manual_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(13)

    def _fail_manual_seed(*_args, **_kwargs):
        raise AssertionError("torch.manual_seed should not be called in mixture sampling")

    monkeypatch.setattr(noise_mod.torch, "manual_seed", _fail_manual_seed)
    samples = sample_noise(
        (64,),
        generator=generator,
        device="cpu",
        family="mixture",
        mixture_weights={"student_t": 1.0},
    )
    assert torch.all(torch.isfinite(samples))


def test_sample_noise_student_t_falls_back_when_standard_gamma_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(17)
    original_standard_gamma = noise_mod.torch._standard_gamma

    def _fail_for_primary_generator(alpha, *args, **kwargs):
        if kwargs.get("generator") is generator:
            raise NotImplementedError("simulated backend gap for primary generator")
        return original_standard_gamma(alpha, *args, **kwargs)

    monkeypatch.setattr(noise_mod.torch, "_standard_gamma", _fail_for_primary_generator)
    samples = sample_noise((64,), generator=generator, device="cpu", family="student_t")
    assert samples.shape == (64,)
    assert torch.all(torch.isfinite(samples))


def test_sample_noise_mixture_student_t_falls_back_when_standard_gamma_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(19)
    original_standard_gamma = noise_mod.torch._standard_gamma

    def _fail_for_primary_generator(alpha, *args, **kwargs):
        if kwargs.get("generator") is generator:
            raise NotImplementedError("simulated backend gap for primary generator")
        return original_standard_gamma(alpha, *args, **kwargs)

    monkeypatch.setattr(noise_mod.torch, "_standard_gamma", _fail_for_primary_generator)
    samples = sample_noise(
        (128,),
        generator=generator,
        device="cpu",
        family="mixture",
        mixture_weights={"student_t": 1.0},
    )
    assert samples.shape == (128,)
    assert torch.all(torch.isfinite(samples))
