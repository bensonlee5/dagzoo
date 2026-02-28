import pytest
import torch

from cauchy_generator.sampling.noise import sample_noise
from conftest import make_generator as _make_generator


@pytest.mark.parametrize(
    ("family", "kwargs"),
    [
        ("legacy", {}),
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


def test_sample_noise_scale_multiplier_is_applied() -> None:
    base = sample_noise(
        (64,), generator=_make_generator(314), device="cpu", family="gaussian", scale=1.0
    )
    scaled = sample_noise(
        (64,), generator=_make_generator(314), device="cpu", family="gaussian", scale=2.5
    )
    torch.testing.assert_close(scaled, base * 2.5)


def test_sample_noise_rejects_nonpositive_shape_dims() -> None:
    with pytest.raises(ValueError, match="shape dimensions must be > 0"):
        sample_noise((16, 0), generator=_make_generator(1), device="cpu", family="gaussian")


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
