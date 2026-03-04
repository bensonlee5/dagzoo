import pytest

from dagzoo.config import (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_MIXTURE,
    NOISE_FAMILY_STUDENT_T,
    GeneratorConfig,
)


def test_noise_config_defaults_from_default_yaml() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.noise.family == NOISE_FAMILY_GAUSSIAN
    assert cfg.noise.base_scale == pytest.approx(1.0)
    assert cfg.noise.student_t_df == pytest.approx(5.0)
    assert cfg.noise.mixture_weights is None


@pytest.mark.parametrize(
    "family",
    [
        NOISE_FAMILY_GAUSSIAN,
        NOISE_FAMILY_LAPLACE,
        NOISE_FAMILY_STUDENT_T,
        NOISE_FAMILY_MIXTURE,
    ],
)
def test_noise_config_accepts_supported_families(family: str) -> None:
    cfg = GeneratorConfig.from_dict({"noise": {"family": family}})
    assert cfg.noise.family == family


def test_noise_config_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unsupported noise.family"):
        GeneratorConfig.from_dict({"noise": {"family": "exponential"}})


def test_noise_config_rejects_removed_legacy_family() -> None:
    with pytest.raises(ValueError, match="Unsupported noise.family"):
        GeneratorConfig.from_dict({"noise": {"family": "legacy"}})


def test_noise_config_rejects_student_t_df_at_or_below_two() -> None:
    with pytest.raises(ValueError, match=r"noise\.student_t_df must be a finite value > 2"):
        GeneratorConfig.from_dict({"noise": {"family": "student_t", "student_t_df": 2.0}})


def test_noise_config_rejects_mixture_weights_when_family_not_mixture() -> None:
    with pytest.raises(ValueError, match="noise.mixture_weights is only allowed"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": "gaussian",
                    "mixture_weights": {"gaussian": 0.5, "laplace": 0.5},
                }
            }
        )


def test_noise_config_normalizes_mixture_weights() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "noise": {
                "family": "mixture",
                "mixture_weights": {"gaussian": 2.0, "laplace": 1.0, "student_t": 1.0},
            }
        }
    )
    weights = cfg.noise.mixture_weights
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["gaussian"] == pytest.approx(0.5)
    assert weights["laplace"] == pytest.approx(0.25)
    assert weights["student_t"] == pytest.approx(0.25)


def test_noise_config_rejects_mixture_weights_with_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported noise.mixture_weights key"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": "mixture",
                    "mixture_weights": {"gaussian": 1.0, "exponential": 1.0},
                }
            }
        )


def test_noise_config_rejects_mixture_weights_with_nonpositive_total() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": "mixture",
                    "mixture_weights": {"gaussian": 0.0, "laplace": 0.0, "student_t": 0.0},
                }
            }
        )


def test_noise_config_stably_normalizes_large_mixture_weights() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "noise": {
                "family": "mixture",
                "mixture_weights": {"gaussian": 1e308, "laplace": 1e308},
            }
        }
    )
    weights = cfg.noise.mixture_weights
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["gaussian"] == pytest.approx(0.5)
    assert weights["laplace"] == pytest.approx(0.5)
