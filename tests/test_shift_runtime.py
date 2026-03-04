from __future__ import annotations

import math

import pytest

from dagzoo.config import (
    SHIFT_MODE_GRAPH_DRIFT,
    SHIFT_MODE_MECHANISM_DRIFT,
    SHIFT_MODE_MIXED,
    SHIFT_MODE_NOISE_DRIFT,
    GeneratorConfig,
)
from dagzoo.core.shift import (
    MECHANISM_FAMILY_ORDER,
    mechanism_nonlinear_mass,
    mechanism_family_probabilities,
    resolve_shift_runtime_params,
)


def _cfg() -> GeneratorConfig:
    return GeneratorConfig.from_yaml("configs/default.yaml")


def _entropy_bits(probabilities: list[float]) -> float:
    return float(-sum(p * math.log(p, 2) for p in probabilities if p > 0.0))


def _nonlinear_mass(probs: dict[str, float]) -> float:
    nonlinear = {"nn", "bnn", "tree", "discretization", "gp", "product"}
    return float(sum(prob for family, prob in probs.items() if family in nonlinear))


def test_resolve_shift_runtime_params_disabled_returns_identity() -> None:
    cfg = _cfg()
    cfg.shift.enabled = False
    cfg.shift.mode = "off"
    params = resolve_shift_runtime_params(cfg)
    assert params.enabled is False
    assert params.mode == "off"
    assert params.graph_scale == 0.0
    assert params.mechanism_scale == 0.0
    assert params.variance_scale == 0.0
    assert params.edge_logit_bias_shift == 0.0
    assert params.mechanism_logit_tilt == 0.0
    assert params.variance_sigma_multiplier == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("profile", "expected_scales"),
    [
        (SHIFT_MODE_GRAPH_DRIFT, (0.5, 0.0, 0.0)),
        (SHIFT_MODE_MECHANISM_DRIFT, (0.0, 0.5, 0.0)),
        (SHIFT_MODE_NOISE_DRIFT, (0.0, 0.0, 0.5)),
        (SHIFT_MODE_MIXED, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    ],
)
def test_resolve_shift_runtime_params_uses_profile_defaults(
    profile: str, expected_scales: tuple[float, float, float]
) -> None:
    cfg = _cfg()
    cfg.shift.enabled = True
    cfg.shift.mode = profile
    params = resolve_shift_runtime_params(cfg)
    assert params.enabled is True
    assert params.mode == profile
    assert params.graph_scale == pytest.approx(expected_scales[0])
    assert params.mechanism_scale == pytest.approx(expected_scales[1])
    assert params.variance_scale == pytest.approx(expected_scales[2])


def test_resolve_shift_runtime_params_prioritizes_explicit_overrides() -> None:
    cfg = _cfg()
    cfg.shift.enabled = True
    cfg.shift.mode = SHIFT_MODE_MIXED
    cfg.shift.graph_scale = 0.9
    cfg.shift.mechanism_scale = 0.1
    cfg.shift.variance_scale = 0.4
    params = resolve_shift_runtime_params(cfg)
    assert params.graph_scale == pytest.approx(0.9)
    assert params.mechanism_scale == pytest.approx(0.1)
    assert params.variance_scale == pytest.approx(0.4)


def test_resolve_shift_runtime_params_matches_formula_mappings() -> None:
    cfg = _cfg()
    cfg.shift.enabled = True
    cfg.shift.mode = SHIFT_MODE_GRAPH_DRIFT
    cfg.shift.graph_scale = 0.75
    cfg.shift.mechanism_scale = 0.25
    cfg.shift.variance_scale = 0.5
    params = resolve_shift_runtime_params(cfg)
    assert params.edge_logit_bias_shift == pytest.approx(math.log(2.0) * 0.75)
    assert params.mechanism_logit_tilt == pytest.approx(0.25)
    assert params.variance_sigma_multiplier == pytest.approx(math.exp((math.log(2.0) / 2.0) * 0.5))


def test_mechanism_family_probabilities_are_uniform_when_tilt_is_zero() -> None:
    probs = mechanism_family_probabilities(mechanism_logit_tilt=0.0)
    expected = 1.0 / float(len(MECHANISM_FAMILY_ORDER))
    assert set(probs) == set(MECHANISM_FAMILY_ORDER)
    for prob in probs.values():
        assert prob == pytest.approx(expected)


def test_mechanism_family_probabilities_tilt_increases_nonlinear_mass() -> None:
    probs_uniform = mechanism_family_probabilities(mechanism_logit_tilt=0.0)
    probs_tilted = mechanism_family_probabilities(mechanism_logit_tilt=1.0)
    entropy_uniform = _entropy_bits([probs_uniform[f] for f in MECHANISM_FAMILY_ORDER])
    entropy_tilted = _entropy_bits([probs_tilted[f] for f in MECHANISM_FAMILY_ORDER])

    assert _nonlinear_mass(probs_tilted) > _nonlinear_mass(probs_uniform)
    assert entropy_tilted < entropy_uniform


def test_mechanism_nonlinear_mass_matches_probability_sum() -> None:
    tilt = 0.7
    probs = mechanism_family_probabilities(mechanism_logit_tilt=tilt)
    assert mechanism_nonlinear_mass(mechanism_logit_tilt=tilt) == pytest.approx(
        _nonlinear_mass(probs)
    )


def test_mechanism_family_probabilities_respect_family_mix_hard_mask() -> None:
    probs = mechanism_family_probabilities(
        mechanism_logit_tilt=0.0,
        family_weights={"nn": 0.7, "linear": 0.3},
    )
    assert probs["nn"] == pytest.approx(0.7)
    assert probs["linear"] == pytest.approx(0.3)
    for family in MECHANISM_FAMILY_ORDER:
        if family not in {"nn", "linear"}:
            assert probs[family] == pytest.approx(0.0)


def test_mechanism_family_probabilities_tilt_reweights_within_mixed_support() -> None:
    probs = mechanism_family_probabilities(
        mechanism_logit_tilt=1.0,
        family_weights={"nn": 0.5, "linear": 0.5},
    )
    assert probs["nn"] > probs["linear"]
    for family in MECHANISM_FAMILY_ORDER:
        if family not in {"nn", "linear"}:
            assert probs[family] == pytest.approx(0.0)


def test_mechanism_family_probabilities_reject_nonpositive_weight_support() -> None:
    with pytest.raises(ValueError, match="must include at least one positive family weight"):
        mechanism_family_probabilities(
            mechanism_logit_tilt=0.0,
            family_weights={"nn": 0.0},
        )


def test_mechanism_nonlinear_mass_respects_family_mix() -> None:
    assert mechanism_nonlinear_mass(
        mechanism_logit_tilt=1.0,
        family_weights={"linear": 1.0},
    ) == pytest.approx(0.0)
    assert mechanism_nonlinear_mass(
        mechanism_logit_tilt=1.0,
        family_weights={"nn": 1.0},
    ) == pytest.approx(1.0)
    assert mechanism_nonlinear_mass(
        mechanism_logit_tilt=1.0,
        family_weights={"bnn": 1.0},
    ) == pytest.approx(1.0)
