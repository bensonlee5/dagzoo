import math

import pytest

from dagsynth.config import GeneratorConfig


def test_mechanism_family_mix_defaults_to_none() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.mechanism.function_family_mix is None


def test_mechanism_family_mix_accepts_partial_map_and_normalizes() -> None:
    cfg = GeneratorConfig.from_dict(
        {"mechanism": {"function_family_mix": {"NN": 3.0, "linear": 1.0, "em": 0.0}}}
    )
    mix = cfg.mechanism.function_family_mix
    assert mix is not None
    assert set(mix) == {"nn", "linear"}
    assert mix["nn"] == pytest.approx(0.75)
    assert mix["linear"] == pytest.approx(0.25)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_mechanism_family_mix_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported mechanism.function_family_mix key"):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": {"bogus": 1.0}}})


@pytest.mark.parametrize("value", [[], 1, "not-a-map", True])
def test_mechanism_family_mix_rejects_non_mapping(value: object) -> None:
    with pytest.raises(ValueError, match="mechanism.function_family_mix must be a mapping"):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": value}})


def test_mechanism_family_mix_rejects_empty_mapping() -> None:
    with pytest.raises(
        ValueError,
        match="mechanism.function_family_mix must include at least one supported family",
    ):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": {}}})


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan"), True])
def test_mechanism_family_mix_rejects_invalid_weights(value: float | bool) -> None:
    with pytest.raises(
        ValueError,
        match=r"mechanism\.function_family_mix\.nn must be a finite value >= 0",
    ):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": {"nn": value}}})


def test_mechanism_family_mix_rejects_nonpositive_total_weight() -> None:
    with pytest.raises(
        ValueError,
        match="mechanism.function_family_mix must have a positive total weight",
    ):
        GeneratorConfig.from_dict(
            {"mechanism": {"function_family_mix": {"nn": 0.0, "linear": 0.0}}}
        )


def test_mechanism_family_mix_rejects_product_without_component_family() -> None:
    with pytest.raises(
        ValueError,
        match="assigns positive weight to 'product' but none of its component families are enabled",
    ):
        GeneratorConfig.from_dict(
            {"mechanism": {"function_family_mix": {"product": 0.7, "nn": 0.3}}}
        )


def test_mechanism_family_mix_accepts_product_with_component_family() -> None:
    cfg = GeneratorConfig.from_dict(
        {"mechanism": {"function_family_mix": {"product": 0.4, "linear": 0.6}}}
    )
    mix = cfg.mechanism.function_family_mix
    assert mix is not None
    assert math.isclose(sum(mix.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert set(mix) == {"product", "linear"}
