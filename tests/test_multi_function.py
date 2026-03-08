"""Tests for functions/multi.py."""

import pytest
import torch

from dagzoo.core.fixed_layout_batched import FixedLayoutBatchRng, apply_function_plan_batch
from dagzoo.core.fixed_layout_plan_types import (
    GaussianMatrixPlan,
    LinearFunctionPlan,
    StackedNodeSource,
)
from dagzoo.core.layout_types import AggregationKind
from dagzoo.functions.multi import apply_multi_function
from conftest import make_generator as _make_generator


def test_single_input() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function([x], g, out_dim=5)
    assert y.shape == (64, 5)


def test_multiple_inputs() -> None:
    g = _make_generator(1)
    a = torch.randn(64, 3, generator=g)
    b = torch.randn(64, 2, generator=g)
    y = apply_multi_function([a, b], g, out_dim=4)
    assert y.shape == (64, 4)


@pytest.mark.parametrize("aggregation_kind", ["sum", "product", "max", "logsumexp"])
def test_multiple_inputs_support_explicit_aggregation_kind(
    aggregation_kind: AggregationKind,
) -> None:
    g = _make_generator(2)
    a = torch.randn(64, 3, generator=g)
    b = torch.randn(64, 2, generator=g)
    y = apply_multi_function([a, b], g, out_dim=4, aggregation_kind=aggregation_kind)
    assert y.shape == (64, 4)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("aggregation_kind", ["sum", "product", "max", "logsumexp"])
def test_multiple_inputs_are_deterministic_for_explicit_aggregation_kind(
    aggregation_kind: AggregationKind,
) -> None:
    a = torch.randn(64, 3, generator=_make_generator(11))
    b = torch.randn(64, 2, generator=_make_generator(12))
    y1 = apply_multi_function(
        [a.clone(), b.clone()],
        _make_generator(13),
        out_dim=4,
        aggregation_kind=aggregation_kind,
    )
    y2 = apply_multi_function(
        [a.clone(), b.clone()],
        _make_generator(13),
        out_dim=4,
        aggregation_kind=aggregation_kind,
    )
    torch.testing.assert_close(y1, y2)


def test_multi_function_matches_explicit_stacked_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        torch.randn(64, 3, generator=_make_generator(21)),
        torch.randn(64, 2, generator=_make_generator(22)),
        torch.randn(64, 4, generator=_make_generator(23)),
    ]
    source = StackedNodeSource(
        aggregation_kind="logsumexp",
        parent_functions=(
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    monkeypatch.setattr(
        "dagzoo.functions.multi.sample_multi_source_plan", lambda *_args, **_kwargs: source
    )

    actual_generator = _make_generator(24)
    reference_generator = _make_generator(24)

    actual = apply_multi_function(
        [inp.clone() for inp in inputs],
        actual_generator,
        out_dim=5,
    )
    rng = FixedLayoutBatchRng.from_generator(reference_generator, batch_size=1, device="cpu")
    transformed = [
        apply_function_plan_batch(
            inp.unsqueeze(0),
            rng,
            source.parent_functions[plan_index],
            out_dim=5,
            noise_sigma_multiplier=1.0,
            noise_spec=None,
        ).squeeze(0)
        for plan_index, inp in enumerate(inputs)
    ]
    expected = torch.logsumexp(torch.stack(transformed, dim=1), dim=1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_empty_raises() -> None:
    g = _make_generator(0)
    with pytest.raises(ValueError, match="non-empty"):
        apply_multi_function([], g, out_dim=3)


def test_deterministic() -> None:
    x = torch.randn(32, 4, generator=_make_generator(99))
    y1 = apply_multi_function([x.clone()], _make_generator(0), out_dim=3)
    y2 = apply_multi_function([x.clone()], _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function([x], g, out_dim=3)
    assert torch.all(torch.isfinite(y))
