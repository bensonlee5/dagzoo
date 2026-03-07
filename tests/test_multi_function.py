"""Tests for functions/multi.py."""

import pytest
import torch

from dagzoo.core.layout_types import AggregationKind
from dagzoo.functions._rng_helpers import randint_scalar
from dagzoo.functions.multi import _AGGREGATION_KIND_ORDER, apply_multi_function
from dagzoo.functions.random_functions import apply_random_function
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


def _reference_apply_multi_function(
    inputs: list[torch.Tensor],
    generator: torch.Generator,
    *,
    out_dim: int,
) -> torch.Tensor:
    transformed = [apply_random_function(inp, generator, out_dim=out_dim) for inp in inputs]
    stacked = torch.stack(transformed, dim=1)
    idx = randint_scalar(0, len(_AGGREGATION_KIND_ORDER), generator)
    aggregation_kind = _AGGREGATION_KIND_ORDER[int(idx)]
    if aggregation_kind == "sum":
        return torch.sum(stacked, dim=1)
    if aggregation_kind == "product":
        return torch.prod(stacked, dim=1)
    if aggregation_kind == "max":
        return torch.max(stacked, dim=1).values
    if aggregation_kind == "logsumexp":
        return torch.logsumexp(stacked, dim=1)
    raise AssertionError(f"Unexpected aggregation kind: {aggregation_kind!r}")


def test_implicit_aggregation_preserves_reference_rng_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.functions.multi.rand_scalar", lambda _generator: 0.75)
    inputs = [
        torch.randn(64, 3, generator=_make_generator(21)),
        torch.randn(64, 2, generator=_make_generator(22)),
        torch.randn(64, 4, generator=_make_generator(23)),
    ]
    actual_generator = _make_generator(24)
    reference_generator = _make_generator(24)

    actual = apply_multi_function(
        [inp.clone() for inp in inputs],
        actual_generator,
        out_dim=5,
    )
    expected = _reference_apply_multi_function(
        [inp.clone() for inp in inputs],
        reference_generator,
        out_dim=5,
    )

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
