import pytest
import torch

from dagzoo.converters.categorical import apply_categorical_converter
from dagzoo.converters.numeric import apply_numeric_converter
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout_batched import (
    FixedLayoutBatchRng,
    _apply_categorical_group_batch,
    _apply_node_plan_batch,
    _sample_random_points_batch,
    apply_numeric_converter_plan_batch,
    apply_function_plan_batch,
)
from dagzoo.core.fixed_layout_plan_types import (
    CategoricalConverterPlan,
    FixedActivationPlan,
    FixedLayoutLatentPlan,
    FixedLayoutNodePlan,
    GaussianMatrixPlan,
    GpFunctionPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    TreeFunctionPlan,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    fixed_layout_converter_groups,
)
from dagzoo.core.node_pipeline import ConverterSpec, apply_node_pipeline
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.sampling.random_points import sample_random_points
from conftest import make_generator as _make_generator
import dagzoo.converters.categorical as categorical_mod
import dagzoo.converters.numeric as numeric_mod
import dagzoo.core.node_pipeline as node_pipeline_mod
import dagzoo.functions.random_functions as random_functions_mod
import dagzoo.sampling.random_points as random_points_mod


@pytest.mark.parametrize(
    ("family", "plan"),
    [
        ("linear", LinearFunctionPlan(matrix=GaussianMatrixPlan())),
        ("quadratic", QuadraticFunctionPlan(matrix=GaussianMatrixPlan())),
        (
            "nn",
            NeuralNetFunctionPlan(
                n_layers=2,
                hidden_width=5,
                input_activation=FixedActivationPlan(name="relu"),
                output_activation=None,
                layer_matrices=(GaussianMatrixPlan(), GaussianMatrixPlan()),
                hidden_activations=(FixedActivationPlan(name="tanh"),),
            ),
        ),
        ("tree", TreeFunctionPlan(n_trees=2, depths=(2, 3))),
        (
            "discretization",
            DiscretizationFunctionPlan(
                n_centers=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
        ),
        ("gp", GpFunctionPlan(branch_kind="ha")),
        (
            "em",
            EmFunctionPlan(
                m_val=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
        ),
        (
            "product",
            ProductFunctionPlan(
                lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                rhs=QuadraticFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        ),
    ],
)
def test_apply_random_function_matches_explicit_plan(
    family: str,
    plan: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(32, 4, generator=_make_generator(1))
    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        lambda *_args, **_kwargs: plan,
    )

    actual_generator = _make_generator(2)
    reference_generator = _make_generator(2)
    actual = apply_random_function(x.clone(), actual_generator, out_dim=3, function_type=family)  # type: ignore[arg-type]

    rng = FixedLayoutBatchRng.from_generator(reference_generator, batch_size=1, device="cpu")
    expected = apply_function_plan_batch(
        random_functions_mod._standardize(x).unsqueeze(0),
        rng,
        plan,  # type: ignore[arg-type]
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        standardize_input=False,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_numeric_converter_matches_explicit_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(24, 1, generator=_make_generator(10))
    plan = NumericConverterPlan(kind="num", warp_enabled=True)
    monkeypatch.setattr(numeric_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    actual_generator = _make_generator(11)
    reference_generator = _make_generator(11)
    actual_x, actual_values = apply_numeric_converter(x.clone(), actual_generator)

    rng = FixedLayoutBatchRng.from_generator(reference_generator, batch_size=1, device="cpu")
    expected_x, expected_values = apply_numeric_converter_plan_batch(
        x.unsqueeze(0),
        rng,
        plan,
    )

    torch.testing.assert_close(actual_x, expected_x.squeeze(0))
    torch.testing.assert_close(actual_values, expected_values.squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_categorical_converter_matches_explicit_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(24, 3, generator=_make_generator(20))
    plan = CategoricalConverterPlan(
        kind="cat",
        method="neighbor",
        variant="center_random_fn",
        function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(categorical_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    actual_generator = _make_generator(21)
    reference_generator = _make_generator(21)
    actual_x, actual_labels = apply_categorical_converter(
        x.clone(), actual_generator, n_categories=5
    )

    rng = FixedLayoutBatchRng.from_generator(reference_generator, batch_size=1, device="cpu")
    expected_x, expected_labels = _apply_categorical_group_batch(
        x.unsqueeze(0).unsqueeze(2),
        rng,
        plan,
        n_categories=5,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(actual_x, expected_x[0, :, 0, :])
    torch.testing.assert_close(actual_labels, expected_labels[0, :, 0])
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_sample_random_points_matches_explicit_root_source_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RandomPointsNodeSource(
        base_kind="uniform",
        function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(
        random_points_mod, "sample_root_source_plan", lambda *_args, **_kwargs: source
    )

    actual_generator = _make_generator(30)
    reference_generator = _make_generator(30)
    actual = sample_random_points(32, 4, actual_generator, "cpu")

    rng = FixedLayoutBatchRng.from_generator(reference_generator, batch_size=1, device="cpu")
    base = _sample_random_points_batch(
        rng,
        n_rows=32,
        dim=4,
        base_kind=source.base_kind,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    expected = apply_function_plan_batch(
        base,
        rng,
        source.function,
        out_dim=4,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_node_pipeline_matches_explicit_node_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=4),
    ]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (
        NumericConverterPlan(kind="num", warp_enabled=True),
        CategoricalConverterPlan(
            kind="cat",
            method="softmax",
            variant="softmax_points",
        ),
    )
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0,),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=4, extra_dim=2, total_dim=6),
        source=StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=(LinearFunctionPlan(matrix=GaussianMatrixPlan()),),
        ),
    )
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: node_plan)

    parent = torch.randn(16, 4, generator=_make_generator(40))
    actual_generator = _make_generator(41)
    reference_generator = _make_generator(41)
    actual_latent, actual_extracted = apply_node_pipeline(
        [parent.clone()],
        16,
        specs,
        actual_generator,
        "cpu",
    )

    rng = FixedLayoutBatchRng.from_generator(reference_generator, batch_size=1, device="cpu")
    expected_latent, expected_extracted = _apply_node_plan_batch(
        None,
        node_plan,
        [parent.unsqueeze(0)],
        n_rows=16,
        rng=rng,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(actual_latent, expected_latent.squeeze(0))
    for key, value in actual_extracted.items():
        torch.testing.assert_close(value, expected_extracted[key].squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())
