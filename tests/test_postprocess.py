"""Tests for postprocess/postprocess.py — Appendix E.13."""

import torch

from cauchy_generator.config import DatasetConfig
from cauchy_generator.postprocess.postprocess import inject_missingness, postprocess_dataset
from conftest import make_generator as _make_generator


def _make_data(
    generator: torch.Generator,
    n_train: int = 100,
    n_test: int = 50,
    n_feat: int = 5,
    n_classes: int = 3,
    task: str = "classification",
    add_constant_col: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    x_train = torch.randn(n_train, n_feat, generator=generator)
    x_test = torch.randn(n_test, n_feat, generator=generator)
    if add_constant_col:
        x_train[:, -1] = 1.0
        x_test[:, -1] = 1.0
    ftypes = ["num"] * n_feat

    if task == "classification":
        y_train = torch.randint(0, n_classes, (n_train,), generator=generator).to(torch.int64)
        y_test = torch.randint(0, n_classes, (n_test,), generator=generator).to(torch.int64)
    else:
        y_train = torch.randn(n_train, generator=generator)
        y_test = torch.randn(n_test, generator=generator)
    return x_train, y_train, x_test, y_test, ftypes, task


def test_removes_constant_columns() -> None:
    g = _make_generator(0)
    xt, yt, xte, yte, ft, task = _make_data(g, n_feat=4, add_constant_col=True)
    xtp, _, xtep, _, ft_out = postprocess_dataset(xt, yt, xte, yte, ft, task, g, "cpu")
    assert xtp.shape[1] < 4
    assert len(ft_out) == xtp.shape[1]


def test_standardizes_numeric() -> None:
    g = _make_generator(1)
    xt, yt, xte, yte, ft, task = _make_data(g)
    xtp, _, xtep, _, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, g, "cpu")
    combined = torch.cat([xtp, xtep], dim=0)
    for i in range(combined.shape[1]):
        col = combined[:, i]
        assert abs(float(torch.mean(col))) < 0.15
        assert float(torch.std(col)) < 1.5


def test_preserves_class_counts() -> None:
    g = _make_generator(2)
    xt, yt, xte, yte, ft, task = _make_data(g, n_classes=4)
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, g, "cpu")
    y_all_before = torch.cat([yt, yte])
    y_all_after = torch.cat([ytp, ytep])
    before_counts = sorted(torch.bincount(y_all_before).tolist())
    after_counts = sorted(torch.bincount(y_all_after).tolist())
    assert before_counts == after_counts


def test_regression_clips_targets() -> None:
    g = _make_generator(3)
    xt, yt, xte, yte, ft, task = _make_data(g, task="regression")
    yt[0] = 1e6
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, g, "cpu")
    y_all = torch.cat([ytp, ytep])
    assert torch.all(torch.isfinite(y_all))


def test_deterministic() -> None:
    g_data = _make_generator(99)
    xt, yt, xte, yte, ft, task = _make_data(g_data)
    out1 = postprocess_dataset(
        xt.clone(), yt.clone(), xte.clone(), yte.clone(), list(ft), task, _make_generator(0), "cpu"
    )
    out2 = postprocess_dataset(
        xt.clone(), yt.clone(), xte.clone(), yte.clone(), list(ft), task, _make_generator(0), "cpu"
    )
    torch.testing.assert_close(out1[0], out2[0])
    torch.testing.assert_close(out1[1], out2[1])


def test_feature_index_map_tracks_dropped_and_permuted_columns() -> None:
    g = _make_generator(10)
    xt, yt, xte, yte, _, task = _make_data(g, n_feat=5, add_constant_col=True)
    feature_types = ["num", "cat", "num", "cat", "num"]
    xtp, _, xtep, _, ft_out, feature_index_map = postprocess_dataset(
        xt,
        yt,
        xte,
        yte,
        feature_types,
        task,
        _make_generator(11),
        "cpu",
        return_feature_index_map=True,
    )

    assert xtp.shape[1] == xtep.shape[1]
    assert len(feature_index_map) == xtp.shape[1]
    assert len(feature_index_map) == len(ft_out)
    assert len(set(feature_index_map)) == len(feature_index_map)
    assert all(0 <= int(i) < len(feature_types) for i in feature_index_map)
    assert 4 not in feature_index_map
    assert [feature_types[i] for i in feature_index_map] == ft_out


def test_inject_missingness_disabled_noop() -> None:
    g = _make_generator(5)
    x_train = torch.randn(16, 4, generator=g)
    x_test = torch.randn(8, 4, generator=g)
    cfg = DatasetConfig(missing_rate=0.0, missing_mechanism="none")

    out_train, out_test, summary = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, seed=77, attempt=0, device="cpu"
    )

    torch.testing.assert_close(out_train, x_train)
    torch.testing.assert_close(out_test, x_test)
    assert summary is None


def test_inject_missingness_adds_nans_and_preserves_shape() -> None:
    g = _make_generator(6)
    x_train = torch.randn(64, 6, generator=g)
    x_test = torch.randn(32, 6, generator=g)
    cfg = DatasetConfig(missing_rate=0.3, missing_mechanism="mcar")

    out_train, out_test, summary = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, seed=88, attempt=1, device="cpu"
    )

    assert out_train.shape == x_train.shape
    assert out_test.shape == x_test.shape
    assert torch.isnan(out_train).any()
    assert torch.isnan(out_test).any()
    assert summary is not None
    assert summary["mechanism"] == "mcar"
    assert summary["target_rate"] == 0.3
    assert 0.0 <= float(summary["realized_rate_overall"]) <= 1.0


def test_inject_missingness_deterministic_for_fixed_seed_and_attempt() -> None:
    g = _make_generator(7)
    x_train = torch.randn(96, 5, generator=g)
    x_test = torch.randn(48, 5, generator=g)
    cfg = DatasetConfig(missing_rate=0.35, missing_mechanism="mar")

    a_train, a_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, seed=101, attempt=2, device="cpu"
    )
    b_train, b_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, seed=101, attempt=2, device="cpu"
    )

    assert torch.equal(torch.isnan(a_train), torch.isnan(b_train))
    assert torch.equal(torch.isnan(a_test), torch.isnan(b_test))


def test_inject_missingness_changes_for_different_seed() -> None:
    g = _make_generator(8)
    x_train = torch.randn(96, 5, generator=g)
    x_test = torch.randn(48, 5, generator=g)
    cfg = DatasetConfig(missing_rate=0.35, missing_mechanism="mnar")

    a_train, a_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, seed=202, attempt=0, device="cpu"
    )
    b_train, b_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, seed=203, attempt=0, device="cpu"
    )

    assert not torch.equal(torch.isnan(a_train), torch.isnan(b_train))
    assert not torch.equal(torch.isnan(a_test), torch.isnan(b_test))
