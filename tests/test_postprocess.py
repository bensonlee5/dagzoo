"""Tests for postprocess/postprocess.py — Appendix E.13."""

import torch

from cauchy_generator.postprocess.postprocess import postprocess_dataset
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
