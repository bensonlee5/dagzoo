import copy
import json

import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.effective_diversity import (
    build_effective_diversity_baseline_payload,
    compare_scale_report_to_baseline,
    generate_effective_diversity_report,
    generate_effective_diversity_scale_report,
    write_effective_diversity_artifacts,
)
from dagzoo.functions.activations import fixed_activation_names


def _find_pair(pairs: list[dict[str, object]], left: str, right: str) -> dict[str, object]:
    target = tuple(sorted((left, right)))
    for pair in pairs:
        current = tuple(sorted((str(pair["left"]), str(pair["right"]))))
        if current == target:
            return pair
    raise AssertionError(f"Pair not found: {left!r}, {right!r}")


def test_effective_diversity_report_is_deterministic() -> None:
    report_a = generate_effective_diversity_report(
        seed=1234,
        n_seeds=3,
        n_rows=256,
        n_cols=8,
        out_dim=8,
        nn_degenerate_trials=2_000,
    )
    report_b = generate_effective_diversity_report(
        seed=1234,
        n_seeds=3,
        n_rows=256,
        n_cols=8,
        out_dim=8,
        nn_degenerate_trials=2_000,
    )

    assert report_a["schema_name"] == "dagzoo_effective_diversity_audit"
    assert report_a["schema_version"] == 2
    assert sorted(report_a["tracks"].keys()) == ["activations", "aggregation", "function_families"]
    assert report_a["config"]["runtime_activation_names"] == list(fixed_activation_names())
    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)


def test_effective_diversity_registry_captures_claims_and_runtime_absence() -> None:
    report = generate_effective_diversity_report(
        seed=9,
        n_seeds=4,
        n_rows=384,
        n_cols=10,
        out_dim=10,
        nn_degenerate_trials=12_000,
    )

    claim_ids = {str(item["claim_id"]) for item in report["hypothesis_registry"]}
    expected_claim_ids = {
        "sigmoid_vs_tanh",
        "sign_vs_heaviside",
        "softplus_vs_logsigmoid",
        "relu6_vs_hardtanh",
        "selu_vs_elu",
        "linear_vs_nn",
        "quadratic_vs_product",
        "gp_vs_nn",
        "em_vs_discretization",
        "tree_vs_discretization",
        "max_vs_logsumexp",
        "rank_vs_argsort",
    }
    assert claim_ids == expected_claim_ids

    sign_heaviside = next(
        item for item in report["hypothesis_registry"] if item["claim_id"] == "sign_vs_heaviside"
    )
    assert sign_heaviside["status"] == "not_applicable_runtime_absent"

    activation_pairs = report["tracks"]["activations"]["pairs"]
    family_pairs = report["tracks"]["function_families"]["pairs"]

    relu6_hardtanh = _find_pair(activation_pairs, "relu6", "hardtanh")
    assert relu6_hardtanh["label"] != "exact_affine_equivalent"

    nn_linear = _find_pair(family_pairs, "nn", "linear")
    assert nn_linear["label"] != "exact_affine_equivalent"

    nn_prob = report["hypothesis_checks"]["nn_linear_degenerate_probability"]
    assert float(nn_prob["analytic"]) == pytest.approx(1.0 / 12.0)
    assert float(nn_prob["empirical"]) == pytest.approx(1.0 / 12.0, abs=0.02)


def test_effective_diversity_scale_report_smoke_and_baseline_regression() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.dataset.n_train = 24
    cfg.dataset.n_test = 12
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 6

    scale_report = generate_effective_diversity_scale_report(
        base_config=cfg,
        arm_set="high_confidence",
        suite="smoke",
        num_datasets_per_arm=2,
        seed=77,
        meaningful_threshold_pct=5.0,
    )

    assert scale_report["schema_name"] == "dagzoo_effective_diversity_scale_impact"
    assert scale_report["baseline"]["status"] == "executed"
    assert len(scale_report["comparisons"]) == len(scale_report["arms"])
    assert scale_report["summary"]["num_arms_total"] == len(scale_report["arms"])

    skipped = [item for item in scale_report["comparisons"] if item["status"] == "skipped"]
    assert any(item["arm_id"] == "act_sign_to_heaviside" for item in skipped)

    evaluated = [item for item in scale_report["comparisons"] if item["status"] == "evaluated"]
    assert evaluated
    assert all(item["composite_shift_pct"] is not None for item in evaluated)

    baseline_payload = build_effective_diversity_baseline_payload(scale_report)
    same_regression = compare_scale_report_to_baseline(
        scale_report,
        baseline_payload,
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )
    assert same_regression["status"] == "pass"
    assert same_regression["issues"] == []

    mutated = copy.deepcopy(scale_report)
    for comparison in mutated["comparisons"]:
        if comparison.get("status") == "evaluated":
            comparison["composite_shift_pct"] = float(comparison["composite_shift_pct"]) + 8.0
            break

    regressed = compare_scale_report_to_baseline(
        mutated,
        baseline_payload,
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )
    assert regressed["status"] in {"warn", "fail"}
    assert regressed["issues"]


def test_effective_diversity_artifact_writer(tmp_path) -> None:
    report = generate_effective_diversity_report(
        seed=77,
        n_seeds=2,
        n_rows=128,
        n_cols=6,
        out_dim=6,
        nn_degenerate_trials=1_000,
    )
    json_path, md_path = write_effective_diversity_artifacts(report, out_dir=tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "dagzoo_effective_diversity_audit"
    assert "No generator behavior/config/schema was changed by this audit." in md_path.read_text(
        encoding="utf-8"
    )
