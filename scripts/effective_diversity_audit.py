#!/usr/bin/env python3
"""Run effective-diversity audit and write report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dagzoo.config import GeneratorConfig  # noqa: E402
from dagzoo.diagnostics.effective_diversity import (  # noqa: E402
    AuditThresholds,
    build_effective_diversity_baseline_payload,
    load_effective_diversity_baseline_payload,
    run_effective_diversity_audit,
    write_effective_diversity_baseline_payload,
    write_effective_diversity_run_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run effective-diversity audit with local-overlap and/or dataset-scale impact phases."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional generator config for scale phase (defaults to GeneratorConfig()).",
    )
    parser.add_argument(
        "--phase",
        choices=["both", "local", "scale"],
        default="both",
        help="Audit phase to run.",
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "standard", "full"],
        default="standard",
        help="Scale-phase run size (used when --num-datasets-per-arm is not set).",
    )
    parser.add_argument(
        "--arm-set",
        choices=["high_confidence", "all_claims"],
        default="high_confidence",
        help="Ablation arm set for scale phase.",
    )
    parser.add_argument(
        "--num-datasets-per-arm",
        type=int,
        default=None,
        help="Optional override for scale-phase datasets per arm.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Optional device override for scale phase generation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("effective_config_artifacts") / "effective_diversity",
        help="Artifact output directory for run/local/scale JSON and Markdown reports.",
    )
    parser.add_argument("--seed", type=int, default=2_026_0304, help="Base random seed.")
    parser.add_argument("--n-seeds", type=int, default=8, help="Number of seed runs to aggregate.")
    parser.add_argument("--n-rows", type=int, default=2_048, help="Rows per run.")
    parser.add_argument("--n-cols", type=int, default=16, help="Columns per run.")
    parser.add_argument(
        "--out-dim",
        type=int,
        default=16,
        help="Output width for function-family overlap probes.",
    )
    parser.add_argument(
        "--nn-degenerate-trials",
        type=int,
        default=50_000,
        help="Trials for estimating nn->linear degenerate path probability.",
    )
    parser.add_argument(
        "--exact-affine-rmse",
        type=float,
        default=1e-6,
        help="Threshold for exact affine equivalence label.",
    )
    parser.add_argument(
        "--near-cosine",
        type=float,
        default=0.95,
        help="Cosine threshold for near-equivalent label.",
    )
    parser.add_argument(
        "--near-affine-rmse",
        type=float,
        default=0.20,
        help="Affine RMSE threshold for near-equivalent label.",
    )
    parser.add_argument(
        "--meaningful-threshold-pct",
        type=float,
        default=5.0,
        help="Composite scale-impact threshold for meaningful overlap impact.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional scale-baseline JSON path for regression comparison.",
    )
    parser.add_argument(
        "--save-baseline",
        type=str,
        default=None,
        help="Optional path to write a scale baseline from this run.",
    )
    parser.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=2.5,
        help="Warn threshold for regression delta in composite shift percentage.",
    )
    parser.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=5.0,
        help="Fail threshold for regression delta in composite shift percentage.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return non-zero exit code when regression status is fail.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    thresholds = AuditThresholds(
        exact_affine_rmse=float(args.exact_affine_rmse),
        near_cosine=float(args.near_cosine),
        near_affine_rmse=float(args.near_affine_rmse),
    )

    base_config = GeneratorConfig.from_yaml(args.config) if args.config else GeneratorConfig()
    baseline_payload = (
        load_effective_diversity_baseline_payload(args.baseline) if args.baseline else None
    )
    report = run_effective_diversity_audit(
        base_config=base_config,
        phase=str(args.phase),
        arm_set=str(args.arm_set),
        suite=str(args.suite),
        num_datasets_per_arm=args.num_datasets_per_arm,
        device=args.device,
        seed=int(args.seed),
        n_seeds=int(args.n_seeds),
        n_rows=int(args.n_rows),
        n_cols=int(args.n_cols),
        out_dim=int(args.out_dim),
        nn_degenerate_trials=int(args.nn_degenerate_trials),
        thresholds=thresholds,
        meaningful_threshold_pct=float(args.meaningful_threshold_pct),
        baseline_payload=baseline_payload,
        warn_threshold_pct=float(args.warn_threshold_pct),
        fail_threshold_pct=float(args.fail_threshold_pct),
        out_dir=args.out_dir,
    )
    artifact_paths = write_effective_diversity_run_artifacts(report, out_dir=args.out_dir)

    for key in sorted(artifact_paths):
        print(f"Wrote {key}: {artifact_paths[key]}")

    if args.save_baseline:
        scale_report = report.get("scale_report")
        if isinstance(scale_report, dict):
            baseline = build_effective_diversity_baseline_payload(scale_report)
            baseline_path = write_effective_diversity_baseline_payload(
                baseline,
                args.save_baseline,
            )
            print(f"Wrote effective-diversity baseline: {baseline_path}")

    regression = report.get("regression")
    if isinstance(regression, dict):
        print(
            "Regression status="
            f"{regression.get('status', 'pass')} issues={len(regression.get('issues', []))}"
        )

    hard_fail = bool(
        args.fail_on_regression
        and isinstance(regression, dict)
        and str(regression.get("status", "pass")) == "fail"
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
