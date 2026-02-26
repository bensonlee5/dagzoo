"""CLI entrypoints."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cauchy_generator.bench.baseline import build_baseline_payload, load_baseline, write_baseline
from cauchy_generator.bench.report import write_suite_json, write_suite_markdown
from cauchy_generator.bench.suite import resolve_profile_run_specs, run_benchmark_suite
from cauchy_generator.config import (
    CURRICULUM_STAGE_AUTO,
    CURRICULUM_STAGE_CLI_CHOICES,
    DatasetConfig,
    GeneratorConfig,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    normalize_missing_mechanism,
)
from cauchy_generator.core.dataset import generate_batch_iter
from cauchy_generator.diagnostics import (
    CoverageAggregationConfig,
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from cauchy_generator.hardware import (
    HardwareInfo,
    apply_hardware_profile,
    detect_hardware,
)
from cauchy_generator.io.parquet_writer import write_parquet_shards_stream
from cauchy_generator.meta_targets import (
    CLASSIFICATION_ONLY_METRICS,
    SUPPORTED_METRICS,
    coerce_quantiles,
    collect_unknown_target_metrics,
    merge_weighted_target_specs,
    validate_target_specs_for_task,
    weighted_specs_to_bands,
)

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
MISSINGNESS_MECHANISM_CLI_CHOICES = (
    MISSINGNESS_MECHANISM_NONE,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MNAR,
)


def _positive_int(value: str) -> int:
    """argparse type: parse an integer > 0."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}.")
    return parsed


def _non_negative_int(value: str) -> int:
    """argparse type: parse an integer >= 0."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value}.")
    return parsed


def _parse_finite_float(raw: str, *, flag: str) -> float:
    """argparse helper: parse a finite float."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid {flag} value '{raw}'. Expected a number."
        ) from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"Invalid {flag} value '{raw}'. Expected a finite number.")
    return value


def _parse_bounded_float(
    raw: str,
    *,
    flag: str,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float:
    """argparse helper: parse a finite float and enforce explicit numeric bounds."""

    value = _parse_finite_float(raw, flag=flag)
    lo_ok = value >= lo if lo_inclusive else value > lo
    hi_ok = True
    if hi is not None:
        hi_ok = value <= hi if hi_inclusive else value < hi
    if lo_ok and hi_ok:
        return value
    raise argparse.ArgumentTypeError(f"Invalid {flag} value '{raw}'. Expected {expectation}.")


def _parse_missing_rate_arg(raw: str) -> float:
    """argparse type: parse missing rate in [0, 1]."""

    return _parse_bounded_float(
        raw,
        flag="--missing-rate",
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )


def _parse_missing_mar_observed_fraction_arg(raw: str) -> float:
    """argparse type: parse MAR observed-feature fraction in (0, 1]."""

    return _parse_bounded_float(
        raw,
        flag="--missing-mar-observed-fraction",
        lo=0.0,
        hi=1.0,
        lo_inclusive=False,
        hi_inclusive=True,
        expectation="a finite value in (0, 1]",
    )


def _parse_missing_mar_logit_scale_arg(raw: str) -> float:
    """argparse type: parse MAR logit scale > 0."""

    return _parse_bounded_float(
        raw,
        flag="--missing-mar-logit-scale",
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def _parse_missing_mnar_logit_scale_arg(raw: str) -> float:
    """argparse type: parse MNAR logit scale > 0."""

    return _parse_bounded_float(
        raw,
        flag="--missing-mnar-logit-scale",
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def _parse_missing_mechanism_arg(raw: str) -> str:
    """argparse type: normalize missingness mechanism values."""

    try:
        return normalize_missing_mechanism(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_meta_target_arg(raw: str) -> tuple[str, tuple[float, float, float]]:
    """argparse type: parse `metric=min:max[:weight]` target overrides."""

    value = raw.strip()
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --meta-target '{raw}'. Expected format: key=min:max[:weight]."
        )
    metric_name, band = value.split("=", maxsplit=1)
    metric_name = metric_name.strip()
    if not metric_name:
        raise argparse.ArgumentTypeError(
            f"Invalid --meta-target '{raw}'. Metric key must be non-empty."
        )
    if metric_name not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise argparse.ArgumentTypeError(
            f"Unsupported --meta-target metric '{metric_name}'. Supported metrics: {supported}."
        )
    parts = [part.strip() for part in band.split(":")]
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            f"Invalid --meta-target '{raw}'. Expected exactly two or three values after '='."
        )
    try:
        lo = float(parts[0])
        hi = float(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else 1.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --meta-target '{raw}'. min/max/weight must be numeric."
        ) from exc
    if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(weight)):
        raise argparse.ArgumentTypeError(
            f"Invalid --meta-target '{raw}'. min/max/weight must be finite."
        )
    if weight <= 0:
        raise argparse.ArgumentTypeError(f"Invalid --meta-target '{raw}'. weight must be > 0.")
    if lo > hi:
        lo, hi = hi, lo
    return metric_name, (lo, hi, weight)


def _resolve_meta_target_specs(
    config: GeneratorConfig,
    cli_overrides: list[tuple[str, tuple[float, float, float]]] | None,
) -> dict[str, tuple[float, float, float]]:
    """Merge target specs across legacy diagnostics, top-level config, and CLI."""

    resolved = merge_weighted_target_specs(
        config.diagnostics.meta_feature_targets,
        config.meta_feature_targets,
        supported_metrics=SUPPORTED_METRICS,
    )
    if cli_overrides:
        for metric_name, spec in cli_overrides:
            resolved[metric_name] = spec
    return resolved


def _target_specs_to_bands(
    target_specs: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float]]:
    """Drop steering weights for diagnostics coverage aggregation payload."""

    return weighted_specs_to_bands(target_specs)


def _raise_usage_error(message: str) -> None:
    """Exit with argparse-compatible usage error semantics."""

    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser and register all subcommands/options."""

    parser = argparse.ArgumentParser(prog="cauchy-gen")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate synthetic datasets.")
    g.add_argument("--config", required=True, help="Path to YAML config.")
    g.add_argument("--out", default=None, help="Output directory for parquet shards.")
    g.add_argument(
        "--num-datasets",
        type=_positive_int,
        default=10,
        help="Number of datasets to generate.",
    )
    g.add_argument("--seed", type=int, default=None, help="Optional override for run seed.")
    g.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Device override (auto/cpu/cuda/mps).",
    )
    g.add_argument(
        "--no-hardware-aware",
        action="store_true",
        help="Disable automatic hardware-based config tuning.",
    )
    g.add_argument(
        "--no-write",
        action="store_true",
        help="Generate in memory only and do not write parquet files.",
    )
    g.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable diagnostics coverage aggregation artifacts for this run.",
    )
    g.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional directory for diagnostics artifacts (defaults to output directory).",
    )
    g.add_argument(
        "--steer-meta",
        action="store_true",
        help="Enable soft steering toward configured meta-feature target bands.",
    )
    g.add_argument(
        "--meta-target",
        action="append",
        default=None,
        type=_parse_meta_target_arg,
        metavar="KEY=MIN:MAX[:WEIGHT]",
        help=(
            "Repeatable meta-feature target override used by diagnostics and steering. "
            "Format: key=min:max[:weight]."
        ),
    )
    g.add_argument(
        "--missing-rate",
        type=_parse_missing_rate_arg,
        default=None,
        help="Override dataset missing rate in [0, 1].",
    )
    g.add_argument(
        "--missing-mechanism",
        type=_parse_missing_mechanism_arg,
        choices=MISSINGNESS_MECHANISM_CLI_CHOICES,
        default=None,
        help="Override missingness mechanism (none/mcar/mar/mnar).",
    )
    g.add_argument(
        "--missing-mar-observed-fraction",
        type=_parse_missing_mar_observed_fraction_arg,
        default=None,
        help="Override MAR observed-feature fraction in (0, 1].",
    )
    g.add_argument(
        "--missing-mar-logit-scale",
        type=_parse_missing_mar_logit_scale_arg,
        default=None,
        help="Override MAR logit scale (> 0).",
    )
    g.add_argument(
        "--missing-mnar-logit-scale",
        type=_parse_missing_mnar_logit_scale_arg,
        default=None,
        help="Override MNAR logit scale (> 0).",
    )
    g.add_argument(
        "--curriculum",
        default=None,
        choices=CURRICULUM_STAGE_CLI_CHOICES,
        help="Optional staged curriculum override (auto/1/2/3) for this run.",
    )
    b = sub.add_parser("benchmark", help="Run benchmark suite across one or more profiles.")
    b.add_argument("--config", default=None, help="Optional YAML config for profile 'custom'.")
    b.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Device override for custom profile.",
    )
    b.add_argument(
        "--num-datasets",
        type=_positive_int,
        default=None,
        help="Override benchmark dataset count.",
    )
    b.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=None,
        help="Override benchmark warmup count.",
    )
    b.add_argument(
        "--no-hardware-aware",
        action="store_true",
        help="Disable automatic hardware-based config tuning.",
    )
    b.add_argument("--json-out", default=None, help="Optional path to write suite summary JSON.")
    b.add_argument(
        "--suite",
        default=None,
        choices=["smoke", "standard", "full"],
        help="Benchmark suite level. Defaults to config benchmark.suite.",
    )
    b.add_argument(
        "--profile",
        action="append",
        default=None,
        choices=["all", "cpu", "cuda_desktop", "cuda_h100", "custom"],
        help="Benchmark profile key. Repeat to run multiple profiles.",
    )
    b.add_argument(
        "--baseline", default=None, help="Optional baseline JSON path for regression checks."
    )
    b.add_argument("--out-dir", default=None, help="Optional directory for summary artifacts.")
    b.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return non-zero exit code if regression status is fail.",
    )
    b.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=None,
        help="Warning degradation threshold percentage.",
    )
    b.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=None,
        help="Failure degradation threshold percentage.",
    )
    b.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory collection for benchmark profiles.",
    )
    b.add_argument(
        "--collect-reproducibility",
        action="store_true",
        help="Force reproducibility checks even outside full suite mode.",
    )
    b.add_argument(
        "--save-baseline",
        default=None,
        help="Optional path to write a baseline JSON derived from this run.",
    )
    b.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable diagnostics coverage aggregation artifacts for each benchmark profile run.",
    )
    b.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional root directory for benchmark diagnostics artifacts.",
    )

    h = sub.add_parser("hardware", help="Inspect detected hardware and profile mapping.")
    h.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Requested device (auto/cpu/cuda/mps).",
    )
    return parser


def _resolve_config_with_hardware(
    config: GeneratorConfig,
    *,
    device: str | None,
    no_hardware_aware: bool,
) -> tuple[GeneratorConfig, HardwareInfo]:
    """Apply runtime hardware detection and optional profile-based tuning."""

    if no_hardware_aware:
        config.runtime.hardware_aware = False
    hw = detect_hardware(device or config.runtime.device)
    config = apply_hardware_profile(config, hw)
    return config, hw


def _apply_missingness_cli_overrides(config: GeneratorConfig, args: argparse.Namespace) -> None:
    """Apply and validate missingness overrides from CLI arguments."""

    has_missingness_override = any(
        value is not None
        for value in (
            args.missing_rate,
            args.missing_mechanism,
            args.missing_mar_observed_fraction,
            args.missing_mar_logit_scale,
            args.missing_mnar_logit_scale,
        )
    )
    if not has_missingness_override:
        return

    if args.missing_rate is not None:
        config.dataset.missing_rate = float(args.missing_rate)
    if args.missing_mechanism is not None:
        config.dataset.missing_mechanism = args.missing_mechanism
    if args.missing_mar_observed_fraction is not None:
        config.dataset.missing_mar_observed_fraction = float(args.missing_mar_observed_fraction)
    if args.missing_mar_logit_scale is not None:
        config.dataset.missing_mar_logit_scale = float(args.missing_mar_logit_scale)
    if args.missing_mnar_logit_scale is not None:
        config.dataset.missing_mnar_logit_scale = float(args.missing_mnar_logit_scale)

    try:
        config.dataset = DatasetConfig(**asdict(config.dataset))
    except ValueError as exc:
        _raise_usage_error(str(exc))


def _run_generate(args: argparse.Namespace) -> int:
    """Execute the ``generate`` command."""

    config = GeneratorConfig.from_yaml(args.config)
    config, hw = _resolve_config_with_hardware(
        config,
        device=args.device,
        no_hardware_aware=bool(args.no_hardware_aware),
    )
    if args.curriculum is not None:
        config.curriculum_stage = (
            CURRICULUM_STAGE_AUTO
            if args.curriculum == CURRICULUM_STAGE_AUTO
            else int(args.curriculum)
        )
    _apply_missingness_cli_overrides(config, args)
    steering_requested = bool(config.steering.enabled or args.steer_meta or bool(args.meta_target))
    resolved_target_specs = _resolve_meta_target_specs(config, args.meta_target)
    if steering_requested:
        unknown_metrics = collect_unknown_target_metrics(
            config.diagnostics.meta_feature_targets,
            config.meta_feature_targets,
            supported_metrics=SUPPORTED_METRICS,
        )
        if unknown_metrics:
            unknown = ", ".join(unknown_metrics)
            supported = ", ".join(sorted(SUPPORTED_METRICS))
            _raise_usage_error(
                f"Unsupported steering target metric(s): {unknown}. Supported metrics: {supported}."
            )
        incompatible_metrics = validate_target_specs_for_task(
            task=str(config.dataset.task),
            target_specs=resolved_target_specs,
            classification_only_metrics=CLASSIFICATION_ONLY_METRICS,
        )
        if incompatible_metrics:
            metrics = ", ".join(incompatible_metrics)
            _raise_usage_error(
                "Incompatible --meta-target metrics for regression task: "
                f"{metrics}. Remove these metrics or switch task=classification."
            )
    if resolved_target_specs:
        config.meta_feature_targets = {
            metric_name: [lo, hi, weight]
            for metric_name, (lo, hi, weight) in sorted(resolved_target_specs.items())
        }
    elif args.meta_target:
        config.meta_feature_targets = {}
    if args.steer_meta or bool(args.meta_target):
        config.steering.enabled = True
    if args.diagnostics:
        config.diagnostics.enabled = True

    seed = args.seed if args.seed is not None else config.seed
    out_dir = args.out or config.output.out_dir
    diagnostics_enabled = bool(config.diagnostics.enabled)
    diagnostics_out_dir: Path | None = None
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_root = args.diagnostics_out_dir or config.diagnostics.out_dir or out_dir
        if diagnostics_root is None:
            diagnostics_root = "diagnostics_artifacts"
        diagnostics_out_dir = Path(diagnostics_root)
        diagnostics_aggregator = CoverageAggregator(
            CoverageAggregationConfig(
                include_spearman=bool(config.diagnostics.include_spearman),
                histogram_bins=int(config.diagnostics.histogram_bins),
                quantiles=coerce_quantiles(config.diagnostics.quantiles),
                underrepresented_threshold=float(config.diagnostics.underrepresented_threshold),
                max_values_per_metric=config.diagnostics.max_values_per_metric,
                target_bands=_target_specs_to_bands(resolved_target_specs),
            )
        )
    if bool(config.steering.enabled) and not resolved_target_specs:
        print(
            "Steering enabled but no valid meta-feature targets were resolved; steering is a no-op."
        )
    print(
        f"Hardware backend={hw.backend} device='{hw.device_name}' "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e} profile={hw.profile}"
    )

    stream: Iterator[Any] = generate_batch_iter(
        config,
        num_datasets=args.num_datasets,
        seed=seed,
        device=args.device,
    )
    if diagnostics_aggregator is not None:
        base_stream = stream

        def _stream_with_diagnostics() -> Iterator[Any]:
            for bundle in base_stream:
                diagnostics_aggregator.update_bundle(bundle)
                yield bundle

        stream = _stream_with_diagnostics()

    if args.no_write:
        generated = sum(1 for _ in stream)
        if diagnostics_aggregator is not None:
            assert diagnostics_out_dir is not None
            summary = diagnostics_aggregator.build_summary()
            json_path = write_coverage_summary_json(
                summary, diagnostics_out_dir / "coverage_summary.json"
            )
            md_path = write_coverage_summary_markdown(
                summary, diagnostics_out_dir / "coverage_summary.md"
            )
            print(f"Wrote diagnostics artifacts: {json_path} and {md_path}")
        print(f"Generated {generated} datasets (no-write mode).")
        return 0

    written = write_parquet_shards_stream(
        stream,
        out_dir=out_dir,
        shard_size=config.output.shard_size,
        compression=config.output.compression,
    )
    if diagnostics_aggregator is not None:
        assert diagnostics_out_dir is not None
        summary = diagnostics_aggregator.build_summary()
        json_path = write_coverage_summary_json(
            summary, diagnostics_out_dir / "coverage_summary.json"
        )
        md_path = write_coverage_summary_markdown(
            summary, diagnostics_out_dir / "coverage_summary.md"
        )
        print(f"Wrote diagnostics artifacts: {json_path} and {md_path}")
    print(f"Wrote {written} datasets to: {Path(out_dir)}")
    return 0


def _default_benchmark_config(args: argparse.Namespace) -> GeneratorConfig:
    """Load benchmark defaults from custom config, falling back to dataclass defaults."""

    if args.config:
        return GeneratorConfig.from_yaml(args.config)
    return GeneratorConfig()


def _default_benchmark_artifact_dir() -> Path:
    """Return a timestamped benchmark artifact directory path."""

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks") / "results" / timestamp


def _benchmark_artifact_dir(
    args: argparse.Namespace,
) -> Path | None:
    """Resolve optional output directory for benchmark summary artifacts."""

    if args.out_dir:
        return Path(args.out_dir)

    if args.json_out:
        return None
    return _default_benchmark_artifact_dir()


def _benchmark_diagnostics_root_dir(
    args: argparse.Namespace,
    *,
    artifact_dir: Path | None,
) -> Path | None:
    """Resolve diagnostics artifact root for benchmark profile coverage summaries."""

    if not bool(args.diagnostics):
        return None
    if args.diagnostics_out_dir:
        return Path(args.diagnostics_out_dir)
    if artifact_dir is not None:
        return artifact_dir
    return _default_benchmark_artifact_dir()


def _print_profile_result_line(result: dict[str, Any]) -> None:
    """Print one compact profile benchmark summary line."""

    diagnostics_hint = ""
    artifacts = result.get("diagnostics_artifacts")
    if isinstance(artifacts, dict):
        json_pointer = artifacts.get("json")
        if isinstance(json_pointer, str) and json_pointer:
            diagnostics_hint = f" diagnostics={json_pointer}"

    missingness_hint = ""
    guardrails = result.get("missingness_guardrails")
    if isinstance(guardrails, dict) and bool(guardrails.get("enabled")):
        missingness_hint = f" missingness={guardrails.get('status', 'pass')}"

    lineage_hint = ""
    lineage_guardrails = result.get("lineage_guardrails")
    if isinstance(lineage_guardrails, dict) and bool(lineage_guardrails.get("enabled")):
        lineage_hint = f" lineage={lineage_guardrails.get('status', 'pass')}"

    curriculum_hint = ""
    curriculum_guardrails = result.get("curriculum_guardrails")
    if isinstance(curriculum_guardrails, dict) and bool(curriculum_guardrails.get("enabled")):
        curriculum_hint = f" curriculum={curriculum_guardrails.get('status', 'pass')}"

    print(
        f"[{result.get('profile_key')}] device={result.get('device')} "
        f"backend={result.get('hardware_backend')} "
        f"datasets/min={float(result.get('datasets_per_minute', 0.0)):.2f} "
        f"latency_p95_ms={float(result.get('latency_p95_ms', 0.0)):.2f}"
        f"{diagnostics_hint}{missingness_hint}{lineage_hint}{curriculum_hint}"
    )


def _run_benchmark(args: argparse.Namespace) -> int:
    """Execute the ``benchmark`` command."""

    artifact_dir = _benchmark_artifact_dir(args)
    diagnostics_root_dir = _benchmark_diagnostics_root_dir(args, artifact_dir=artifact_dir)

    default_cfg = _default_benchmark_config(args)
    suite = (args.suite or default_cfg.benchmark.suite).strip().lower()
    warn_pct = (
        float(args.warn_threshold_pct)
        if args.warn_threshold_pct is not None
        else float(default_cfg.benchmark.warn_threshold_pct)
    )
    fail_pct = (
        float(args.fail_threshold_pct)
        if args.fail_threshold_pct is not None
        else float(default_cfg.benchmark.fail_threshold_pct)
    )

    profile_specs = resolve_profile_run_specs(
        profile_keys=args.profile,
        config_path=args.config,
    )
    if args.device and len(profile_specs) == 1:
        profile_specs[0].device = args.device

    baseline_payload = load_baseline(args.baseline) if args.baseline else None

    summary = run_benchmark_suite(
        profile_specs,
        suite=suite,
        warn_threshold_pct=warn_pct,
        fail_threshold_pct=fail_pct,
        baseline_payload=baseline_payload,
        num_datasets_override=args.num_datasets,
        warmup_override=args.warmup,
        collect_memory=not bool(args.no_memory),
        collect_reproducibility=(
            bool(args.collect_reproducibility)
            or bool(default_cfg.benchmark.collect_reproducibility)
        ),
        collect_diagnostics=bool(args.diagnostics),
        diagnostics_root_dir=diagnostics_root_dir,
        fail_on_regression=bool(args.fail_on_regression),
        no_hardware_aware=bool(args.no_hardware_aware),
    )

    for result in summary.get("profile_results", []):
        _print_profile_result_line(result)

    regression = summary.get("regression", {})
    print(
        f"Regression status={regression.get('status', 'pass')} issues={len(regression.get('issues', []))}"
    )

    if artifact_dir is not None:
        json_path = write_suite_json(summary, artifact_dir / "summary.json")
        md_path = write_suite_markdown(summary, artifact_dir / "summary.md")
        print(f"Wrote benchmark artifacts: {json_path} and {md_path}")

    if args.json_out:
        path = write_suite_json(summary, args.json_out)
        print(f"Wrote benchmark JSON: {path}")

    if args.save_baseline:
        payload = build_baseline_payload(summary)
        baseline_path = write_baseline(payload, args.save_baseline)
        print(f"Wrote benchmark baseline: {baseline_path}")

    hard_fail = bool(regression.get("hard_fail"))
    return 1 if hard_fail else 0


def _run_hardware(args: argparse.Namespace) -> int:
    """Execute the ``hardware`` command."""

    hw = detect_hardware(args.device)
    print(
        f"backend={hw.backend} device='{hw.device_name}' profile={hw.profile} "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "hardware":
        return _run_hardware(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
