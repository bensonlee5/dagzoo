"""CLI entrypoints."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from cauchy_generator.bench.baseline import build_baseline_payload, load_baseline, write_baseline
from cauchy_generator.bench.report import write_suite_json, write_suite_markdown
from cauchy_generator.bench.suite import resolve_profile_run_specs, run_benchmark_suite
from cauchy_generator.config import (
    CURRICULUM_STAGE_AUTO,
    CURRICULUM_STAGE_CLI_CHOICES,
    GeneratorConfig,
)
from cauchy_generator.core.dataset import generate_batch_iter
from cauchy_generator.hardware import (
    HardwareInfo,
    apply_hardware_profile,
    detect_hardware,
)
from cauchy_generator.io.parquet_writer import write_parquet_shards_stream

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


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
    seed = args.seed if args.seed is not None else config.seed
    out_dir = args.out or config.output.out_dir
    print(
        f"Hardware backend={hw.backend} device='{hw.device_name}' "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e} profile={hw.profile}"
    )

    if args.no_write:
        generated = sum(
            1
            for _ in generate_batch_iter(
                config,
                num_datasets=args.num_datasets,
                seed=seed,
                device=args.device,
            )
        )
        print(f"Generated {generated} datasets (no-write mode).")
        return 0

    written = write_parquet_shards_stream(
        generate_batch_iter(
            config,
            num_datasets=args.num_datasets,
            seed=seed,
            device=args.device,
        ),
        out_dir=out_dir,
        shard_size=config.output.shard_size,
        compression=config.output.compression,
    )
    print(f"Wrote {written} datasets to: {Path(out_dir)}")
    return 0


def _default_benchmark_config(args: argparse.Namespace) -> GeneratorConfig:
    """Load benchmark defaults from custom config, falling back to dataclass defaults."""

    if args.config:
        return GeneratorConfig.from_yaml(args.config)
    return GeneratorConfig()


def _benchmark_artifact_dir(args: argparse.Namespace) -> Path | None:
    """Resolve optional output directory for benchmark summary artifacts."""

    if args.out_dir:
        return Path(args.out_dir)

    if not args.json_out:
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        return Path("benchmarks") / "results" / timestamp
    return None


def _print_profile_result_line(result: dict[str, Any]) -> None:
    """Print one compact profile benchmark summary line."""

    print(
        f"[{result.get('profile_key')}] device={result.get('device')} "
        f"backend={result.get('hardware_backend')} "
        f"datasets/min={float(result.get('datasets_per_minute', 0.0)):.2f} "
        f"latency_p95_ms={float(result.get('latency_p95_ms', 0.0)):.2f}"
    )


def _run_benchmark(args: argparse.Namespace) -> int:
    """Execute the ``benchmark`` command."""

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
        fail_on_regression=bool(args.fail_on_regression),
        no_hardware_aware=bool(args.no_hardware_aware),
    )

    for result in summary.get("profile_results", []):
        _print_profile_result_line(result)

    regression = summary.get("regression", {})
    print(
        f"Regression status={regression.get('status', 'pass')} issues={len(regression.get('issues', []))}"
    )

    artifact_dir = _benchmark_artifact_dir(args)
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
