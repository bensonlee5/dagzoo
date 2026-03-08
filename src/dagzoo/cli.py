"""CLI entrypoints."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from dagzoo.bench.baseline import (
    build_baseline_payload,
    load_baseline,
    write_baseline,
)
from dagzoo.bench.report import write_suite_json, write_suite_markdown
from dagzoo.bench.suite import resolve_preset_run_specs, run_benchmark_suite
from dagzoo.config import (
    GeneratorConfig,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    normalize_missing_mechanism,
)
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core.config_resolution import (
    append_config_diff_events,
    resolve_generate_config,
    serialize_resolution_events,
)
from dagzoo.core.fixed_layout import realize_generation_config_for_run
from dagzoo.diagnostics import (
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics.effective_diversity import (
    AuditThresholds,
    build_effective_diversity_baseline_payload,
    load_effective_diversity_baseline_payload,
    run_effective_diversity_audit,
    write_effective_diversity_baseline_payload,
    write_effective_diversity_run_artifacts,
)
from dagzoo.hardware import detect_hardware
from dagzoo.hardware_policy import list_hardware_policies
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.diagnostics_targets import (
    build_diagnostics_aggregation_config,
)
from dagzoo.filtering import run_deferred_filter
from dagzoo.rng import SEED32_MAX, SEED32_MIN

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
HARDWARE_POLICY_CHOICES = list_hardware_policies()
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


def _seed_32bit_int(value: str) -> int:
    """argparse type: parse an integer seed in the unsigned 32-bit range."""

    parsed = int(value)
    if parsed < SEED32_MIN or parsed > SEED32_MAX:
        raise argparse.ArgumentTypeError(
            f"Expected a seed in [{SEED32_MIN}, {SEED32_MAX}], got {value}."
        )
    return parsed


def _filter_n_jobs(value: str) -> int:
    """argparse type: parse filter worker count (-1 or >= 1)."""

    parsed = int(value)
    if parsed == -1 or parsed >= 1:
        return parsed
    raise argparse.ArgumentTypeError(f"Expected -1 or an integer >= 1 for --n-jobs, got {value}.")


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


def _raise_usage_error(message: str) -> None:
    """Exit with argparse-compatible usage error semantics."""

    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser and register all subcommands/options."""

    parser = argparse.ArgumentParser(prog="dagzoo")
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
    g.add_argument(
        "--seed",
        type=_seed_32bit_int,
        default=None,
        help=f"Optional override for run seed in [{SEED32_MIN}, {SEED32_MAX}].",
    )
    g.add_argument(
        "--rows",
        default=None,
        help=(
            "Optional total-row spec override for generation. "
            "Supports fixed int (e.g. 1024), range (e.g. 400..60000), "
            "or CSV choices (e.g. 1024,2048,4096)."
        ),
    )
    g.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Device override (auto/cpu/cuda/mps).",
    )
    g.add_argument(
        "--hardware-policy",
        default="none",
        choices=HARDWARE_POLICY_CHOICES,
        help="Explicit hardware policy to apply to config (default: none).",
    )
    g.add_argument(
        "--no-dataset-write",
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
        "--print-effective-config",
        action="store_true",
        help="Print resolved effective config YAML before generation.",
    )
    g.add_argument(
        "--print-resolution-trace",
        action="store_true",
        help="Print field-level override trace for resolved config before generation.",
    )
    f = sub.add_parser(
        "filter",
        help="Run deferred CPU filtering on existing shard outputs.",
    )
    f.add_argument(
        "--in",
        dest="in_dir",
        required=True,
        help="Input directory containing shard_* outputs (or a single shard directory).",
    )
    f.add_argument(
        "--out",
        required=True,
        help="Directory for deferred filter manifest/summary artifacts.",
    )
    f.add_argument(
        "--curated-out",
        default=None,
        help="Optional output directory for accepted-only curated shards.",
    )
    f.add_argument(
        "--config",
        default=None,
        help="Optional fallback YAML config when shard metadata lacks embedded filter settings.",
    )
    f.add_argument(
        "--n-jobs",
        type=_filter_n_jobs,
        default=None,
        help="Optional override for ExtraTrees worker count (-1 or >= 1).",
    )
    b = sub.add_parser("benchmark", help="Run benchmark suite across one or more presets.")
    b.add_argument("--config", default=None, help="Optional YAML config for preset 'custom'.")
    b.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Device override for custom preset.",
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
        "--hardware-policy",
        default="none",
        choices=HARDWARE_POLICY_CHOICES,
        help="Explicit hardware policy to apply to configs (default: none).",
    )
    b.add_argument("--json-out", default=None, help="Optional path to write suite summary JSON.")
    b.add_argument(
        "--suite",
        default=None,
        choices=["smoke", "standard", "full"],
        help="Benchmark suite level. Defaults to config benchmark.suite.",
    )
    b.add_argument(
        "--preset",
        action="append",
        default=None,
        choices=["all", "cpu", "cuda_desktop", "cuda_h100", "custom"],
        help="Benchmark preset key. Repeat to run multiple presets.",
    )
    b.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline JSON path for regression checks.",
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
        help="Disable memory collection for benchmark presets.",
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
        help="Enable diagnostics coverage aggregation artifacts for each benchmark preset run.",
    )
    b.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional root directory for benchmark diagnostics artifacts.",
    )
    b.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print each preset's resolved effective config YAML before execution.",
    )
    b.add_argument(
        "--print-resolution-trace",
        action="store_true",
        help="Print each preset's field-level override trace before execution.",
    )

    d = sub.add_parser(
        "diversity-audit",
        help="Audit local equivalence + dataset-scale effective diversity overlap impact.",
    )
    d.add_argument(
        "--config",
        default=None,
        help="Optional generator config for scale phase (defaults to in-code defaults).",
    )
    d.add_argument(
        "--phase",
        choices=["both", "local", "scale"],
        default="both",
        help="Audit phase to run.",
    )
    d.add_argument(
        "--suite",
        choices=["smoke", "standard", "full"],
        default="standard",
        help="Scale-phase run size when --num-datasets-per-arm is omitted.",
    )
    d.add_argument(
        "--arm-set",
        choices=["high_confidence", "all_claims"],
        default="high_confidence",
        help="Ablation arm set for the scale phase.",
    )
    d.add_argument(
        "--num-datasets-per-arm",
        type=_positive_int,
        default=None,
        help="Optional override for datasets generated per scale arm.",
    )
    d.add_argument(
        "--seed",
        type=_seed_32bit_int,
        default=2_026_0304,
        help=f"Base audit seed in [{SEED32_MIN}, {SEED32_MAX}].",
    )
    d.add_argument(
        "--n-seeds",
        type=_positive_int,
        default=8,
        help="Number of local-overlap seed runs to aggregate.",
    )
    d.add_argument(
        "--n-rows",
        type=_positive_int,
        default=2_048,
        help="Rows per local-overlap run.",
    )
    d.add_argument(
        "--n-cols",
        type=_positive_int,
        default=16,
        help="Columns per local-overlap run.",
    )
    d.add_argument(
        "--out-dim",
        type=_positive_int,
        default=16,
        help="Output width for family-overlap probes.",
    )
    d.add_argument(
        "--nn-degenerate-trials",
        type=_positive_int,
        default=50_000,
        help="Trials for estimating nn->linear degenerate probability.",
    )
    d.add_argument(
        "--exact-affine-rmse",
        type=float,
        default=1e-6,
        help="Threshold for exact affine equivalence label.",
    )
    d.add_argument(
        "--near-cosine",
        type=float,
        default=0.95,
        help="Cosine threshold for near-equivalent label.",
    )
    d.add_argument(
        "--near-affine-rmse",
        type=float,
        default=0.20,
        help="Affine RMSE threshold for near-equivalent label.",
    )
    d.add_argument(
        "--meaningful-threshold-pct",
        type=float,
        default=5.0,
        help="Composite shift threshold for a meaningful scale-impact change.",
    )
    d.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline JSON path for diversity regression checks.",
    )
    d.add_argument(
        "--save-baseline",
        default=None,
        help="Optional path to write a diversity baseline JSON from this run.",
    )
    d.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=2.5,
        help="Warn threshold for composite-shift delta vs baseline.",
    )
    d.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=5.0,
        help="Fail threshold for composite-shift delta vs baseline.",
    )
    d.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return non-zero exit code if diversity regression status is fail.",
    )
    d.add_argument(
        "--out-dir",
        default=str(Path("effective_config_artifacts") / "effective_diversity"),
        help="Output directory for diversity audit artifacts.",
    )
    d.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Optional device override for scale-phase generation.",
    )

    h = sub.add_parser("hardware", help="Inspect detected hardware and tier mapping.")
    h.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Requested device (auto/cpu/cuda/mps).",
    )
    return parser


def _effective_config_yaml(config: GeneratorConfig) -> str:
    """Render an effective config payload as YAML text."""

    return yaml.safe_dump(
        config.to_dict(),
        sort_keys=False,
        default_flow_style=False,
    )


def _write_effective_config(config: GeneratorConfig, path: Path) -> Path:
    """Persist effective config YAML to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_effective_config_yaml(config), encoding="utf-8")
    return path


def _effective_resolution_trace_yaml(trace_payload: list[dict[str, Any]]) -> str:
    """Render a field-level config resolution trace as YAML text."""

    return yaml.safe_dump(
        trace_payload,
        sort_keys=False,
        default_flow_style=False,
    )


def _write_effective_config_trace(trace_payload: list[dict[str, Any]], path: Path) -> Path:
    """Persist effective config resolution trace YAML to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_effective_resolution_trace_yaml(trace_payload), encoding="utf-8")
    return path


def _print_effective_config(config: GeneratorConfig, *, header: str) -> None:
    """Print effective config YAML to stdout with a short header."""

    print(header)
    print(_effective_config_yaml(config).rstrip())


def _print_resolution_trace(trace_payload: list[dict[str, Any]], *, header: str) -> None:
    """Print config resolution trace YAML to stdout with a short header."""

    print(header)
    print(_effective_resolution_trace_yaml(trace_payload).rstrip())


def _write_generate_diagnostics_artifacts(
    diagnostics_aggregator: CoverageAggregator,
    *,
    diagnostics_out_dir: Path,
) -> None:
    """Write generation diagnostics coverage artifacts and print output paths."""

    summary = diagnostics_aggregator.build_summary()
    json_path = write_coverage_summary_json(summary, diagnostics_out_dir / "coverage_summary.json")
    md_path = write_coverage_summary_markdown(summary, diagnostics_out_dir / "coverage_summary.md")
    print(f"Wrote diagnostics artifacts: {json_path} and {md_path}")


def _load_config_or_usage_error(path: str) -> GeneratorConfig:
    """Load one config file or raise a CLI usage error with its parse message."""

    try:
        return GeneratorConfig.from_yaml(path)
    except (TypeError, ValueError) as exc:
        _raise_usage_error(str(exc))
    raise AssertionError("unreachable")


def _run_generate(args: argparse.Namespace) -> int:
    """Execute the ``generate`` command."""

    config = _load_config_or_usage_error(args.config)
    try:
        resolved = resolve_generate_config(
            config,
            device_override=args.device,
            rows=args.rows,
            hardware_policy=str(args.hardware_policy),
            missing_rate=args.missing_rate,
            missing_mechanism=args.missing_mechanism,
            missing_mar_observed_fraction=args.missing_mar_observed_fraction,
            missing_mar_logit_scale=args.missing_mar_logit_scale,
            missing_mnar_logit_scale=args.missing_mnar_logit_scale,
            diagnostics_enabled=bool(args.diagnostics),
        )
    except ValueError as exc:
        _raise_usage_error(str(exc))

    seed = args.seed if args.seed is not None else resolved.config.seed
    config, run_seed, requested_device, _resolved_device = realize_generation_config_for_run(
        resolved.config,
        seed=seed,
        device=resolved.requested_device,
    )
    if bool(config.filter.enabled):
        _raise_usage_error(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation."
        )
    hw = resolved.hardware
    trace_events = list(resolved.trace_events)
    append_config_diff_events(
        resolved.config,
        config,
        source="generate.run_realization",
        events=trace_events,
    )
    trace_payload = serialize_resolution_events(trace_events)
    out_dir = args.out or config.output.out_dir
    effective_config_root = out_dir
    if effective_config_root is None:
        effective_config_root = (
            args.diagnostics_out_dir or config.diagnostics.out_dir or "effective_config_artifacts"
        )
    if args.print_effective_config:
        _print_effective_config(config, header="Effective config:")

    effective_config_path = _write_effective_config(
        config,
        Path(effective_config_root) / "effective_config.yaml",
    )
    trace_path = _write_effective_config_trace(
        trace_payload,
        Path(effective_config_root) / "effective_config_trace.yaml",
    )
    print(f"Wrote effective config: {effective_config_path}")
    print(f"Wrote effective config trace: {trace_path}")
    if args.print_resolution_trace:
        _print_resolution_trace(trace_payload, header="Resolution trace:")

    diagnostics_enabled = bool(config.diagnostics.enabled)
    diagnostics_out_dir: Path | None = None
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_root = args.diagnostics_out_dir or config.diagnostics.out_dir or out_dir
        if diagnostics_root is None:
            diagnostics_root = "diagnostics_artifacts"
        diagnostics_out_dir = Path(diagnostics_root)
        diagnostics_aggregator = CoverageAggregator(
            build_diagnostics_aggregation_config(config.diagnostics)
        )
    print(
        f"Hardware backend={hw.backend} device='{hw.device_name}' "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e} tier={hw.tier} "
        f"hardware_policy={args.hardware_policy}"
    )

    stream: Iterator[Any] = generate_batch_iter(
        config,
        num_datasets=args.num_datasets,
        seed=run_seed,
        device=requested_device,
    )
    if diagnostics_aggregator is not None:
        base_stream = stream

        def _stream_with_diagnostics() -> Iterator[Any]:
            for bundle in base_stream:
                diagnostics_aggregator.update_bundle(bundle)
                yield bundle

        stream = _stream_with_diagnostics()

    if args.no_dataset_write:
        generated = sum(1 for _ in stream)
        if diagnostics_aggregator is not None:
            assert diagnostics_out_dir is not None
            _write_generate_diagnostics_artifacts(
                diagnostics_aggregator, diagnostics_out_dir=diagnostics_out_dir
            )
        print(f"Generated {generated} datasets (no-dataset-write mode).")
        return 0

    written = write_packed_parquet_shards_stream(
        stream,
        out_dir=out_dir,
        shard_size=config.output.shard_size,
        compression=config.output.compression,
    )
    if diagnostics_aggregator is not None:
        assert diagnostics_out_dir is not None
        _write_generate_diagnostics_artifacts(
            diagnostics_aggregator, diagnostics_out_dir=diagnostics_out_dir
        )
    print(f"Wrote {written} datasets to: {Path(out_dir)}")
    return 0


def _run_filter(args: argparse.Namespace) -> int:
    """Execute the ``filter`` command."""

    try:
        fallback_config = _load_config_or_usage_error(args.config) if args.config else None
        result = run_deferred_filter(
            in_dir=args.in_dir,
            out_dir=args.out,
            config=fallback_config,
            curated_out_dir=args.curated_out,
            n_jobs_override=args.n_jobs,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _raise_usage_error(str(exc))
    print(f"Wrote filter manifest: {result.manifest_path}")
    print(f"Wrote filter summary: {result.summary_path}")
    print(
        "Deferred filter summary: "
        f"total={result.total_datasets} accepted={result.accepted_datasets} "
        f"rejected={result.rejected_datasets} dpm={result.datasets_per_minute:.2f}"
    )
    if result.curated_out_dir is not None:
        print(
            f"Wrote curated accepted-only shards: {result.curated_out_dir} "
            f"(datasets={result.curated_accepted_datasets})"
        )
    return 0


def _default_benchmark_config(args: argparse.Namespace) -> GeneratorConfig:
    """Load benchmark defaults from custom config, falling back to dataclass defaults."""

    if args.config:
        return _load_config_or_usage_error(args.config)
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
    """Resolve diagnostics artifact root for benchmark preset coverage summaries."""

    if not bool(args.diagnostics):
        return None
    if args.diagnostics_out_dir:
        return Path(args.diagnostics_out_dir)
    if artifact_dir is not None:
        return artifact_dir
    return _default_benchmark_artifact_dir()


def _sanitize_preset_segment(preset_key: str) -> str:
    """Normalize preset key into a filesystem-safe segment."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(preset_key)).strip("._-")
    return normalized or "preset"


def _write_benchmark_effective_configs(
    summary: dict[str, Any], artifact_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Persist per-preset effective config payloads and resolution traces."""

    preset_results = summary.get("preset_results", [])
    if not isinstance(preset_results, list):
        return [], []

    config_paths: list[Path] = []
    trace_paths: list[Path] = []
    key_counts: dict[str, int] = {}
    out_root = artifact_dir / "effective_configs"
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(preset_results):
        if not isinstance(result, dict):
            continue
        payload = result.get("effective_config")
        if not isinstance(payload, dict):
            continue
        key = _sanitize_preset_segment(str(result.get("preset_key", f"preset_{idx}")))
        key_counts[key] = key_counts.get(key, 0) + 1
        count = key_counts[key]
        suffix = f"_run{count}" if count > 1 else ""
        path = out_root / f"{key}{suffix}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        config_paths.append(path)

        trace_payload = result.get("effective_config_trace")
        if isinstance(trace_payload, list):
            trace_path = out_root / f"{key}{suffix}_trace.yaml"
            trace_path.write_text(
                yaml.safe_dump(trace_payload, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            trace_paths.append(trace_path)
    return config_paths, trace_paths


def _print_preset_result_line(result: dict[str, Any]) -> None:
    """Print one compact preset benchmark summary line."""

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

    shift_hint = ""
    shift_guardrails = result.get("shift_guardrails")
    if isinstance(shift_guardrails, dict) and bool(shift_guardrails.get("enabled")):
        shift_hint = f" shift={shift_guardrails.get('status', 'pass')}"

    noise_hint = ""
    noise_guardrails = result.get("noise_guardrails")
    if isinstance(noise_guardrails, dict) and bool(noise_guardrails.get("enabled")):
        noise_hint = f" noise={noise_guardrails.get('status', 'pass')}"

    stage_hint = (
        " "
        f"gen/min={float(result.get('generation_datasets_per_minute', 0.0)):.2f} "
        f"write/min={float(result.get('write_datasets_per_minute', 0.0)):.2f}"
    )
    filter_stage_dpm = result.get("filter_datasets_per_minute")
    filter_stage_hint = (
        f" filter/min={float(filter_stage_dpm):.2f}"
        if isinstance(filter_stage_dpm, (int, float))
        else " filter/min=-"
    )
    filter_reject_ratio = result.get("filter_rejection_rate_attempt_level")
    filter_retry_ratio = result.get("filter_retry_dataset_rate")
    filter_reject_hint = (
        f" filter_reject_attempt_pct={float(filter_reject_ratio) * 100.0:.2f}"
        if isinstance(filter_reject_ratio, (int, float))
        else " filter_reject_attempt_pct=-"
    )
    filter_retry_hint = (
        f" filter_retry_dataset_pct={float(filter_retry_ratio) * 100.0:.2f}"
        if isinstance(filter_retry_ratio, (int, float))
        else " filter_retry_dataset_pct=-"
    )
    latency_p95 = result.get("latency_p95_ms")
    latency_hint = (
        f"latency_p95_ms={float(latency_p95):.2f}"
        if isinstance(latency_p95, (int, float))
        else "latency_p95_ms=-"
    )

    print(
        f"[{result.get('preset_key')}] device={result.get('device')} "
        f"rows={result.get('dataset_rows_total', '-')} "
        f"mode={result.get('generation_mode', 'dynamic')} "
        f"backend={result.get('hardware_backend')} "
        f"datasets/min={float(result.get('datasets_per_minute', 0.0)):.2f} "
        f"{latency_hint}"
        f"{stage_hint}{filter_stage_hint}{filter_reject_hint}{filter_retry_hint}"
        f"{diagnostics_hint}{missingness_hint}{lineage_hint}{shift_hint}{noise_hint}"
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

    preset_specs = resolve_preset_run_specs(
        preset_keys=args.preset,
        config_path=args.config,
    )
    if args.device and len(preset_specs) > 1:
        _raise_usage_error(
            "benchmark --device cannot be combined with multiple --preset values; "
            "the override would be ambiguous."
        )
    effective_device_override = args.device if args.device and len(preset_specs) == 1 else None
    for spec in preset_specs:
        normalized_requested_device = (
            effective_device_override or spec.device or spec.config.runtime.device or "auto"
        ).lower()
        if effective_device_override is not None:
            spec.device = normalized_requested_device

    baseline_payload = load_baseline(args.baseline) if args.baseline else None

    summary = run_benchmark_suite(
        preset_specs,
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
        hardware_policy=str(args.hardware_policy),
    )
    if args.print_effective_config:
        for result in summary.get("preset_results", []):
            if not isinstance(result, dict):
                continue
            payload = result.get("effective_config")
            if not isinstance(payload, dict):
                continue
            preset_key = str(result.get("preset_key", "unknown"))
            print(f"Effective config [{preset_key}]:")
            print(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).rstrip())
    if args.print_resolution_trace:
        for result in summary.get("preset_results", []):
            if not isinstance(result, dict):
                continue
            trace_payload = result.get("effective_config_trace")
            if not isinstance(trace_payload, list):
                continue
            preset_key = str(result.get("preset_key", "unknown"))
            _print_resolution_trace(trace_payload, header=f"Resolution trace [{preset_key}]:")

    for result in summary.get("preset_results", []):
        _print_preset_result_line(result)

    regression = summary.get("regression", {})
    print(
        f"Regression status={regression.get('status', 'pass')} issues={len(regression.get('issues', []))}"
    )

    if artifact_dir is not None:
        json_path = write_suite_json(summary, artifact_dir / "summary.json")
        md_path = write_suite_markdown(summary, artifact_dir / "summary.md")
        effective_paths, trace_paths = _write_benchmark_effective_configs(summary, artifact_dir)
        if effective_paths or trace_paths:
            print(
                "Wrote benchmark effective configs under: "
                f"{(artifact_dir / 'effective_configs').resolve()}"
            )
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


def _run_diversity_audit(args: argparse.Namespace) -> int:
    """Execute the ``diversity-audit`` command."""

    base_config = _load_config_or_usage_error(args.config) if args.config else GeneratorConfig()
    if args.device is not None:
        base_config.runtime.device = str(args.device)

    thresholds = AuditThresholds(
        exact_affine_rmse=float(args.exact_affine_rmse),
        near_cosine=float(args.near_cosine),
        near_affine_rmse=float(args.near_affine_rmse),
    )
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
    )

    out_dir = Path(args.out_dir)
    artifact_paths = write_effective_diversity_run_artifacts(report, out_dir=out_dir)
    for key in sorted(artifact_paths):
        print(f"Wrote diversity artifact [{key}]: {artifact_paths[key]}")

    if args.save_baseline:
        scale_report = report.get("scale_report")
        if isinstance(scale_report, dict):
            baseline = build_effective_diversity_baseline_payload(scale_report)
            baseline_path = write_effective_diversity_baseline_payload(
                baseline,
                args.save_baseline,
            )
            print(f"Wrote diversity baseline: {baseline_path}")

    regression = report.get("regression")
    if isinstance(regression, dict):
        print(
            "Diversity regression status="
            f"{regression.get('status', 'pass')} issues={len(regression.get('issues', []))}"
        )

    hard_fail = bool(
        args.fail_on_regression
        and isinstance(regression, dict)
        and str(regression.get("status", "pass")) == "fail"
    )
    return 1 if hard_fail else 0


def _run_hardware(args: argparse.Namespace) -> int:
    """Execute the ``hardware`` command."""

    hw = detect_hardware(args.device)
    print(
        f"backend={hw.backend} device='{hw.device_name}' tier={hw.tier} "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "filter":
        return _run_filter(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "diversity-audit":
        return _run_diversity_audit(args)
    if args.command == "hardware":
        return _run_hardware(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
