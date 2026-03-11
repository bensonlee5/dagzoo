"""Generate command handler."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dagzoo.core.config_resolution import (
    append_config_diff_events,
    resolve_generate_config,
    serialize_resolution_events,
)
from dagzoo.core.fixed_layout_runtime import realize_generation_config_for_run
from dagzoo.diagnostics import (
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream

from ..common import get_cli_public_api, load_config_or_usage_error, raise_usage_error
from ..effective_config import (
    print_effective_config,
    print_resolution_trace,
    write_effective_config,
    write_effective_config_trace,
)


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


def run_generate_command(args: argparse.Namespace) -> int:
    """Execute the ``generate`` command."""

    cli_api = get_cli_public_api()
    config = load_config_or_usage_error(args.config)
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
        raise_usage_error(str(exc))

    seed = args.seed if args.seed is not None else resolved.config.seed
    config, run_seed, requested_device, _resolved_device = realize_generation_config_for_run(
        resolved.config,
        seed=seed,
        device=resolved.requested_device,
    )
    if bool(config.filter.enabled):
        raise_usage_error(
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
        print_effective_config(config, header="Effective config:")

    effective_config_path = write_effective_config(
        config,
        Path(effective_config_root) / "effective_config.yaml",
    )
    trace_path = write_effective_config_trace(
        trace_payload,
        Path(effective_config_root) / "effective_config_trace.yaml",
    )
    print(f"Wrote effective config: {effective_config_path}")
    print(f"Wrote effective config trace: {trace_path}")
    if args.print_resolution_trace:
        print_resolution_trace(trace_payload, header="Resolution trace:")

    diagnostics_enabled = bool(config.diagnostics.enabled)
    diagnostics_out_dir: Path | None = None
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_root = args.diagnostics_out_dir or config.diagnostics.out_dir or out_dir
        if diagnostics_root is None:
            diagnostics_root = "diagnostics_artifacts"
        diagnostics_out_dir = Path(diagnostics_root)
        diagnostics_aggregator = cli_api.CoverageAggregator(
            build_diagnostics_aggregation_config(config.diagnostics)
        )
    print(
        f"Hardware backend={hw.backend} device='{hw.device_name}' "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e} tier={hw.tier} "
        f"hardware_policy={args.hardware_policy}"
    )

    stream: Iterator[Any] = cli_api.generate_batch_iter(
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
