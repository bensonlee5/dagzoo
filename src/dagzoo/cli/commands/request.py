"""Request command handler."""

from __future__ import annotations

import argparse

import yaml

from ..common import get_cli_public_api, raise_usage_error


def run_request_command(args: argparse.Namespace) -> int:
    """Execute the ``request`` command."""

    cli_api = get_cli_public_api()
    try:
        result = cli_api.run_request_execution(
            request_path=args.request,
            device_override=args.device,
            hardware_policy=str(args.hardware_policy),
            n_jobs_override=args.n_jobs,
            print_effective_config_flag=bool(args.print_effective_config),
            print_resolution_trace_flag=bool(args.print_resolution_trace),
        )
    except yaml.YAMLError as exc:
        raise_usage_error(f"Failed to parse request file {args.request}: {exc}")
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        raise_usage_error(str(exc))

    print(f"Wrote effective config: {result.effective_config_path}")
    print(f"Wrote effective config trace: {result.effective_config_trace_path}")
    print(f"Wrote {result.generated_datasets} datasets to: {result.generated_dir}")
    print(f"Wrote filter manifest: {result.filter_result.manifest_path}")
    print(f"Wrote filter summary: {result.filter_result.summary_path}")
    print(
        "Deferred filter summary: "
        f"total={result.filter_result.total_datasets} "
        f"accepted={result.filter_result.accepted_datasets} "
        f"rejected={result.filter_result.rejected_datasets} "
        f"dpm={result.filter_result.datasets_per_minute:.2f}"
    )
    if result.filter_result.curated_out_dir is not None:
        print(
            f"Wrote curated accepted-only shards: {result.filter_result.curated_out_dir} "
            f"(datasets={result.filter_result.curated_accepted_datasets})"
        )
    return 0
