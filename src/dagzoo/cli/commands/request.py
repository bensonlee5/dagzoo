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
    print(f"Wrote handoff manifest: {result.handoff_manifest_path}")
    print(f"Wrote {result.generated_datasets} datasets to: {result.generated_dir}")
    return 0
