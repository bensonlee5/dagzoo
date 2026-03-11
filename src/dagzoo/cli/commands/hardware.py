"""Hardware command handler."""

from __future__ import annotations

import argparse

from ..common import get_cli_public_api


def run_hardware_command(args: argparse.Namespace) -> int:
    """Execute the ``hardware`` command."""

    hw = get_cli_public_api().detect_hardware(args.device)
    print(
        f"backend={hw.backend} device='{hw.device_name}' tier={hw.tier} "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e}"
    )
    return 0
