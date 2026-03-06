#!/usr/bin/env python3
"""Dry-run or remove ignored local artifact trees.

Targets are limited to known ignored runtime/docs outputs under the repo root.
Use ``--apply`` to actually delete them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET_GROUPS: dict[str, tuple[Path, ...]] = {
    "runtime": (
        REPO_ROOT / "data",
        REPO_ROOT / "benchmarks" / "results",
        REPO_ROOT / "effective_config_artifacts",
    ),
    "docs": (
        REPO_ROOT / "public",
        REPO_ROOT / "site" / "public",
        REPO_ROOT / "site" / ".generated",
    ),
}


def _iter_targets(group: str) -> list[Path]:
    if group == "all":
        paths = TARGET_GROUPS["runtime"] + TARGET_GROUPS["docs"]
    else:
        paths = TARGET_GROUPS[group]
    return list(paths)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("runtime", "docs", "all"),
        default="all",
        help="Artifact group to inspect or remove.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the listed paths. Default is dry-run only.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    targets = _iter_targets(str(args.group))

    found = 0
    for path in targets:
        rel = _relpath(path, REPO_ROOT)
        if not path.exists():
            print(f"Skip missing: {rel}")
            continue
        found += 1
        if args.apply:
            _remove_path(path)
            print(f"Removed: {rel}")
        else:
            print(f"Would remove: {rel}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to remove the listed paths.")
    elif found == 0:
        print("Nothing to remove.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
