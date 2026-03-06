#!/usr/bin/env python3
"""Convenience wrapper for `dagzoo diversity-audit`."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dagzoo.cli import main as cli_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return cli_main(["diversity-audit", *args])


if __name__ == "__main__":
    raise SystemExit(main())
