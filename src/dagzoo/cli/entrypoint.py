"""CLI main entrypoint."""

from __future__ import annotations

from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
