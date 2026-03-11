"""Shared CLI helpers."""

from __future__ import annotations

import sys
from types import ModuleType

from dagzoo.config import GeneratorConfig


def raise_usage_error(message: str) -> None:
    """Exit with argparse-compatible usage error semantics."""

    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_config_or_usage_error(path: str) -> GeneratorConfig:
    """Load one config file or raise a CLI usage error with its parse message."""

    try:
        return GeneratorConfig.from_yaml(path)
    except (TypeError, ValueError) as exc:
        raise_usage_error(str(exc))
    raise AssertionError("unreachable")


def get_cli_public_api() -> ModuleType:
    """Return the root ``dagzoo.cli`` package for compatibility lookups."""

    root_name = __name__.rsplit(".", 1)[0]
    root_module = sys.modules.get(root_name)
    if root_module is None:
        raise RuntimeError(f"CLI root module {root_name!r} is not loaded")
    return root_module
