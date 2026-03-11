"""Config cloning and I/O helpers."""

from __future__ import annotations

import copy
from importlib.resources import files
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .models import GeneratorConfig


def clone_generator_config(
    config: "GeneratorConfig",
    *,
    revalidate: bool,
) -> "GeneratorConfig":
    """Clone generator config state with explicit validation semantics."""

    if revalidate:
        from .models import GeneratorConfig

        return GeneratorConfig.from_dict(config.to_dict())
    return copy.deepcopy(config)


def load_packaged_generator_config(resource_name: str) -> "GeneratorConfig":
    """Load one packaged YAML config resource shipped inside the wheel."""

    from .models import GeneratorConfig

    resource = files("dagzoo.config.resources").joinpath(resource_name)
    loaded = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Packaged config resource {resource_name!r} must be a mapping.")
    return GeneratorConfig.from_dict(loaded)
