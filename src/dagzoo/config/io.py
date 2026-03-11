"""Config cloning and I/O helpers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

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
