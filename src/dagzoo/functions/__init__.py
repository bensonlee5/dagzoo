"""Random function families and combiners."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["apply_multi_function", "apply_random_function", "apply_random_activation"]


def __getattr__(name: str) -> Any:
    if name == "apply_multi_function":
        return import_module("dagzoo.functions.multi").apply_multi_function
    if name == "apply_random_function":
        return import_module("dagzoo.functions.random_functions").apply_random_function
    if name == "apply_random_activation":
        return import_module("dagzoo.functions.activations").apply_random_activation
    raise AttributeError(name)
