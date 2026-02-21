"""Public package exports."""

from .config import GeneratorConfig
from .core.dataset import generate_batch, generate_batch_iter, generate_one
from .hardware import get_peak_flops
from .types import DatasetBundle

__all__ = [
    "DatasetBundle",
    "GeneratorConfig",
    "generate_batch",
    "generate_batch_iter",
    "generate_one",
    "get_peak_flops",
]
