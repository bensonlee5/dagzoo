import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded torch Generator on CPU for deterministic tests."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g
