import torch


def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded torch Generator on CPU for deterministic tests."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g
