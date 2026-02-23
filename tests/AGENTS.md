# Test Agent Notes

## Scope

Pytest coverage for generation, benchmarks, hardware detection, and I/O behavior.

## Rules

- Add regression tests for any behavior change.
- Prefer targeted unit tests plus one integration-path assertion.
- Keep tests deterministic; always pass explicit seeds.
- Use strict numeric tolerances for deterministic formulas; use bounded tolerances for bootstrap/RNG-driven proxy metrics.
- For CLI validation, assert `SystemExit` code when argparse rejects input.

## Minimum Checks

- `uv run pytest -q`
- `uv run ruff check src tests`
