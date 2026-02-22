# Repository Guidelines

## Project Structure & Module Organization
Keep implementation code in `src/cauchy_generator/`, organized by function:
- `sampling/`, `graph/`, `functions/`, `converters/`, `linalg/`, `postprocess/`, `filtering/`, `io/`, `bench/`
- `cli.py` for command-line entrypoints
- `config.py` and `rng.py` for typed configuration and seeded randomness

Store tests in `tests/` with the same package layout as `src/`. Keep benchmark profiles/results in `benchmarks/`, static configs in `configs/`, and paper inputs in `reference/` (read-only).

## Build, Test, and Development Commands
- `uv sync --group dev`: create/update local environment with dev tooling.
- `uv run pytest -q`: run full test suite.
- `uv run pytest tests -q`: run focused tests.
- `uv run ruff check src tests`: lint code.
- `uv run ruff format src tests`: apply formatting.
- `uv run mypy src`: run static type checks.
- `uv run cauchy-gen generate --config configs/default.yaml --out data/run1`: generate datasets.
- `uv run cauchy-gen benchmark --config configs/benchmark_medium_cuda.yaml --device cuda`: run throughput benchmark.

## Coding Style & Naming Conventions
Use Python 3.13+, 4-space indentation, and explicit type hints for all public interfaces.  
Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.  
Prefer small, pure functions in sampling/math code; isolate side effects (I/O, logging, timing) in boundary modules.

## Testing Guidelines
Use `pytest` with file names `test_<module>.py` and test names `test_<behavior>`.  
Target at least `85%` line coverage for `src/cauchy_generator` excluding CLI-only wrappers and benchmark scripts.  
Every new generator component must include:
- shape/range invariants
- seed reproducibility checks
- integration coverage for end-to-end dataset generation

## Commit & Pull Request Guidelines
Adopt Conventional Commits (for example: `feat: add random cauchy graph sampler`).  
PRs must include:
- concise summary of behavior changes
- linked issue/task (if available)
- test/lint/type-check evidence
- benchmark deltas for performance-sensitive code

## Versioning & Release Workflow
The project follows semantic versioning. The single source of truth is `pyproject.toml`
(`version = "X.Y.Z"`). Do not add `__version__` to Python source files.

Bump the version:
- `scripts/bump-version.sh patch` — e.g. 0.1.2 → 0.1.3
- `scripts/bump-version.sh minor` — e.g. 0.1.2 → 0.2.0
- `scripts/bump-version.sh major` — e.g. 0.1.2 → 1.0.0
- Add `--dry-run` to preview, `--tag` to commit and create a git tag.

Release:
1. Ensure all tests pass on `main`.
2. Run `scripts/bump-version.sh <type> --tag`.
3. Push with `git push origin main --tags`.
4. CI builds artifacts and creates the GitHub Release automatically.

Release notes are auto-generated from commit messages by CI.

## Architecture & Performance Notes
Primary runtime is PyTorch (CPU/CUDA/MPS) with NumPy used only where sklearn-backed postprocessing/filtering requires arrays. Keep hot paths vectorized and batch-oriented. Preserve Appendix E (`E.2`-`E.14`) behavior as the source of truth; use other papers in `reference/` only to clarify ambiguous details.
