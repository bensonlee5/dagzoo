# Source Tree Agent Notes

## Scope

This directory contains the Python package source (`src/cauchy_generator`).

## Expectations

- Keep runtime behavior deterministic for fixed seeds.
- Preserve torch-first execution across CPU/CUDA/MPS.
- Prefer small composable functions over cross-module side effects.
- Avoid introducing new global state.

## Before You Submit

- Run: `uv run ruff check src tests`
- Run: `uv run mypy src`
- Run: `uv run pytest -q`
- If public APIs changed, update root `README.md`.
