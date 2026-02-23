# Config Agent Notes

## Purpose

YAML files here define runtime and benchmark presets.

## Rules

- Keep keys aligned with `GeneratorConfig` dataclasses (`src/cauchy_generator/config.py`).
- Use explicit device values: `auto`, `cpu`, `cuda`, `mps`.
- Benchmark presets should match supported profile keys (`cpu`, `cuda_desktop`, `cuda_h100`).
- `custom` is a runtime benchmark profile selection (`--profile custom` with `--config`), not a preset key family to add under `benchmark.profiles`.
- Keep `curriculum_stage` values within supported options: `off`, `auto`, `1`, `2`, `3`.
- Avoid adding keys that are ignored by the loader.

## When Editing

- Verify loading with `GeneratorConfig.from_yaml(...)`.
- Update `tests/test_config.py` when adding/changing presets.
