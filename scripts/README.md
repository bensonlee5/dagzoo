# Generation Scripts

These wrappers call `uv run cauchy-gen ...` from the repo root.

## Scripts

- `scripts/generate-from-config.sh [config] [num_datasets] [device] [out_dir] [seed]`
  - Generic generator wrapper for any config.
- `scripts/generate-default.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/default.yaml`.
- `scripts/generate-h100.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/preset_cuda_h100.yaml`.
- `scripts/generate-smoke.sh [config] [num_datasets] [device]`
  - Runs quick in-memory generation with `--no-write`.
- `scripts/fetch-additional-references.sh`
  - Downloads the additional arXiv papers listed in `reference/ADDITIONAL_PAPERS.md`.
- `scripts/benchmark-suite.sh [suite] [profile] [out_dir]`
  - Runs `cauchy-gen benchmark` with suite/profile selection.
- `scripts/benchmark-smoke.sh [profile]`
  - Quick smoke benchmark for a single profile.
- `scripts/bump-version.sh <major|minor|patch> [--dry-run] [--tag]`
  - Bump the semver version in `pyproject.toml`. Use `--tag` to commit and create a git tag.

## Examples

```bash
./scripts/generate-default.sh
./scripts/generate-default.sh 50 cpu data/run_cpu_50
./scripts/generate-h100.sh 500 cuda data/run_h100_500 123
./scripts/generate-from-config.sh configs/benchmark_medium_cuda.yaml 100 cuda data/run_medium 42
./scripts/generate-smoke.sh configs/default.yaml 3 cpu
./scripts/fetch-additional-references.sh
./scripts/benchmark-smoke.sh cpu
./scripts/benchmark-suite.sh standard all benchmarks/results/latest
./scripts/bump-version.sh patch --dry-run
./scripts/bump-version.sh minor --tag
```

`benchmark-suite.sh` with profile `all` includes CUDA profiles and will hard-fail if CUDA is unavailable.
