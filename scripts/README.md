# Generation Scripts

These wrappers run `cauchy-gen` from the repo root (typically via `uv run`).

## Scripts

- `scripts/generate-from-config.sh [config] [num_datasets] [device] [out_dir] [seed]`
  - Generic generator wrapper for any config.
- `scripts/generate-default.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/default.yaml`.
- `scripts/generate-h100.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/preset_cuda_h100.yaml`.
- `scripts/generate-many-class.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/preset_many_class_generate_smoke.yaml`.
- `scripts/generate-noise.sh [family] [num_datasets] [device] [out_dir] [seed]`
  - Runs preset-based noise family workflows (`legacy`, `gaussian`, `laplace`, `student_t`, `mixture`).
- `scripts/generate-smoke.sh [config] [num_datasets] [device]`
  - Runs quick in-memory generation with `--no-write`.
- `scripts/generate-curriculum.sh --base-config ... --out-root ... --datasets-per-stage ... --n-test ... (--train-start/--train-stop/--train-step | --train-values)`
  - Runs a curriculum as repeated `cauchy-gen generate` calls over stage row counts.
  - Stage rows are required; columns are optional (`--n-features` or `--stage-columns`).
  - `--chunk-size` controls sequential datasets per call.
- `scripts/generate-missingness.sh [mechanism] [missing_rate] [num_datasets] [device] [out_dir] [seed]`
  - Runs generation with CLI-level missingness overrides (`mcar`, `mar`, `mnar`).
- `scripts/fetch-additional-references.sh`
  - Downloads the additional arXiv papers listed in `reference/ADDITIONAL_PAPERS.md`.
- `scripts/benchmark-suite.sh [suite] [profile] [out_dir] [diagnostics] [diagnostics_out_dir]`
  - Runs `cauchy-gen benchmark` with suite/profile selection and optional diagnostics.
- `scripts/benchmark-smoke.sh [profile] [diagnostics] [diagnostics_out_dir]`
  - Quick smoke benchmark for a single profile with optional diagnostics.
- `scripts/bump-version.sh <major|minor|patch> [--dry-run] [--tag]`
  - Bump the semver version in `pyproject.toml`. Use `--tag` to commit and create a git tag.

## Examples

```bash
./scripts/generate-default.sh
./scripts/generate-default.sh 50 cpu data/run_cpu_50
./scripts/generate-h100.sh 500 cuda data/run_h100_500 123
./scripts/generate-many-class.sh 25 cpu data/run_many_class 123
./scripts/generate-noise.sh gaussian 25 cpu data/run_noise_gaussian 123
./scripts/generate-noise.sh mixture 25 cpu data/run_noise_mixture 124
./scripts/generate-from-config.sh configs/benchmark_medium_cuda.yaml 100 cuda data/run_medium 42
./scripts/generate-smoke.sh configs/default.yaml 3 cpu
./scripts/generate-curriculum.sh --base-config configs/default.yaml --out-root data/run_curriculum --datasets-per-stage 4 --n-test 256 --train-start 1024 --train-stop 1026 --train-step 1 --chunk-size 2 --device cpu
./scripts/generate-curriculum.sh --base-config configs/default.yaml --out-root data/run_curriculum_cols --datasets-per-stage 2 --n-test 128 --train-values 512,768,1024 --stage-columns 16,24,32 --no-write
./scripts/generate-missingness.sh mcar 0.2 25 cpu data/run_missing_mcar 101
./scripts/generate-missingness.sh mar 0.25 25 cpu data/run_missing_mar 102
./scripts/fetch-additional-references.sh
./scripts/benchmark-smoke.sh cpu
./scripts/benchmark-smoke.sh cpu on benchmarks/results/smoke_diag
./scripts/benchmark-suite.sh standard all benchmarks/results/latest
./scripts/benchmark-suite.sh smoke cpu benchmarks/results/smoke_cpu_diag on
uv run cauchy-gen generate --config configs/preset_diagnostics_on.yaml --num-datasets 25 --diagnostics --out data/run_diag
uv run cauchy-gen generate --config configs/preset_missingness_mnar.yaml --num-datasets 25 --out data/run_missing_mnar
uv run cauchy-gen generate --config configs/preset_noise_student_t_generate_smoke.yaml --num-datasets 25 --out data/run_noise_student_t
uv run cauchy-gen benchmark --config configs/preset_missingness_mar.yaml --profile custom --suite smoke --no-memory --out-dir benchmarks/results/smoke_missing_mar
uv run cauchy-gen benchmark --config configs/preset_noise_benchmark_smoke.yaml --profile custom --suite smoke --no-memory --out-dir benchmarks/results/smoke_noise
./scripts/bump-version.sh patch --dry-run
./scripts/bump-version.sh minor --tag
```

`benchmark-suite.sh` with profile `all` includes CUDA profiles and will hard-fail if CUDA is unavailable.

When diagnostics is enabled for benchmark scripts, coverage artifacts are written under:

- `<out_dir>/diagnostics/<sanitized_profile_key>_<hash>/coverage_summary.json`
- `<out_dir>/diagnostics/<sanitized_profile_key>_<hash>/coverage_summary.md`

The diagnostics profile directory is sanitized and hash-suffixed (for example, `cpu_ca49ca4b`) to keep paths unique and filesystem-safe.

When missingness is enabled in benchmark configs, summary JSON includes
`profile_results[*].missingness_guardrails` and may escalate regression status via runtime or acceptance issues.

When non-legacy noise is enabled in benchmark configs, summary JSON includes
`profile_results[*].noise_guardrails` and may escalate regression status via runtime or metadata validity issues.
