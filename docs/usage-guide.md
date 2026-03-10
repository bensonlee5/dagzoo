# Usage Guide

Operational entrypoint for end users running generation and benchmarking
workflows. This page stays concise and links to detailed feature guides.

______________________________________________________________________

## Prerequisites

Examples on this page use repo-local presets under `configs/`.

For a repo checkout:

```bash
uv sync --group dev
source .venv/bin/activate
```

For a global CLI install (without repo presets/config files):

```bash
uv tool install dagzoo
```

______________________________________________________________________

## 1. Basic generation

Use this when you want a default high-quality batch.

```bash
dagzoo generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

Each generate run writes `effective_config.yaml` and `effective_config_trace.yaml`
under the resolved output directory.
`dagzoo generate` now samples one internal fixed-layout plan per run, so all
datasets emitted in the same run share one layout signature / plan signature.
Generation does not run inline filtering; keep `filter.enabled: false` for
generate flows.
Parallel generation has been removed. Config files must not include
`runtime.worker_count` or `runtime.worker_index`.

______________________________________________________________________

## 2. Deferred filtering (`dagzoo filter`)

Run acceptance filtering as a separate CPU stage over persisted shards:

```bash
dagzoo filter --in data/run1 --out data/run1_filter
dagzoo filter --in data/run1 --out data/run1_filter --curated-out data/run1_curated
```

Deferred filtering replays strictly from embedded shard metadata; current
generated outputs include the required task and filter config under
`metadata.config`.

______________________________________________________________________

## 3. Total-row control (`dataset.rows` / `--rows`)

Use `dataset.rows` (or CLI `--rows`) to control total rows with one field:

```bash
dagzoo generate --config configs/default.yaml --rows 1024 --num-datasets 10 --out data/run_rows_fixed
dagzoo generate --config configs/default.yaml --rows 400..60000 --num-datasets 25 --no-dataset-write
dagzoo generate --config configs/default.yaml --rows 1024,2048,4096 --num-datasets 25 --no-dataset-write
```

When rows mode is active, `dataset.n_test` stays fixed and `n_train` is derived as:
`n_train = total_rows - n_test`.
For canonical `generate` runs, `range` and `choices` rows modes are realized
once per run, not once per dataset.

Historical curriculum shell workflows are retired. To migrate prior train-row stages:

- old train range `A..B` with fixed `n_test=T` -> new total-row range `(A+T)..(B+T)`
- old train choices `a,b,c` with fixed `n_test=T` -> new total-row choices `(a+T),(b+T),(c+T)`

______________________________________________________________________

## 4. Diagnostics

Use diagnostics to emit per-dataset observability artifacts.

```bash
dagzoo generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --out data/run_diag
```

Detailed guides:

- [Diagnostics](features/diagnostics.md)

______________________________________________________________________

## 5. Canonical fixed-layout generation

Explicit fixed-layout plan workflows have been removed from the public CLI and
top-level Python API. Use `dagzoo generate`, `generate_one`, `generate_batch`,
or `generate_batch_iter`; those entrypoints all run on the canonical
fixed-layout engine internally.

For persisted outputs, replay canonical batch bundles with the shared run seed
plus `dataset_index` / `run_num_datasets` from the recorded metadata.

______________________________________________________________________

## 6. Missingness workflows

Use missingness workflows for MCAR/MAR/MNAR robustness regimes:

```bash
dagzoo generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
```

Detailed guide: [Missingness](features/missingness.md)

______________________________________________________________________

## 7. Many-class workflows

Use many-class workflows to exercise the rollout envelope (`n_classes_max <= 32`).

```bash
dagzoo generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 25 --out data/run_many_class_smoke

dagzoo benchmark \
  --config configs/preset_many_class_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_many_class
```

Note: the built-in CPU benchmark preset (`dagzoo benchmark --preset cpu`) now
measures three explicit row profiles: `1024`, `4096`, and `8192` total rows per
dataset. Those runs report `generation_mode="fixed_batched"` plus explicit row
counts in their summary artifacts and use the same canonical generation path as
`dagzoo generate`.

Custom/standard benchmark presets also support `dataset.rows`. For benchmark
flows, rows specs stay variable through preset config resolution and then
realize once per preset run. Smoke suites cap rows before that run
realization so smoke benchmarks stay within the intended split-size envelope.

Detailed guide: [Many-class](features/many-class.md)

______________________________________________________________________

## 8. Shift workflows

Use shift profiles for controlled graph/mechanism/noise drift:

```bash
dagzoo generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

Detailed guide: [Shift / Drift](features/shift.md)

______________________________________________________________________

## 9. Noise workflows

Use noise workflows for explicit Gaussian/Laplace/Student-t/mixture regimes:

```bash
dagzoo generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --out data/run_noise_mixture
```

Detailed guide: [Noise Diversification](features/noise.md)

______________________________________________________________________

## 10. Benchmark workflows and guardrails

Use benchmark workflows for smoke checks, feature guardrails, and regression
gating.

```bash
dagzoo benchmark --suite smoke --preset cpu --out-dir benchmarks/results/smoke_cpu
```

`--device` is a single-preset override. When you run multiple `--preset`
values in one command, set device selection in each preset/config instead of
passing a shared CLI device override.

Detailed guide: [Benchmark Workflows and Guardrails](features/benchmark-guardrails.md)

______________________________________________________________________

## Related documents

- Feature deep dives:
  [diagnostics](features/diagnostics.md),
  [missingness](features/missingness.md),
  [many-class](features/many-class.md),
  [shift](features/shift.md),
  [noise](features/noise.md),
  [benchmark guardrails](features/benchmark-guardrails.md)
- Output contract: [output-format.md](output-format.md)
- Config precedence and trace artifacts: [development/config-resolution.md](development/config-resolution.md)
- System guide and terminology: [how-it-works.md](how-it-works.md)
- Architecture rationale and evolution policy: [development/design-decisions.md](development/design-decisions.md)
