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
Generation does not run inline filtering; keep `filter.enabled: false` for
generate flows.

______________________________________________________________________

## 2. Deferred filtering (`dagzoo filter`)

Run acceptance filtering as a separate CPU stage over persisted shards:

```bash
dagzoo filter --in data/run1 --out data/run1_filter
dagzoo filter --in data/run1 --out data/run1_filter --curated-out data/run1_curated
```

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

## 5. Fixed-layout batch generation (Python API)

Use a fixed layout plan when you want many datasets with consistent structure
and aligned emitted columns across the batch.

```python
from dagzoo import (
    GeneratorConfig,
    generate_batch_fixed_layout,
    sample_fixed_layout,
)

cfg = GeneratorConfig.from_yaml("configs/default.yaml")
plan = sample_fixed_layout(cfg, seed=7, device="cpu")
batch = generate_batch_fixed_layout(cfg, plan=plan, num_datasets=32, seed=101)
```

`generate_batch_fixed_layout(_iter)` validates plan/config compatibility before
generation. If layout-driving config fields drift from the plan snapshot, it
raises and asks you to resample the plan.

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
- System guide and terminology: [how-it-works.html](how-it-works.html)
- Architecture rationale and evolution policy: [development/design-decisions.md](development/design-decisions.md)
