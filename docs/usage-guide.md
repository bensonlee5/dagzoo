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
uv tool install dagsynth
```

______________________________________________________________________

## 1. Basic generation

Use this when you want a default high-quality batch.

```bash
dagsynth generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

Each generate run writes `effective_config.yaml` and `effective_config_trace.yaml`
under the resolved output directory.

______________________________________________________________________

## 2. Diagnostics

Use diagnostics to emit per-dataset observability artifacts.

```bash
dagsynth generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --out data/run_diag
```

Detailed guides:

- [Diagnostics](features/diagnostics.md)

______________________________________________________________________

## 3. Fixed-layout batch generation (Python API)

Use a fixed layout plan when you want many datasets with consistent structure
and aligned emitted columns across the batch.

```python
from dagsynth import (
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

## 4. Missingness workflows

Use missingness workflows for MCAR/MAR/MNAR robustness regimes:

```bash
dagsynth generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
```

Detailed guide: [Missingness](features/missingness.md)

______________________________________________________________________

## 5. Many-class workflows

Use many-class workflows to exercise the rollout envelope (`n_classes_max <= 32`).

```bash
dagsynth generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 25 --out data/run_many_class_smoke

dagsynth benchmark \
  --config configs/preset_many_class_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_many_class
```

Detailed guide: [Many-class](features/many-class.md)

______________________________________________________________________

## 6. Shift workflows

Use shift profiles for controlled graph/mechanism/noise drift:

```bash
dagsynth generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

Detailed guide: [Shift / Drift](features/shift.md)

______________________________________________________________________

## 7. Noise workflows

Use noise workflows for explicit Gaussian/Laplace/Student-t/mixture regimes:

```bash
dagsynth generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --out data/run_noise_mixture
```

Detailed guide: [Noise Diversification](features/noise.md)

______________________________________________________________________

## 8. Benchmark workflows and guardrails

Use benchmark workflows for smoke checks, feature guardrails, and regression
gating.

```bash
dagsynth benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
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
- Config precedence and trace artifacts: [config-resolution.md](config-resolution.md)
- System guide and terminology: [how-it-works.md](how-it-works.md)
- Architecture rationale and evolution policy: [design-decisions.md](design-decisions.md)
