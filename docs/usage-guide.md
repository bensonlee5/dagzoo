# Usage Guide

Operational entrypoint for end users running generation and benchmarking
workflows. This page stays concise and links to detailed feature guides.

______________________________________________________________________

## Prerequisites

Install `cauchy-generator`:

```bash
uv tool install cauchy-generator
```

For development checkouts, use:

```bash
uv sync --group dev
source .venv/bin/activate
```

______________________________________________________________________

## 1. Basic generation

Use this when you want a default high-quality batch.

```bash
cauchy-gen generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

______________________________________________________________________

## 2. Diagnostics and meta steering

Use diagnostics to emit per-dataset observability artifacts. Add steering when
coverage should favor specific meta-feature target bands.

```bash
cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --steer-meta \
  --meta-target linearity_proxy=0.25:0.75:1.5 \
  --out data/run_steer
```

Detailed guides:

- [Diagnostics](features/diagnostics.md)
- [Steering](features/steering.md)

______________________________________________________________________

## 3. Missingness workflows

Use missingness workflows for MCAR/MAR/MNAR robustness regimes:

```bash
cauchy-gen generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
```

Detailed guide: [Missingness](features/missingness.md)

______________________________________________________________________

## 4. Curriculum workflows

Use curriculum workflows for staged complexity progression.

```bash
cauchy-gen generate --config configs/preset_curriculum_auto_staged.yaml --num-datasets 25 --out data/run_curriculum_auto
```

Detailed guide: [Curriculum](features/curriculum.md)

______________________________________________________________________

## 5. Many-class workflows

Use many-class workflows to exercise the rollout envelope (`n_classes_max <= 32`).

```bash
cauchy-gen generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 25 --out data/run_many_class_smoke

cauchy-gen benchmark \
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
cauchy-gen generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

Detailed guide: [Shift / Drift](features/shift.md)

______________________________________________________________________

## 7. Benchmark workflows and guardrails

Use benchmark workflows for smoke checks, feature guardrails, and regression
gating.

```bash
cauchy-gen benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
```

Detailed guide: [Benchmark Workflows and Guardrails](features/benchmark-guardrails.md)

______________________________________________________________________

## Related documents

- Feature deep dives:
  [diagnostics](features/diagnostics.md),
  [steering](features/steering.md),
  [missingness](features/missingness.md),
  [curriculum](features/curriculum.md),
  [many-class](features/many-class.md),
  [shift](features/shift.md),
  [benchmark guardrails](features/benchmark-guardrails.md)
- Output contract: [output-format.md](output-format.md)
- System guide and terminology: [how-it-works.md](how-it-works.md)
- Architecture rationale and evolution policy: [design-decisions.md](design-decisions.md)
