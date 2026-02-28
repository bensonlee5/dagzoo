# Usage Guide

Operational guide for end users running generation and benchmarking workflows.
This document describes the current baseline CLI surface; function-family sets
and parameterizations can expand over time with backward-compatible defaults.

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

Use diagnostics to emit per-dataset artifacts; add steering when coverage needs
to favor specific meta-feature bands.

```bash
cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --steer-meta \
  --meta-target linearity_proxy=0.25:0.75:1.5 \
  --out data/run_steer
```

If you want discoverable conservative defaults:

```bash
cauchy-gen generate --config configs/preset_diagnostics_on.yaml --num-datasets 25 --diagnostics --out data/run_diag
cauchy-gen generate --config configs/preset_steering_conservative.yaml --num-datasets 25 --diagnostics --out data/run_steering
```

______________________________________________________________________

## 3. Missingness workflows

Use presets for standard MCAR/MAR/MNAR runs:

```bash
cauchy-gen generate --config configs/preset_missingness_mcar.yaml --num-datasets 25 --out data/run_missing_mcar
cauchy-gen generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
cauchy-gen generate --config configs/preset_missingness_mnar.yaml --num-datasets 25 --out data/run_missing_mnar
```

Use CLI overrides for targeted MAR calibration:

```bash
cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 25 \
  --device cpu \
  --missing-rate 0.25 \
  --missing-mechanism mar \
  --missing-mar-observed-fraction 0.6 \
  --missing-mar-logit-scale 1.4 \
  --out data/run_missing_cli_mar
```

______________________________________________________________________

## 4. Curriculum workflows

Use fixed stages for controlled complexity progression, or auto mode for
sampled stage assignment.

```bash
cauchy-gen generate --config configs/preset_curriculum_stage1.yaml --num-datasets 25 --out data/run_curriculum_stage1
cauchy-gen generate --config configs/preset_curriculum_stage2.yaml --num-datasets 25 --out data/run_curriculum_stage2
cauchy-gen generate --config configs/preset_curriculum_stage3.yaml --num-datasets 25 --out data/run_curriculum_stage3
cauchy-gen generate --config configs/preset_curriculum_auto_staged.yaml --num-datasets 25 --out data/run_curriculum_auto
```

______________________________________________________________________

## 5. Many-class workflows

Use many-class presets to exercise the current rollout envelope (`n_classes_max <= 32`) with smoke-stable defaults.

```bash
cauchy-gen generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 25 --out data/run_many_class_smoke

cauchy-gen benchmark \
  --config configs/preset_many_class_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_many_class
```

The benchmark summary includes throughput/latency plus standard guardrail payloads (for example, `lineage_guardrails`).

______________________________________________________________________

## 6. Shift workflows

Use shift profiles when you want controlled distribution drift while preserving
deterministic seeds.

Profile examples:

```yaml
shift:
  enabled: true
  profile: graph_drift
```

```yaml
shift:
  enabled: true
  profile: mechanism_drift
```

```yaml
shift:
  enabled: true
  profile: noise_drift
```

```yaml
shift:
  enabled: true
  profile: mixed
```

Custom overrides:

```yaml
shift:
  enabled: true
  profile: custom
  graph_scale: 0.6
  mechanism_scale: 0.2
  noise_scale: 0.4
```

How to pick scales:

- `graph_scale`: edge-odds multiplier is `exp(ln(2) * graph_scale)`. Start at
  `0.5` for moderate structure drift.
- `mechanism_scale`: increases probability mass on nonlinear mechanism
  families. Start at `0.5` for moderate tilt.
- `noise_scale`: variance multiplier is `exp(ln(2) * noise_scale)`. Start at
  `0.5` (+1.5 dB) for moderate noise drift.

Run with any shift-enabled config:

```bash
cauchy-gen generate --config path/to/shift_config.yaml --num-datasets 25 --out data/run_shift
```

Discoverable shift presets:

```bash
cauchy-gen generate --config configs/preset_shift_graph_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_graph
cauchy-gen generate --config configs/preset_shift_mechanism_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mechanism
cauchy-gen generate --config configs/preset_shift_noise_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_noise
cauchy-gen generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

How to interpret outputs:

- Per-dataset `metadata.json` now includes `shift` with resolved profile/scales
  and derived multipliers (`edge_odds_multiplier`,
  `noise_variance_multiplier`, `mechanism_nonlinear_mass`).
- Diagnostics coverage summaries include shift observability metrics such as
  `shift_graph_scale`, `shift_edge_odds_multiplier`,
  `shift_mechanism_nonlinear_mass`, and
  `shift_noise_variance_multiplier`.

______________________________________________________________________

## 7. Benchmark workflows and guardrails

Use smoke benchmarks for quick validation and standard benchmarks for broader
performance checks.

```bash
cauchy-gen benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
cauchy-gen benchmark --suite standard --profile cpu --out-dir benchmarks/results/standard_cpu
```

Benchmark diagnostics-enabled runs:

```bash
cauchy-gen benchmark \
  --suite smoke \
  --profile cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

Benchmark missingness and curriculum guardrails:

```bash
cauchy-gen benchmark \
  --config configs/preset_missingness_mar.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_missing_mar

cauchy-gen benchmark \
  --config configs/preset_curriculum_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_curriculum_guardrails

cauchy-gen benchmark \
  --config configs/preset_shift_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_shift_guardrails
```

For regression gating in CI-like checks:

```bash
cauchy-gen benchmark \
  --config configs/preset_curriculum_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --warn-threshold-pct 10 \
  --fail-threshold-pct 20 \
  --fail-on-regression \
  --no-hardware-aware \
  --no-memory \
  --out-dir benchmarks/results/ci_smoke_curriculum_local
```

When available in a run, review benchmark summary sections such as
`missingness_guardrails`, `lineage_guardrails`, `curriculum_guardrails`, and
`shift_guardrails`.

______________________________________________________________________

## Related documents

- Output contract: [output-format.md](output-format.md)
- System guide and terminology: [how-it-works.md](how-it-works.md)
- Architecture rationale and evolution policy: [design-decisions.md](design-decisions.md)
