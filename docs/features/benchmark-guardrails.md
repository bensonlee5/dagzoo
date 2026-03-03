# Benchmark Workflows and Guardrails

Use benchmark workflows to validate throughput/latency and enforce regression
guardrails across default and feature-specific configs.

______________________________________________________________________

## When to use

- You need fast smoke checks before wider experimentation.
- You want standardized performance baselines by profile/suite.
- You need CI gating with warn/fail regression thresholds.

______________________________________________________________________

## Baseline workflows

Quick smoke and broader standard runs:

```bash
cauchy-gen benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
cauchy-gen benchmark --suite standard --profile cpu --out-dir benchmarks/results/standard_cpu
```

Diagnostics-enabled benchmark:

```bash
cauchy-gen benchmark \
  --suite smoke \
  --profile cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

______________________________________________________________________

## Feature-specific guardrail runs

```bash
cauchy-gen benchmark \
  --config configs/preset_missingness_mar.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_missing_mar

cauchy-gen benchmark \
  --config configs/preset_shift_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_shift_guardrails

cauchy-gen benchmark \
  --config configs/preset_noise_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_noise_guardrails
```

______________________________________________________________________

## Regression gating

For CI-like checks:

```bash
cauchy-gen benchmark \
  --config configs/preset_shift_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --warn-threshold-pct 10 \
  --fail-threshold-pct 20 \
  --fail-on-regression \
  --no-hardware-aware \
  --no-memory \
  --out-dir benchmarks/results/ci_smoke_shift_local
```

______________________________________________________________________

## What to inspect

When present in a run summary, inspect:

- `missingness_guardrails`
- `lineage_guardrails`
- `shift_guardrails`
- `noise_guardrails`

Also review throughput/latency aggregates for profile/suite trends.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Output contract: [output-format.md](../output-format.md)
- Noise workflows: [noise.md](noise.md)
