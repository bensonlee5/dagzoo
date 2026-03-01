# Curriculum

Use curriculum workflows when you want controlled complexity progression across
datasets, either by fixed stage or automatic stage sampling.

______________________________________________________________________

## When to use

- You are training models that benefit from staged difficulty.
- You want reproducible low/medium/high complexity datasets.
- You need benchmark guardrails that validate curriculum behavior.

______________________________________________________________________

## Preset workflows

Use fixed stages:

```bash
cauchy-gen generate --config configs/preset_curriculum_stage1.yaml --num-datasets 25 --out data/run_curriculum_stage1
cauchy-gen generate --config configs/preset_curriculum_stage2.yaml --num-datasets 25 --out data/run_curriculum_stage2
cauchy-gen generate --config configs/preset_curriculum_stage3.yaml --num-datasets 25 --out data/run_curriculum_stage3
```

Use automatic staged assignment:

```bash
cauchy-gen generate --config configs/preset_curriculum_auto_staged.yaml --num-datasets 25 --out data/run_curriculum_auto
```

______________________________________________________________________

## Benchmark workflow

```bash
cauchy-gen benchmark \
  --config configs/preset_curriculum_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_curriculum_guardrails
```

______________________________________________________________________

## What to inspect

- Per-dataset metadata for resolved stage and sampled shape parameters.
- Benchmark summaries for `curriculum_guardrails` (when available).

For schema and persistence details, see
[output-format.md](../output-format.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
