# Many-Class

Use many-class workflows to generate and benchmark classification datasets near
the current rollout envelope (`n_classes_max <= 32`).

______________________________________________________________________

## When to use

- You are stress-testing multi-class performance beyond low-class regimes.
- You need smoke-stable presets for higher class cardinality.
- You want guardrail visibility during many-class benchmarking.

______________________________________________________________________

## Generation workflow

```bash
dagzoo generate \
  --config configs/preset_many_class_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_many_class_smoke
```

______________________________________________________________________

## Benchmark workflow

```bash
dagzoo benchmark \
  --config configs/preset_many_class_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_many_class
```

Benchmark summaries include throughput/latency plus standard guardrail payloads
such as `lineage_guardrails`.

______________________________________________________________________

## What to inspect

- Class count and target distribution in emitted metadata.
- Benchmark summary sections for latency, throughput, and guardrails.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
