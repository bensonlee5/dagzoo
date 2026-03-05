# Diagnostics

Use diagnostics when you want per-dataset observability artifacts to verify
coverage, spot drift, and debug generation behavior.

______________________________________________________________________

## When to use

- You need per-dataset records in shard `metadata.ndjson` and summary-level metric coverage.
- You are validating whether presets or CLI overrides hit expected ranges.
- You want benchmark runs to include richer context for guardrail triage.

______________________________________________________________________

## Quick start

Enable diagnostics directly:

```bash
dagzoo generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --out data/run_diag
```

Use the discoverable preset:

```bash
dagzoo generate \
  --config configs/preset_diagnostics_on.yaml \
  --num-datasets 25 \
  --diagnostics \
  --out data/run_diag_preset
```

______________________________________________________________________

## Key options

- `--diagnostics`: emit diagnostics artifacts for generated datasets.
- `--out`: output directory containing datasets and diagnostic payloads.

Diagnostics also work with `benchmark`:

```bash
dagzoo benchmark \
  --suite smoke \
  --preset cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

______________________________________________________________________

## What to inspect

- Per-dataset `metadata.ndjson` records for realized generation parameters.
- Coverage summaries for meta-features and enabled observability metrics.
- Benchmark summary guardrail sections that include diagnostics context.

Exact output contracts are documented in
[output-format.md](../output-format.md).

______________________________________________________________________

## Diagnostics target bands

Diagnostics supports optional `diagnostics.meta_feature_targets` to annotate
coverage summaries with in-band counts/fractions for selected metrics.

Target bands do not alter generation; they are reporting metadata only.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- System terminology: [how-it-works.html](../how-it-works.html)
