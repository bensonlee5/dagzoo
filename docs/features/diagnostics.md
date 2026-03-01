# Diagnostics

Use diagnostics when you want per-dataset observability artifacts to verify
coverage, spot drift, and debug generation behavior.

______________________________________________________________________

## When to use

- You need per-dataset `metadata.json` and summary-level metric coverage.
- You are validating whether presets or CLI overrides hit expected ranges.
- You want benchmark runs to include richer context for guardrail triage.

______________________________________________________________________

## Quick start

Enable diagnostics directly:

```bash
cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --out data/run_diag
```

Use the discoverable preset:

```bash
cauchy-gen generate \
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
cauchy-gen benchmark \
  --suite smoke \
  --profile cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

______________________________________________________________________

## What to inspect

- Per-dataset `metadata.json` values for realized generation parameters.
- Coverage summaries for meta-features and enabled observability metrics.
- Benchmark summary guardrail sections that include diagnostics context.

Exact output contracts are documented in
[output-format.md](../output-format.md).

______________________________________________________________________

## Diagnostics and steering

Diagnostics and steering are complementary:

- Steering biases candidate selection during generation.
- Diagnostics reports what was emitted after selection.

For steering configuration and behavior, see
[steering.md](steering.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- System terminology: [how-it-works.md](../how-it-works.md)
