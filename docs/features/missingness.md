# Missingness

Use missingness workflows to inject deterministic synthetic null patterns for
robustness testing under MCAR, MAR, and MNAR regimes.

______________________________________________________________________

## When to use

- You want realistic training/evaluation with incomplete tabular data.
- You need controlled ablations across missingness mechanisms.
- You want benchmark guardrails to include missingness-aware checks.

______________________________________________________________________

## Preset workflows

Use presets for standard mechanism runs:

```bash
dagzoo generate --config configs/preset_missingness_mcar.yaml --num-datasets 25 --out data/run_missing_mcar
dagzoo generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
dagzoo generate --config configs/preset_missingness_mnar.yaml --num-datasets 25 --out data/run_missing_mnar
```

______________________________________________________________________

## Targeted MAR calibration via CLI

```bash
dagzoo generate \
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

## Key options

- `--missing-rate`: overall missingness probability.
- `--missing-mechanism`: `mcar`, `mar`, or `mnar`.
- `--missing-mar-observed-fraction`: fraction of observed features used for MAR
  logits.
- `--missing-mar-logit-scale`: MAR logit sensitivity multiplier.

______________________________________________________________________

## What to inspect

- `metadata.ndjson` dataset records for resolved missingness configuration.
- Benchmark summaries for `missingness_guardrails` (when present).

For output details, see
[output-format.md](../output-format.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
