# Mechanism Diversity

Use mechanism-diversity workflows when you want to stage new mechanism families
through the existing `mechanism.function_family_mix` surface and verify that
they produce realized family uptake plus measurable diversity shift.

______________________________________________________________________

## When to use

- You want to compare the current baseline sampler against an opt-in family mix.
- You need realized mechanism-family counts in bundle metadata and audit reports.
- You want benchmark, diversity-audit, and filter-calibration evidence before
  considering any broader rollout.

______________________________________________________________________

## Public interface rule

This workflow intentionally keeps the config surface narrow:

- No new config sections.
- No family-specific scalar knobs.
- No new CLI flags.
- The only public family toggle added in this epic is
  `mechanism.function_family_mix.piecewise`, and it must be paired with at
  least one explicit branch family from `tree`, `discretization`, `gp`,
  `linear`, or `quadratic`.

The curated smoke presets use the minimal valid staged mix, `piecewise` +
`linear`, so the rollout remains explicit about which leaf family can appear
inside piecewise branches.

______________________________________________________________________

## Generate with `piecewise`

Use the curated smoke preset for direct generation:

```bash
dagzoo generate \
  --config configs/preset_mechanism_piecewise_generate_smoke.yaml \
  --num-datasets 10 \
  --device cpu \
  --hardware-policy none \
  --out data/run_piecewise_smoke_local
```

Inspect shard `metadata.ndjson` for:

- `mechanism_families.sampled_family_counts`
- `mechanism_families.families_present`
- `mechanism_families.total_function_plans`

______________________________________________________________________

## Diversity-audit workflow

Compare the matched baseline preset against the opt-in `piecewise` preset:

```bash
dagzoo diversity-audit \
  --baseline-config configs/preset_mechanism_baseline_benchmark_smoke.yaml \
  --variant-config configs/preset_mechanism_piecewise_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_piecewise
```

Inspect `summary.json` and `summary.md` for:

- `comparisons[*].diversity_composite_shift_pct`
- `baseline.mechanism_family_summary`
- `variants[*].mechanism_family_summary`

The `piecewise` preset is only a viable winner if the report shows both
realized `piecewise` usage and clearly non-zero diversity shift against the
baseline.

______________________________________________________________________

## Filter-calibration workflow

Use the filter-enabled preset to check accepted-corpus throughput and yield
against diversity shift:

```bash
dagzoo filter-calibration \
  --config configs/preset_mechanism_piecewise_filter_smoke.yaml \
  --suite smoke \
  --device cpu \
  --out-dir benchmarks/results/filter_calibration_piecewise
```

Inspect `summary.json` and `summary.md` for:

- `summary.best_overall_threshold_requested`
- `candidates[*].filter_accepted_datasets_per_minute`
- `candidates[*].diversity_status`
- `candidates[*].mechanism_family_summary`

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
- Diagnostics: [diagnostics.md](diagnostics.md)
