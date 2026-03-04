# Effective Diversity Audit

This document describes the rerunnable effective-diversity audit tool.

The audit has two phases:

- local overlap/equivalence checks for activations, mechanism families, and aggregations
- dataset-scale impact checks that run through the full generation pipeline
  (converter + postprocess + filter metadata paths)

The audit is analysis-only and does not mutate generator runtime behavior outside
explicit in-process ablation overrides used during the scale phase.

## Goals

- keep redundancy detection synchronized with runtime implementation
- measure whether overlaps change emitted dataset diagnostics at scale
- support baseline/regression comparisons as the repo evolves

## Runtime Synchronization

Activation auditing now reads activation names from the runtime source of truth
(`dagzoo.functions.activations.fixed_activation_names`).

If a hypothesis references an unavailable runtime primitive (for example,
`sign` vs `heaviside` after `heaviside` removal), it is emitted as:

- `status = not_applicable_runtime_absent`

## Hypothesis Registry Coverage

The audit registry explicitly tracks these claims:

- `sigmoid_vs_tanh`
- `sign_vs_heaviside`
- `softplus_vs_logsigmoid`
- `relu6_vs_hardtanh`
- `selu_vs_elu`
- `linear_vs_nn`
- `quadratic_vs_product`
- `gp_vs_nn`
- `em_vs_discretization`
- `tree_vs_discretization`
- `max_vs_logsumexp`
- `rank_vs_argsort`

## Scale-Impact Protocol

Scale phase runs one baseline arm plus ablations:

- one-at-a-time arms
- one combined arm (`combined_high_confidence` or `combined_all_claims`)

Impact metric:

- per-metric weighted shift from baseline using diagnostics summaries
- composite score = median of per-metric shifts
- meaningful by default when composite shift >= `5.0%`

Default datasets per arm by suite:

- `smoke`: 128
- `standard`: 10000
- `full`: 20000

## Baseline + Regression Workflow

The tool can persist and compare scale baselines:

- baseline payload contains per-arm `composite_shift_pct`
- regression compares current vs baseline composite shift deltas
- thresholds default to `warn=2.5`, `fail=5.0`

## CLI Usage

Local + scale with artifact output:

```bash
dagzoo diversity-audit --phase both --out-dir effective_config_artifacts/effective_diversity
```

Scale-only smoke run with custom arm count:

```bash
dagzoo diversity-audit \
  --phase scale \
  --suite smoke \
  --num-datasets-per-arm 200 \
  --arm-set all_claims
```

Save and reuse baseline:

```bash
dagzoo diversity-audit --phase scale --save-baseline benchmarks/baselines/effective_diversity.json

dagzoo diversity-audit \
  --phase scale \
  --baseline benchmarks/baselines/effective_diversity.json \
  --fail-on-regression
```

Script wrapper (equivalent options):

```bash
.venv/bin/python scripts/effective_diversity_audit.py --phase both
```

## Artifacts

Under `--out-dir`:

- `run_summary.json`
- `run_summary.md`
- `equivalence_report.json` / `.md` (when local phase runs)
- `impact_summary.json` / `.md` (when scale phase runs)
