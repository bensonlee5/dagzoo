# Shift / Drift

Use shift workflows when you want controlled distribution drift while preserving
deterministic seeds and interpretable scale semantics.

______________________________________________________________________

## When to use

- You need train/test distribution shift for robustness evaluation.
- You want independent control over graph, mechanism, and noise drift.
- You want shift-aware observability in metadata and diagnostics coverage.

______________________________________________________________________

## Shift modes

Mode-only examples:

```yaml
shift:
  enabled: true
  mode: graph_drift
```

```yaml
shift:
  enabled: true
  mode: mechanism_drift
```

```yaml
shift:
  enabled: true
  mode: noise_drift
```

```yaml
shift:
  enabled: true
  mode: mixed
```

Custom mode with explicit scales:

```yaml
shift:
  enabled: true
  mode: custom
  graph_scale: 0.6
  mechanism_scale: 0.2
  variance_scale: 0.4
```

______________________________________________________________________

## Scale interpretation

- `graph_scale`: edge-odds multiplier is `exp(ln(2) * graph_scale)`.
  Start at `0.5` for moderate structure drift.
- `mechanism_scale`: increases probability mass on nonlinear mechanism
  families. Start at `0.5` for moderate mechanism tilt.
- `variance_scale`: variance multiplier is `exp(ln(2) * variance_scale)`.
  Start at `0.5` (+1.5 dB) for moderate noise drift.

______________________________________________________________________

## Generation workflows

Run any shift-enabled config:

```bash
dagzoo generate --config path/to/shift_config.yaml --num-datasets 25 --out data/run_shift
```

Use discoverable smoke presets:

```bash
dagzoo generate --config configs/preset_shift_graph_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_graph
dagzoo generate --config configs/preset_shift_mechanism_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mechanism
dagzoo generate --config configs/preset_shift_noise_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_noise
dagzoo generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

______________________________________________________________________

## What to inspect

- Per-dataset `metadata.ndjson` records include resolved mode/scales and derived
  multipliers (`edge_odds_multiplier`, `noise_variance_multiplier`,
  `mechanism_nonlinear_mass`).
- Diagnostics coverage summaries include shift observability metrics such as
  `shift_graph_scale`, `shift_edge_odds_multiplier`,
  `shift_mechanism_nonlinear_mass`, and `shift_noise_variance_multiplier`.

Benchmark runs can surface `shift_guardrails` in summaries.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
