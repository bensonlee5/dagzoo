# Meta-Feature Steering

Use steering when you want generation to bias toward specific target bands
without hard-rejecting out-of-band candidates.

Steering is a soft-selection mechanism: for each dataset slot, the generator
auditions multiple candidates, scores them against target bands, and samples
one candidate from a temperature-scaled distribution over scores.

______________________________________________________________________

## When to use

- You need better coverage for specific meta-feature ranges.
- Random draws are too easy, too noisy, or otherwise misaligned.
- You want deterministic seeds with controllable bias strength.

______________________________________________________________________

## Quick start

CLI-only run with explicit targets:

```bash
cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --steer-meta \
  --meta-target linearity_proxy=0.25:0.75:1.5 \
  --out data/run_steer
```

Discoverable conservative preset:

```bash
cauchy-gen generate \
  --config configs/preset_steering_conservative.yaml \
  --num-datasets 25 \
  --diagnostics \
  --out data/run_steering
```

______________________________________________________________________

## Configuration

Steering is controlled by `steering` and `meta_feature_targets`:

```yaml
steering:
  enabled: true
  n_candidates: 8
  temperature: 0.25

meta_feature_targets:
  # metric_name: [min, max, weight]
  linearity_proxy: [0.1, 0.4, 1.0]
  snr_proxy_db: [15.0, 40.0, 2.0]
```

Key parameters:

- `n_candidates`: higher values improve target fit but increase generation time
  roughly linearly.
- `temperature`: lower values are greedier; higher values are more uniform.
- target `weight`: relative metric priority in the steering penalty.

______________________________________________________________________

## Selection mechanics

For each dataset slot:

1. Generate `n_candidates` candidate datasets with deterministic child seeds.
1. Compute steering metrics for each candidate.
1. Assign zero penalty in-band; assign scaled distance penalty out-of-band.
1. Convert penalties to probabilities with temperature-scaled softmax.
1. Sample one candidate.

The batch-level feedback loop increases effective weight for under-covered
metrics so hard targets are not ignored over time.

______________________________________________________________________

## Steering vs diagnostics

Steering and diagnostics are complementary:

| Feature    | Steering            | Diagnostics                      |
| :--------- | :------------------ | :------------------------------- |
| Purpose    | Live selection bias | Final reporting and verification |
| Data flow  | Pre-selection       | Post-emission                    |
| Metric set | Steering subset     | Wider reporting suite            |

For diagnostics workflows, see [diagnostics.md](diagnostics.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- System terminology: [how-it-works.md](../how-it-works.md)
