# Noise Diversification

Use noise-family workflows when you want non-Gaussian stochastic regimes while
retaining deterministic seed behavior and explicit metadata reporting.

______________________________________________________________________

## When to use

- You want heavier-tail stochasticity than legacy defaults.
- You need deterministic comparisons across Gaussian/Laplace/Student-t regimes.
- You want benchmark guardrails for runtime impact and metadata validity.

______________________________________________________________________

## Supported families

- `legacy`: compatibility mode for default behavior.
- `gaussian`: explicit Gaussian sampling.
- `laplace`: heavier-tailed Laplace noise.
- `student_t`: heavy-tailed Student-t (`df > 2`).
- `mixture`: weighted mixture over Gaussian/Laplace/Student-t.

______________________________________________________________________

## Preset workflows

Generate smoke datasets for each family:

```bash
cauchy-gen generate --config configs/preset_noise_gaussian_generate_smoke.yaml --num-datasets 25 --out data/run_noise_gaussian
cauchy-gen generate --config configs/preset_noise_laplace_generate_smoke.yaml --num-datasets 25 --out data/run_noise_laplace
cauchy-gen generate --config configs/preset_noise_student_t_generate_smoke.yaml --num-datasets 25 --out data/run_noise_student_t
cauchy-gen generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --out data/run_noise_mixture
```

Benchmark guardrail smoke run:

```bash
cauchy-gen benchmark \
  --config configs/preset_noise_benchmark_smoke.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_noise
```

______________________________________________________________________

## What to inspect

- Per-dataset `metadata.ndjson` entries:
  - `noise.family_requested`
  - `noise.family_sampled`
  - `noise.sampling_strategy`
  - `noise.scale`
  - `noise.student_t_df`
  - `noise.mixture_weights` (when requested family is `mixture`)
- Benchmark summary `noise_guardrails`:
  - metadata coverage/validity
  - sampled-family counts
  - runtime delta vs legacy-noise control

For output details, see [output-format.md](../output-format.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
