# cauchy-generator

High-throughput synthetic tabular data generation built around causal structure.
Use it to generate, benchmark, and stress-test tabular datasets with
deterministic seed behavior.

## Quick Start

Install:

```bash
uv tool install cauchy-generator
```

Generate a default batch:

```bash
cauchy-gen generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

Run a smoke benchmark:

```bash
cauchy-gen benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
```

Inspect detected hardware profile:

```bash
cauchy-gen hardware
```

## Features

- Diagnostics: exposes per-dataset artifacts so you can verify coverage, inspect drift, and debug generation outcomes.
- Missingness (MCAR/MAR/MNAR): injects deterministic null patterns to evaluate models under realistic incomplete-data regimes.
- Meta-feature steering: biases sampling toward target metric bands when baseline draws under-cover the task characteristics you need.
- Curriculum staging: lets you scale complexity from easier to harder regimes in a controlled, reproducible way.
- Many-class workflows: stress-tests classification behavior near the current rollout envelope with stable preset and benchmark paths.
- Shift/drift controls: introduces interpretable graph/mechanism/noise drift for robustness and distribution-shift evaluation.
- Benchmark guardrails: provides repeatable runtime and metadata checks for local validation and CI-style regression gating.

## Documentation (End Users)

- [docs/usage-guide.md](docs/usage-guide.md): Primary workflow hub.
- [docs/how-it-works.md](docs/how-it-works.md): System flow and terminology.
- [docs/output-format.md](docs/output-format.md): Output schema and artifacts.
- Feature guides:
  [diagnostics](docs/features/diagnostics.md),
  [steering](docs/features/steering.md),
  [missingness](docs/features/missingness.md),
  [curriculum](docs/features/curriculum.md),
  [many-class](docs/features/many-class.md),
  [shift](docs/features/shift.md),
  [benchmark guardrails](docs/features/benchmark-guardrails.md)

## Codebase Navigation (Contributors)

Use this sequence for a comprehensive audit of control flow, generation logic, and reproducibility guarantees.

1. Entry points (orchestration): audit how config and runtime choices flow into generation.
   - [src/cauchy_generator/cli.py](src/cauchy_generator/cli.py): map CLI flags to `GeneratorConfig` and command dispatch.
   - [src/cauchy_generator/core/dataset.py](src/cauchy_generator/core/dataset.py): audit `generate_batch_iter` and `_generate_torch` for end-to-end synchronization.
1. Generative blueprint (structure): audit how dataset skeletons are sampled before value generation.
   - [src/cauchy_generator/core/curriculum.py](src/cauchy_generator/core/curriculum.py): stagewise complexity scaling.
   - [src/cauchy_generator/core/layout.py](src/cauchy_generator/core/layout.py): feature/target assignment to DAG nodes.
   - [src/cauchy_generator/graph/cauchy_graph.py](src/cauchy_generator/graph/cauchy_graph.py): Cauchy-logit graph sampling.
1. Assembly line (node execution): audit how latent node outputs become observables.
   - [src/cauchy_generator/core/node_pipeline.py](src/cauchy_generator/core/node_pipeline.py): per-node execution sequence and normalization/scaling flow.
   - [src/cauchy_generator/functions/random_functions.py](src/cauchy_generator/functions/random_functions.py): mechanism-family implementations and vectorized execution paths.
   - [src/cauchy_generator/converters/](src/cauchy_generator/converters/): latent-to-observable conversion for numeric/categorical outputs.
1. Integrity and reproducibility (infrastructure): audit determinism and quality gates.
   - [src/cauchy_generator/rng.py](src/cauchy_generator/rng.py): `SeedManager` child-seed isolation and determinism.
   - [src/cauchy_generator/filtering/torch_rf_filter.py](src/cauchy_generator/filtering/torch_rf_filter.py): learnability gate behavior and rejection criteria.
   - [src/cauchy_generator/core/steering_metrics.py](src/cauchy_generator/core/steering_metrics.py): unified torch-native metrics used in steering and diagnostics.
1. Key specification references: cross-check implementation against intended contracts.
   - [docs/how-it-works.md](docs/how-it-works.md): conceptual data flow.
   - [docs/design-decisions.md](docs/design-decisions.md): rationale for architecture and implementation choices.
   - [docs/output-format.md](docs/output-format.md): emitted artifact and metadata contract.

### Pro Tip: Audit via Verification

Run focused integration tests while auditing to verify enforced invariants and determinism:

```bash
uv run pytest -v tests/test_generate.py tests/test_rng.py
```

## Python API

```python
from cauchy_generator import GeneratorConfig, generate_one

config = GeneratorConfig.from_yaml("configs/default.yaml")
bundle = generate_one(config, seed=42)
print(bundle.X_train.shape, bundle.y_train.shape)
```

For command-line and workflow details, use
[docs/usage-guide.md](docs/usage-guide.md).

## Roadmap and Development

- [docs/development/roadmap.md](docs/development/roadmap.md)
- [docs/development/backlog_decision_rules.md](docs/development/backlog_decision_rules.md)
- [docs/design-decisions.md](docs/design-decisions.md)
- [reference/literature_evidence_2026.md](reference/literature_evidence_2026.md)
