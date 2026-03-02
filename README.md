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

View help and available options for commands:

```bash
cauchy-gen --help
cauchy-gen generate --help
cauchy-gen benchmark --help
```

## Features

- Diagnostics: exposes per-dataset artifacts so you can verify coverage, inspect drift, and debug generation outcomes.
- Missingness (MCAR/MAR/MNAR): injects deterministic null patterns to evaluate models under realistic incomplete-data regimes.
- Fixed-layout batch generation: reuse one sampled layout across many datasets for easier high-throughput generation and analysis.
- Many-class workflows: stress-tests classification behavior near the current rollout envelope with stable preset and benchmark paths.
- Shift/drift controls: introduces interpretable graph/mechanism/noise drift for robustness and distribution-shift evaluation.
- Benchmark guardrails: provides repeatable runtime and metadata checks for local validation and CI-style regression gating.

## Documentation (End Users)

- [docs/usage-guide.md](docs/usage-guide.md): Primary workflow hub.
- [docs/how-it-works.md](docs/how-it-works.md): System flow and terminology.
- [docs/output-format.md](docs/output-format.md): Output schema and artifacts.
- Feature guides:
  [diagnostics](docs/features/diagnostics.md),
  [missingness](docs/features/missingness.md),
  [many-class](docs/features/many-class.md),
  [shift](docs/features/shift.md),
  [benchmark guardrails](docs/features/benchmark-guardrails.md)

## Codebase Navigation

The project is organized into functional modules that manage the lifecycle of a synthetic dataset, from configuration and causal graph sampling to node execution and quality filtering.

### 1. Entry Points & Orchestration

The high-level logic that bridges CLI/API requests to the generation engine.

- [`src/cauchy_generator/cli.py`](src/cauchy_generator/cli.py): Maps CLI flags to `GeneratorConfig` and handles command dispatch.
- [`src/cauchy_generator/core/dataset.py`](src/cauchy_generator/core/dataset.py): The main orchestration engine. Manages batch generation, fixed-layout planning, and end-to-end synchronization.

### 2. The Generation Pipeline (The "Assembly Line")

Follow this sequence to understand how a latent causal structure becomes a realized dataset.

- **Structure ([`graph/`](src/cauchy_generator/graph/)):** Samples the underlying Directed Acyclic Graph (DAG).
- **Layout ([`core/layout.py`](src/cauchy_generator/core/layout.py)):** Maps features and targets to DAG nodes and assigns data types.
- **Execution ([`core/node_pipeline.py`](src/cauchy_generator/core/node_pipeline.py)):** Processes nodes in topological order, applying functional relationships.
- **Mechanisms ([`functions/`](src/cauchy_generator/functions/)):** Contains the mathematical families (linear, non-linear, mixture) that define how nodes interact.
- **Conversion ([`converters/`](src/cauchy_generator/converters/)):** Transforms latent continuous values into observable numeric or categorical data.

### 3. Control, Integrity & Quality

Infrastructure that ensures reproducibility, deterministic behavior, and data quality.

- [`src/cauchy_generator/rng.py`](src/cauchy_generator/rng.py): The `SeedManager` ensures strictly isolated, deterministic child seeds for every component.
- [`src/cauchy_generator/filtering/`](src/cauchy_generator/filtering/): Implements the "learnability gate" (via CPU ExtraTrees) to reject invalid or trivial datasets.
- [`src/cauchy_generator/core/metrics_torch.py`](src/cauchy_generator/core/metrics_torch.py): Unified torch-native metric extraction used by diagnostics and generation telemetry.
- [`src/cauchy_generator/postprocess/`](src/cauchy_generator/postprocess/): Handles final-stage transformations, including deterministic missingness (MCAR/MAR/MNAR) injection.

### 4. Configuration & Architecture

- [`src/cauchy_generator/config.py`](src/cauchy_generator/config.py): The source of truth for all generator settings, implemented as strongly-typed dataclasses.
- [`docs/design-decisions.md`](docs/design-decisions.md): Rationale behind the architectural choices and reproducibility guarantees.

### Tip: Audit via Verification

To verify the system's invariants and determinism while exploring the code:

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
