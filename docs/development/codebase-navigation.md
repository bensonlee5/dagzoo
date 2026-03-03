# Codebase Navigation

The project is organized into functional modules that manage the lifecycle
of a synthetic dataset, from configuration and causal graph sampling to
node execution and quality filtering.

## 1. Entry Points & Orchestration

The high-level logic that bridges CLI/API requests to the generation engine.

- [`src/dagsynth/cli.py`](../../src/dagsynth/cli.py): Maps CLI flags to `GeneratorConfig` and handles command dispatch.
- [`src/dagsynth/core/dataset.py`](../../src/dagsynth/core/dataset.py): The main orchestration engine. Manages batch generation, fixed-layout planning, and end-to-end synchronization.
- [`src/dagsynth/core/generation_engine.py`](../../src/dagsynth/core/generation_engine.py): Per-dataset Torch generation loop: node execution, retry, stratified split, X/y assembly.
- [`src/dagsynth/core/config_resolution.py`](../../src/dagsynth/core/config_resolution.py): Layered config resolution, produces `effective_config_trace.yaml`.

## 2. The Generation Pipeline (The "Assembly Line")

Follow this sequence to understand how a latent causal structure becomes a realized dataset.

- **Structure ([`graph/`](../../src/dagsynth/graph/)):** Samples the underlying Directed Acyclic Graph (DAG).
- **Layout ([`core/layout.py`](../../src/dagsynth/core/layout.py)):** Maps features and targets to DAG nodes and assigns data types.
- **Execution ([`core/node_pipeline.py`](../../src/dagsynth/core/node_pipeline.py)):** Processes nodes in topological order, applying functional relationships.
- **Mechanisms ([`functions/`](../../src/dagsynth/functions/)):** Contains the mathematical families (linear, non-linear, mixture) that define how nodes interact.
- **Conversion ([`converters/`](../../src/dagsynth/converters/)):** Transforms latent continuous values into observable numeric or categorical data.
- **Sampling ([`sampling/`](../../src/dagsynth/sampling/)):** Noise-family primitives (gaussian/laplace/student_t), random-point geometry, correlated sampling.
- **Linear algebra ([`linalg/`](../../src/dagsynth/linalg/)):** Random matrix families used by mechanism functions.

## 3. Control, Integrity & Quality

Infrastructure that ensures reproducibility, deterministic behavior, and data quality.

- [`src/dagsynth/rng.py`](../../src/dagsynth/rng.py): The `SeedManager` ensures strictly isolated, deterministic child seeds for every component.
- [`src/dagsynth/filtering/`](../../src/dagsynth/filtering/): Implements the "learnability gate" (via CPU ExtraTrees) to reject invalid or trivial datasets.
- [`src/dagsynth/core/metrics_torch.py`](../../src/dagsynth/core/metrics_torch.py): Unified torch-native metric extraction used by diagnostics and generation telemetry.
- [`src/dagsynth/postprocess/`](../../src/dagsynth/postprocess/): Handles final-stage transformations; deterministic missingness (MCAR/MAR/MNAR) logic lives in [`sampling/missingness.py`](../../src/dagsynth/sampling/missingness.py).
- [`src/dagsynth/core/shift.py`](../../src/dagsynth/core/shift.py): Shift mode resolution and scale coefficients.
- [`src/dagsynth/core/noise_runtime.py`](../../src/dagsynth/core/noise_runtime.py): Per-dataset noise family resolution.

## 4. Configuration, Hardware & Architecture

- [`src/dagsynth/config.py`](../../src/dagsynth/config.py): The source of truth for all generator settings, implemented as strongly-typed dataclasses.
- [`src/dagsynth/hardware.py`](../../src/dagsynth/hardware.py) + [`hardware_policy.py`](../../src/dagsynth/hardware_policy.py): Tier detection and extensible policy registry.
- [`docs/design-decisions.md`](../design-decisions.md): Rationale behind the architectural choices and reproducibility guarantees.

## 5. Output & Benchmarking

- [`src/dagsynth/io/`](../../src/dagsynth/io/): Parquet shard writer, lineage bit-packing and schema.
- [`src/dagsynth/bench/`](../../src/dagsynth/bench/): Benchmark suite orchestration, guardrails (missingness/lineage/shift/noise), baseline regression.

______________________________________________________________________

### Tip: Audit via Verification

To verify the system's invariants and determinism while exploring the code:

```bash
uv run pytest -v tests/test_generate.py tests/test_rng.py
```
