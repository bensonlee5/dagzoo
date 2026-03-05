# Codebase Navigation

The project is organized into functional modules that manage the lifecycle
of a synthetic dataset, from configuration and causal graph sampling to
node execution and quality filtering.

## 1. Entry Points & Orchestration

The high-level logic that bridges CLI/API requests to the generation engine.

- [`src/dagzoo/cli.py`](../../src/dagzoo/cli.py): Maps CLI flags to `GeneratorConfig` and handles command dispatch.
- [`src/dagzoo/core/dataset.py`](../../src/dagzoo/core/dataset.py): The main orchestration engine. Manages batch generation, fixed-layout planning, and end-to-end synchronization.
- [`src/dagzoo/core/generation_engine.py`](../../src/dagzoo/core/generation_engine.py): Per-dataset Torch generation loop: node execution, retry, stratified split, X/y assembly.
- [`src/dagzoo/core/config_resolution.py`](../../src/dagzoo/core/config_resolution.py): Layered config resolution, produces `effective_config_trace.yaml`.

## 2. The Generation Pipeline (The "Assembly Line")

Follow this sequence to understand how a latent causal structure becomes a realized dataset.

- **Structure ([`graph/`](../../src/dagzoo/graph/)):** Samples the underlying Directed Acyclic Graph (DAG).
- **Layout ([`core/layout.py`](../../src/dagzoo/core/layout.py)):** Maps features and targets to DAG nodes and assigns data types.
- **Execution ([`core/node_pipeline.py`](../../src/dagzoo/core/node_pipeline.py)):** Processes nodes in topological order, applying functional relationships.
- **Mechanisms ([`functions/`](../../src/dagzoo/functions/)):** Contains the mathematical families (linear, non-linear, mixture) that define how nodes interact.
- **Conversion ([`converters/`](../../src/dagzoo/converters/)):** Transforms latent continuous values into observable numeric or categorical data.
- **Sampling ([`sampling/`](../../src/dagzoo/sampling/)):** Noise-family primitives (gaussian/laplace/student_t), random-point geometry, correlated sampling.
- **Linear algebra ([`linalg/`](../../src/dagzoo/linalg/)):** Random matrix families used by mechanism functions.

## 3. Control, Integrity & Quality

Infrastructure that ensures reproducibility, deterministic behavior, and data quality.

- [`src/dagzoo/rng.py`](../../src/dagzoo/rng.py): The `SeedManager` ensures strictly isolated, deterministic child seeds for every component.
- [`src/dagzoo/filtering/`](../../src/dagzoo/filtering/): Implements deferred CPU ExtraTrees filtering (`dagzoo filter`) and filter replay utilities.
- [`src/dagzoo/core/metrics_torch.py`](../../src/dagzoo/core/metrics_torch.py): Unified torch-native metric extraction used by diagnostics and generation telemetry.
- [`src/dagzoo/postprocess/`](../../src/dagzoo/postprocess/): Handles final-stage transformations; deterministic missingness (MCAR/MAR/MNAR) logic lives in [`sampling/missingness.py`](../../src/dagzoo/sampling/missingness.py).
- [`src/dagzoo/core/shift.py`](../../src/dagzoo/core/shift.py): Shift mode resolution and scale coefficients.
- [`src/dagzoo/core/noise_runtime.py`](../../src/dagzoo/core/noise_runtime.py): Per-dataset noise family resolution.

## 4. Configuration, Hardware & Architecture

- [`src/dagzoo/config.py`](../../src/dagzoo/config.py): The source of truth for all generator settings, implemented as strongly-typed dataclasses.
- [`src/dagzoo/hardware.py`](../../src/dagzoo/hardware.py) + [`hardware_policy.py`](../../src/dagzoo/hardware_policy.py): Tier detection and extensible policy registry.
- [`docs/development/design-decisions.md`](design-decisions.md): Rationale behind the architectural choices and reproducibility guarantees.

## 5. Output & Benchmarking

- [`src/dagzoo/io/`](../../src/dagzoo/io/): Parquet shard writer, lineage bit-packing and schema.
- [`src/dagzoo/bench/`](../../src/dagzoo/bench/): Benchmark suite orchestration, guardrails (missingness/lineage/shift/noise), baseline regression.

______________________________________________________________________

### Tip: Audit via Verification

To verify the system's invariants and determinism while exploring the code:

```bash
uv run pytest -v tests/test_generate.py tests/test_rng.py
```
