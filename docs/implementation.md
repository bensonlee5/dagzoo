# Implementation Details

## Objective

Build a Python repository that generates synthetic tabular datasets according to TabICLv2 Appendix E (`E.2`-`E.14`) with high throughput on NVIDIA CUDA GPUs and deterministic-seed best effort behavior.

Related docs:

- Canonical roadmap: `docs/roadmap.md`
- Prioritized queue: `docs/improvement_ideas.md`
- Decision rubric: `docs/backlog_decision_rules.md`
- Literature evidence: `docs/literature_evidence_2026.md`

## Source of Truth

- Normative behavior: `reference/TabICLv2.pdf` Appendix E.
- Clarification-only sources:\
  `reference/A Closer Look at TabPFN v2.pdf` and\
  `reference/Accurate predictions on small data with a tabular foundation model.pdf`.

## Current Scope vs README Mission Claims

| Mission/Pillar Claim from README                                | Current Scope                                                                                                                             | Status  | Roadmap Follow-up |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------- |
| Foundation model pretraining with diverse priors                | Implemented baseline generation, diagnostics extraction, soft steering, configurable missingness, coverage aggregation, and benchmarks    | partial | RD-006, RD-007    |
| Causal discovery with ground-truth DAGs and interventions       | DAG sampling exists in pipeline internals; interventional generation does not                                                             | partial | RD-001, RD-002    |
| Robustness testing with hard tasks, shifts, adversarial regimes | Basic filtering and diagnostics proxies exist; missingness mechanisms and benchmark guardrails are implemented; shift/stress modes remain | partial | RD-004, RD-005    |
| Complexity curriculum across features/nodes/samples             | Current curriculum stages rows/split regime only                                                                                          | partial | RD-006            |
| Hardware-native performance with parallel streaming             | Torch + hardware-aware tuning implemented with coarse profile-tier overrides; streaming writes are sequential                             | partial | RD-009, RD-010    |

## Known Missing Capabilities (Roadmap-Tracked)

- RD-002: add interventional and counterfactual generation tracks.
- RD-004: add shift-aware SCM generation controls.
- RD-005: add robustness stress profiles for hard-task/adversarial regimes.
- RD-006: extend curriculum to feature and graph complexity.
- RD-007: expand many-class and high-cardinality support.
- RD-009: add parallel/distributed generation and shard writing.
- RD-010: add bounded hardware-adaptive autotuning beyond coarse FLOPs-tier overrides.

## Public Interfaces

### Python API

- `generate_one(config: GeneratorConfig, seed: int, device: str) -> DatasetBundle`
- `generate_batch(config: GeneratorConfig, num_datasets: int, seed: int, device: str) -> list[DatasetBundle]`
- `write_parquet_shards(bundles, out_dir, shard_size, compression="zstd")`
- `DatasetConfig` missingness controls:
  - `missing_rate`
  - `missing_mechanism` (`none|mcar|mar|mnar`)
  - `missing_mar_observed_fraction`
  - `missing_mar_logit_scale`
  - `missing_mnar_logit_scale`

### CLI

- `cauchy-gen generate --config ... --num-datasets ... --device cuda --seed ...`
- `cauchy-gen generate --missing-rate ... --missing-mechanism ... --missing-mar-observed-fraction ... --missing-mar-logit-scale ... --missing-mnar-logit-scale ...`
- `cauchy-gen benchmark --suite standard --profile all --baseline ... --fail-on-regression`

### Output Contract

Each `DatasetBundle` contains:

- `X_train`, `y_train`, `X_test`, `y_test`
- `feature_types` (`"num"` or `"cat"`)
- metadata (seed lineage, graph stats, function selections, filter decision, missingness summary when enabled)

Persist generated outputs as Parquet shards with a sidecar metadata JSON per shard.

Current metadata includes summary graph stats, versioned DAG lineage (`metadata.lineage`), config lineage, and optional `missingness` payload fields (configured/realized rates and per-split counts). Persisted shard metadata rewrites dense lineage adjacency into compact shard-level artifact pointers (`adjacency.bitpack.bin` + `adjacency.index.json`), and benchmark profile summaries include `lineage_guardrails` to measure export overhead.

#### DAG Lineage Schema (v1.0 + v1.1 compact persistence)

RD-001 rollout defines versioned lineage payloads with strict validation and compact persistence.

- Location: `metadata.lineage`
- Envelope:
  - `schema_name`: must equal `cauchy_generator.dag_lineage`
  - `schema_version`: one of `1.0.0` (dense) or `1.1.0` (compact persisted form)
- Dense graph payload (`1.0.0`):
  - `graph.n_nodes`: integer, `>= 2`
  - `graph.adjacency`: dense square `n_nodes x n_nodes` matrix of integer `0/1` values
  - Validation enforces diagonal zeros and upper-triangular encoding (topological-order DAG form)
- Compact graph payload (`1.1.0`):
  - `graph.n_nodes`, `graph.edge_count`
  - `graph.adjacency_ref`: `{encoding, blob_path, index_path, dataset_index, bit_offset, bit_length, sha256}`
  - Encoding is fixed at `upper_triangle_bitpack_v1`; `bit_length` must match `n_nodes`.
- Assignment payload:
  - `assignments.feature_to_node`: list of node indices in `[0, n_nodes - 1]`
  - `assignments.target_to_node`: node index in `[0, n_nodes - 1]`

Compatibility contract:

- Validation accepts metadata without `lineage` by default (`required=False`) so existing configs/runs stay valid.
- Consumers can require lineage explicitly (`required=True`) when later rollout phases make emission mandatory.

## Runtime Profiles

- `configs/default.yaml`: balanced local development profile.
- `configs/benchmark_cpu.yaml`: CPU benchmark profile.
- `configs/benchmark_cuda_desktop.yaml`: desktop CUDA benchmark profile.
- `configs/benchmark_cuda_h100.yaml`: H100 CUDA benchmark profile.
- `configs/preset_cuda_h100.yaml`: high-throughput datacenter preset.
- `configs/preset_missingness_mcar.yaml`: MCAR missingness preset.
- `configs/preset_missingness_mar.yaml`: MAR missingness preset.
- `configs/preset_missingness_mnar.yaml`: MNAR missingness preset.
- `configs/preset_lineage_benchmark_smoke.yaml`: CPU smoke benchmark preset for lineage export guardrail checks.
- Runtime currently applies coarse profile-tier overrides from GPU FLOPS lookup + fallback behavior; adaptive autotuning is tracked in RD-010.

## Module Plan (Appendix Mapping)

- `sampling/correlated.py`: correlated scalar sampler (`E.2`)
- `core/dataset.py`: dataset orchestration (`E.3`)
- `graph/cauchy_graph.py`: random Cauchy DAG (`E.4`)
- `core/node_pipeline.py`: per-node flow (`E.5`)
- `converters/numeric.py`, `converters/categorical.py`: converters (`E.6`)
- `functions/multi.py`: concatenation vs per-parent aggregation (`E.7`)
- `functions/random_functions.py`: NN/tree/discretization/GP/linear/quadratic/EM/product (`E.8`)
- `functions/activations.py`: fixed + parametric activations (`E.9`)
- `linalg/random_matrices.py`: five matrix families and postprocessing (`E.10`)
- `sampling/random_weights.py`: positive normalized weights (`E.11`)
- `sampling/random_points.py`: base distributions + random function transform (`E.12`)
- `postprocess/postprocess.py`: cleanup, scaling, class/index permutation (`E.13`)
- `filtering/torch_rf_filter.py`: Torch-native RF OOB filter (`E.14`)

## Performance Strategy

1. Current generator path runs Torch on all devices (CPU/CUDA/MPS); NumPy usage is mainly in diagnostics/reporting and serialization boundaries.
1. Keep kernels batch-oriented with vectorized torch operations and avoid Python loops in inner math paths.
1. Use optional filtering (`E.14`) behind config flags to avoid CPU bottlenecks in throughput benchmarks.
1. Profile with `bench/throughput.py` and track JSON baseline regressions by preset.
1. Missingness-enabled benchmark runs include acceptance/runtime guardrails against missingness-off controls.
1. Benchmark profile summaries include lineage-export persistence overhead guardrails (`lineage_guardrails`) against lineage-stripped control persistence runs.
1. Next hardware-aware step is bounded adaptive autotuning with explicit telemetry/guardrails (RD-010).
1. Next roadmap step for throughput is controlled multi-worker execution (RD-009) while preserving seeded behavior.

## Reproducibility Strategy

1. Global run seed -> per-dataset seed -> per-component derived seeds.
1. Central RNG utilities wrap Python/NumPy/Torch RNGs.
1. Document expected backend variation (best effort, not strict bitwise determinism).

## Validation and Benchmarks

### Correctness

- Unit invariants for ranges, shapes, DAG validity, converter class ranges, and matrix normalization.
- Unit/integration coverage for missingness mask invariants, deterministic behavior, and end-to-end metadata emission.
- Integration tests for end-to-end classification/regression paths.

### Reproducibility

- Fixed seed should reproduce metadata exactly and numeric outputs within tolerance.

### Performance

- Benchmark suites: `smoke`, `standard`, `full`.
- Artifacts: JSON + Markdown summaries under `benchmarks/results/<timestamp>/`.
- Soft regression gate: warn at configurable threshold, fail only on severe regression with `--fail-on-regression`.

## Delivery Phases

1. Scaffold package/config/CLI + minimal generation loop.
1. Implement `E.2`-`E.7` plus unit tests.
1. Implement `E.8`-`E.10` with GPU-first tensor kernels.
1. Implement `E.11`-`E.14`, parquet writing, and integration tests.
1. Add benchmark harness, tune bottlenecks, and lock baseline.
1. Extend mission coverage through roadmap items RD-001..RD-010 in `docs/roadmap.md`.
