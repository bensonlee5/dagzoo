# Mission-Aligned Roadmap (2026Q1)

This is the canonical roadmap for `cauchy-generator`.

It maps the mission and strategic pillars in `README.md` to:

- current implemented capabilities
- known gaps
- prioritized roadmap items with explicit exit criteria

Related docs:

- Prioritized queue: `docs/improvement_ideas.md`
- Decision rubric and go/no-go gates: `docs/backlog_decision_rules.md`
- Evidence appendix: `docs/literature_evidence_2026.md`
- Current implementation details: `docs/implementation.md`

## Status Labels

- `implemented`: available in current code and exposed through config/CLI.
- `partial`: some building blocks exist, but mission-level claim is not fully met.
- `planned`: scoped and prioritized, not implemented.
- `research`: exploratory with higher uncertainty or risk.

## Current Capability Matrix

| README Mission/Pillar Claim                                         | Current State | Evidence in Repo                                                                                                               | Gap                                                                                               | Roadmap IDs                    |
| ------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------ |
| Foundation model pretraining with diverse structural priors         | `partial`     | DAG-based generation, mixed-type conversion, diagnostics extraction/coverage aggregation, soft steering, throughput benchmarks | Missingness/shift/hard-task regimes are not implemented end-to-end                                | RD-003, RD-004, RD-005, RD-007 |
| Causal discovery with ground-truth DAGs and interventional datasets | `partial`     | DAG is sampled during generation                                                                                               | Full adjacency is not exported as first-class artifact; no intervention/counterfactual generation | RD-001, RD-002                 |
| Robustness testing with hard tasks, shifts, adversarial regimes     | `planned`     | Basic filtering and diagnostics proxies exist                                                                                  | No explicit robustness profiles, missingness mechanisms, or drift controls                        | RD-003, RD-004, RD-005         |
| Causal structural integrity (hierarchical dependencies)             | `implemented` | Graph-driven node pipeline and multi-family function composition                                                               | Deeper mechanism controls are not user-configurable                                               | RD-007                         |
| Tabular realism (mixed type + postprocess hooks)                    | `partial`     | Numeric/categorical converters and E.13 postprocessing are implemented                                                         | High-cardinality/many-class limits and missingness controls are not exposed                       | RD-003, RD-006                 |
| Complexity curriculum scales features/nodes/samples                 | `partial`     | Curriculum mode stages row/split regime                                                                                        | Curriculum does not yet stage feature count or graph complexity                                   | RD-006                         |
| Hardware-native performance (Torch + hardware-aware tuning)         | `implemented` | Torch CPU/CUDA/MPS path, hardware detection, profile-based tuning, benchmark suite                                             | Parallel/distributed generation is not implemented                                                | RD-009                         |
| Parallel streaming Parquet sharding                                 | `partial`     | Streaming Parquet writing exists                                                                                               | Writing is currently single-process sequential                                                    | RD-009                         |

## Roadmap Items

### RD-001: Ground-Truth DAG Artifact Export

- Status: `planned`
- Milestone: `Now`
- Mission alignment: causal discovery
- Pillar alignment: causal structural integrity
- Goal: persist full adjacency matrix and node assignment lineage as stable dataset artifacts.
- Repo touchpoints: `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/io/parquet_writer.py`, `src/cauchy_generator/types.py`
- Exit criteria:
  - Every generated dataset includes adjacency and assignment metadata.
  - Metadata schema is documented and covered by integration tests.
  - Existing configs remain backward-compatible.

### RD-002: Interventional and Counterfactual Generation Modes

- Status: `research`
- Milestone: `Later`
- Mission alignment: causal discovery
- Pillar alignment: causal structural integrity
- Goal: support observational + interventional sampling tracks with explicit intervention specs.
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/core/node_pipeline.py`, `src/cauchy_generator/cli.py`
- Exit criteria:
  - Config supports opt-in intervention mode with safe default (`off`).
  - Generated artifacts contain intervention set and pre/post intervention metadata.
  - Acceptance tests verify truncated-factorization behavior for fixed interventions.

### RD-003: Missingness Generation (MCAR/MAR/MNAR)

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: add configurable missing-data mechanisms with deterministic seeded behavior.
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/postprocess/postprocess.py`, `src/cauchy_generator/diagnostics/metrics.py`
- Exit criteria:
  - Config supports opt-in mechanism selection and missing rate controls.
  - Generated datasets match expected missing-rate and dependency patterns.
  - Diagnostics reports include missingness summary metrics.

### RD-004: Shift-Aware SCM Generation

- Status: `planned`
- Milestone: `Next`
- Mission alignment: robustness testing, causal discovery
- Pillar alignment: causal structural integrity, tabular realism
- Goal: introduce controlled distribution-shift/drift modes in graph and mechanism sampling.
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/sampling/random_weights.py`
- Exit criteria:
  - Shift profiles are opt-in and backward-compatible.
  - Drift metrics show measurable controlled shift under enabled profiles.
  - No regression in baseline benchmark thresholds when shift is disabled.

### RD-005: Robustness Stress Profiles (Hard-Task/Adversarial Regimes)

- Status: `research`
- Milestone: `Next`
- Mission alignment: robustness testing
- Pillar alignment: tabular realism
- Goal: define reproducible stress presets (low-SNR, class imbalance, harder interactions).
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/functions/random_functions.py`, `src/cauchy_generator/postprocess/postprocess.py`, `src/cauchy_generator/bench/`
- Exit criteria:
  - Presets are selectable via config/CLI and remain opt-in.
  - Benchmarks and diagnostics confirm regimes differ from baseline in intended directions.
  - Reproducibility tests pass for fixed seed runs.

### RD-006: Curriculum Complexity Scaling (Features + Graph)

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism
- Goal: extend curriculum stages beyond row count to feature/node/depth complexity.
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/core/dataset.py`, `configs/curriculum_tabiclv2.yaml`
- Exit criteria:
  - Stage definitions include row, feature, and graph complexity controls.
  - Stage monotonicity tests verify non-decreasing complexity across stages.
  - Existing curriculum mode remains valid when new knobs are absent.

### RD-007: Many-Class and High-Cardinality Expansion

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism, causal structural integrity
- Goal: raise practical class/cardinality limits while preserving filter quality.
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/converters/categorical.py`, `src/cauchy_generator/filtering/torch_rf_filter.py`
- Exit criteria:
  - Config supports expanded ranges with safe defaults preserving current behavior.
  - Integration tests cover high-class-count generation success and label validity.
  - Filter rejection rates remain within defined guardrails on benchmark profiles.

### RD-008: Meta-Feature Coverage Steering

- Status: `implemented`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism
- Goal: maintain and harden soft steering loop using existing diagnostics coverage targets.
- Repo touchpoints: `src/cauchy_generator/diagnostics/coverage.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/cli.py`
- Exit criteria:
  - Steering can be enabled/disabled without breaking existing flows.
  - Coverage under configured target bands improves versus baseline in controlled tests.
  - Runtime overhead stays within benchmark warn threshold under default profiles.

### RD-009: Parallel and Distributed Generation/Writing

- Status: `research`
- Milestone: `Next`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: support multi-worker generation and shard writing with deterministic seed partitioning.
- Repo touchpoints: `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/io/parquet_writer.py`, `src/cauchy_generator/cli.py`
- Exit criteria:
  - Worker-aware config/API is backward-compatible and opt-in.
  - Multi-worker mode matches single-worker outputs for fixed seed equivalence checks.
  - Throughput improves on supported hardware without violating fail threshold regressions.

## Milestone Board

### Implemented

- RD-008 meta-feature coverage steering

### Now

- RD-001 ground-truth DAG artifact export
- RD-003 missingness mechanisms
- RD-006 curriculum complexity scaling
- RD-007 many-class/high-cardinality expansion

### Next

- RD-004 shift-aware SCM generation
- RD-005 robustness stress profiles
- RD-009 parallel/distributed generation

### Later

- RD-002 interventional and counterfactual generation modes

## Dependencies and Sequencing

- RD-008 is implemented and now benefits from RD-003/RD-007 because expanded data regimes improve steering utility.
- RD-005 depends on RD-003 and RD-004 for robust, controllable stress-profile construction.
- RD-002 depends on RD-001 for stable causal graph artifact lineage.
- RD-009 should land after interface contracts for RD-001/RD-006 are stable to avoid repeated parallelism refactors.

## Guardrails

- All new behavior is opt-in by default.
- Existing config files remain valid unless explicitly versioned.
- Reproducibility expectations are mandatory for every roadmap item.
- Benchmark warn/fail thresholds remain the performance gate.
