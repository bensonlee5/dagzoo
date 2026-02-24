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

| README Mission/Pillar Claim                                         | Current State | Evidence in Repo                                                                                                                                         | Gap                                                                                                 | Roadmap IDs            |
| ------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------- |
| Foundation model pretraining with diverse structural priors         | `partial`     | DAG-based generation, mixed-type conversion, diagnostics extraction/coverage aggregation, soft steering, configurable missingness, throughput benchmarks | Shift/hard-task regimes and many-class expansion are not implemented end-to-end                     | RD-004, RD-005, RD-007 |
| Causal discovery with ground-truth DAGs and interventional datasets | `partial`     | DAG lineage metadata is emitted per dataset and persisted as compact shard-level artifacts with schema validation and benchmark guardrails               | Interventional/counterfactual generation semantics are not implemented                              | RD-002                 |
| Robustness testing with hard tasks, shifts, adversarial regimes     | `partial`     | Basic filtering and diagnostics proxies exist; missingness mechanisms are implemented with deterministic controls and benchmark guardrails               | No explicit robustness profiles or shift/drift controls                                             | RD-004, RD-005         |
| Causal structural integrity (hierarchical dependencies)             | `implemented` | Graph-driven node pipeline and multi-family function composition                                                                                         | Deeper mechanism controls are not user-configurable                                                 | RD-007                 |
| Tabular realism (mixed type + postprocess hooks)                    | `partial`     | Numeric/categorical converters, E.13 postprocessing, and configurable missingness mechanisms are implemented                                             | High-cardinality/many-class limits remain conservative                                              | RD-006, RD-007         |
| Complexity curriculum scales features/nodes/samples                 | `partial`     | Curriculum mode stages row/split regime                                                                                                                  | Curriculum does not yet stage feature count or graph complexity                                     | RD-006                 |
| Hardware-native performance (Torch + hardware-aware tuning)         | `partial`     | Torch CPU/CUDA/MPS path, hardware detection, coarse profile-based tuning, and benchmark suite                                                            | Hardware-adaptive autotuning is not implemented; parallel/distributed generation is not implemented | RD-010, RD-009         |
| Parallel streaming Parquet sharding                                 | `partial`     | Streaming Parquet writing exists                                                                                                                         | Writing is currently single-process sequential                                                      | RD-009                 |

## Roadmap Items

### RD-001: Ground-Truth DAG Artifact Export

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#44`, `#45`, `#46`, `#47`, `#48`)
- Mission alignment: causal discovery
- Pillar alignment: causal structural integrity
- Goal: persist full adjacency matrix and node assignment lineage as stable dataset artifacts.
- Repo touchpoints: `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/io/parquet_writer.py`, `src/cauchy_generator/types.py`
- Delivered scope:
  - Every generated dataset emits lineage metadata with adjacency + assignment lineage and deterministic seed behavior.
  - Persisted shard outputs rewrite dense adjacency into compact bit-packed artifacts with per-shard index files.
  - Validator enforces versioned dense/compact lineage schemas and compatibility rules.
  - Benchmark profiles report `lineage_guardrails` for export-overhead checks using warn/fail thresholds.
- Completion evidence:
  - Docs and config presets include lineage workflow and benchmark examples.
  - Integration tests cover classification and regression generation + artifact persistence.
  - Existing config defaults remain backward-compatible.

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

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#17` and `#18`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: provide configurable missing-data mechanisms with deterministic seeded behavior and benchmark-time acceptance/runtime guardrails.
- Delivered scope:
  - `DatasetConfig` supports missingness controls (`missing_rate`, mechanism, MAR/MNAR scales).
  - `cauchy-gen generate` supports missingness CLI overrides.
  - Generation path injects deterministic MCAR/MAR/MNAR masks and emits per-bundle missingness metadata.
  - Benchmark profiles emit `missingness_guardrails` including metadata coverage, realized-rate accuracy, and runtime degradation checks.
- Repo touchpoints: `src/cauchy_generator/config.py`, `src/cauchy_generator/sampling/missingness.py`, `src/cauchy_generator/postprocess/postprocess.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/cli.py`, `src/cauchy_generator/bench/suite.py`
- Completion evidence:
  - Config and CLI support opt-in mechanism selection and missing rate controls.
  - Tests validate expected missing-rate and dependency behavior for MCAR/MAR/MNAR.
  - Benchmark summaries include missingness guardrail metrics and status.

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

### RD-010: Hardware-Adaptive Autotuning Beyond Coarse FLOPs Tiers

- Status: `planned`
- Milestone: `Next`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: evolve hardware-aware scaling from static coarse profile tiers to bounded adaptive tuning based on observed throughput/memory behavior.
- Repo touchpoints: `src/cauchy_generator/hardware.py`, `src/cauchy_generator/config.py`, `src/cauchy_generator/cli.py`, `src/cauchy_generator/bench/suite.py`, `src/cauchy_generator/bench/report.py`
- Exit criteria:
  - Adaptive mode improves throughput versus profile baseline on at least one CUDA hardware class without violating memory guardrails.
  - Unknown CUDA devices can run adaptive tuning without relying only on static fallback tiers.
  - Fixed seed + fixed hardware signature reproduces selected tuning settings within declared deterministic behavior.
  - Opt-out mode preserves current profile-only behavior.

## Milestone Board

### Implemented

- RD-001 ground-truth DAG artifact export, completed via `#44`, `#45`, `#46`, `#47`, and `#48`
- RD-003 missingness generation (MCAR/MAR/MNAR), completed via `#17` and `#18`
- RD-008 meta-feature coverage steering

### Now

- RD-006 curriculum complexity scaling
- RD-007 many-class/high-cardinality expansion

### Next

- RD-004 shift-aware SCM generation
- RD-005 robustness stress profiles
- RD-009 parallel/distributed generation
- RD-010 hardware-adaptive autotuning

### Later

- RD-002 interventional and counterfactual generation modes

## Dependencies and Sequencing

- RD-003 and RD-008 are implemented and provide a stronger baseline for remaining realism and robustness roadmap work.
- RD-005 now primarily depends on RD-004 for shift/drift controls, and can build on existing RD-003 missingness infrastructure.
- RD-002 builds on completed RD-001 lineage artifacts for intervention metadata extensions.
- RD-009 should land after interface contracts for RD-001/RD-006 are stable to avoid repeated parallelism refactors.
- RD-010 can build on the current detection/profile baseline while remaining opt-in and benchmark-guarded.

## Guardrails

- All new behavior is opt-in by default.
- Existing config files remain valid unless explicitly versioned.
- Reproducibility expectations are mandatory for every roadmap item.
- Benchmark warn/fail thresholds remain the performance gate.
