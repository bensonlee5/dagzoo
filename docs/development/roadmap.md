# Mission-Aligned Roadmap (2026Q1)

This is the canonical roadmap for `dagzoo`.

It maps the mission and strategic pillars in `README.md` to:

- current implemented capabilities
- known gaps
- prioritized roadmap items with explicit exit criteria
- canonical status/rank sequencing and tracker links

Related docs:

- Decision rubric and go/no-go gates: `docs/development/backlog_decision_rules.md`
- Evidence appendix: `reference/literature_evidence_2026.md`
- System behavior walkthrough: `docs/how-it-works.md`
- Output contract: `docs/output-format.md`

## Status Labels

- `implemented`: available in current code and exposed through config/CLI.
- `partial`: some building blocks exist, but mission-level claim is not fully met.
- `planned`: scoped and prioritized, not implemented.
- `research`: exploratory with higher uncertainty or risk.
- `retired`: historical item that no longer represents active roadmap work.

## Canonical Planning Metadata

`docs/development/roadmap.md` is the single source of truth for planning state. Every active item is tracked here with:

- status and milestone lane
- priority rank
- active Linear issue mapping plus historical GitHub references for completed work
- dependencies and exit criteria

If any other document disagrees with this file, this file is authoritative.

Active execution now lives in the Linear project `dagzoo`
(`https://linear.app/bl-personal/project/dagzoo-4867d49bb182/overview`).
The GitHub-to-Linear migration map is committed at
`reference/linear_issue_map_2026-03-08.json`.

## PFN Utility Prioritization Lens

Roadmap ranking is currently optimized for downstream PFN utility:

- primary: dataset diversity and coverage gains for classification/regression, with time-series tracked explicitly via `RD-013`
- secondary: throughput/cost improvements when they unblock corpus scale rather than directly increase diversity

## Canonical Priority Queue

Lower rank means higher priority. Rank `0` is reserved for completed items retained for traceability.

| Rank | Roadmap ID | Item                                                         | Status      | Milestone | Tracker Links                                                                          |
| ---- | ---------- | ------------------------------------------------------------ | ----------- | --------- | -------------------------------------------------------------------------------------- |
| 0    | RD-001     | Ground-truth DAG artifact export                             | implemented | Now       | `#44 -> #45 -> #46 -> #47 -> #48` (completed)                                          |
| 0    | RD-003     | Missingness generation (MCAR/MAR/MNAR)                       | implemented | Now       | `#15 -> #17 -> #18` (completed)                                                        |
| 0    | RD-008     | Meta-feature coverage steering (retired)                     | retired     | Now       | `#9` (historical)                                                                      |
| 0    | RD-006     | Curriculum complexity scaling (features + graph)             | retired     | Now       | `#49 -> #50 -> #51 -> #90 -> #52 -> #53` (historical), `#142` (rows-based replacement) |
| 0    | RD-004     | Shift-aware SCM generation                                   | implemented | Now       | `#64 -> #72 -> #73 -> #74 -> #75` (completed)                                          |
| 0    | RD-012     | Noise family diversification for synthetic generation        | implemented | Now       | `#24 -> #25 -> #26 -> #27` (completed)                                                 |
| 3    | RD-011     | Mechanism family mix expansion (BNN/GP kernels/interactions) | planned     | Next      | `BL-26 -> BL-27 -> BL-28 -> BL-51 -> BL-52 -> BL-29 -> BL-30`                          |
| 5    | RD-013     | Time-series generation tracks for PFN pretraining            | research    | Next      | `BL-73 -> BL-74 -> BL-75 -> BL-76 -> BL-77`                                            |
| 6    | RD-014     | Run-time bottleneck observability and telemetry              | research    | Next      | `epic TBD; dependency chain TBD`                                                       |
| 7    | RD-007     | Many-class and high-cardinality expansion                    | research    | Next      | `BL-17 -> BL-31 -> (BL-18 -> BL-19 -> BL-20 -> BL-21)`                                 |
| 8    | RD-005     | Robustness stress profiles (hard-task/adversarial regimes)   | research    | Next      | `BL-48 -> BL-59 -> BL-62 -> BL-61 -> BL-60`                                            |
| 9    | RD-009     | Parallel/distributed generation and writing                  | research    | Next      | `BL-49 -> BL-63 -> BL-64 -> BL-65 -> BL-66`                                            |
| 10   | RD-002     | Interventional and counterfactual generation modes           | research    | Later     | `BL-50 -> BL-67 -> BL-68 -> BL-69 -> BL-70`                                            |
| 11   | RD-010     | Hardware-adaptive autotuning beyond coarse FLOPs tiers       | planned     | Later     | `BL-42 -> BL-43 -> BL-44 -> BL-53 -> BL-45 -> BL-54 -> BL-46`                          |

## Current Capability Matrix

| README Mission/Pillar Claim                                         | Current State | Evidence in Repo                                                                                                                                                                                           | Gap                                                                                                                                                                  | Roadmap IDs                            |
| ------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Foundation model pretraining with diverse structural priors         | `partial`     | DAG-based generation, mixed-type conversion, diagnostics extraction/coverage aggregation, configurable missingness, explicit noise-family controls, throughput benchmarks, and opt-in shift/drift controls | Hard-task regimes, mechanism-family controls, many-class expansion, and time-series generation are not implemented end-to-end                                        | RD-004, RD-005, RD-007, RD-011, RD-013 |
| Causal discovery with ground-truth DAGs and interventional datasets | `partial`     | DAG lineage metadata is emitted per dataset and persisted as compact shard-level artifacts with schema validation and benchmark guardrails                                                                 | Interventional/counterfactual generation semantics are not implemented                                                                                               | RD-002                                 |
| Robustness testing with hard tasks, shifts, adversarial regimes     | `partial`     | Basic filtering and diagnostics proxies exist; missingness mechanisms, explicit noise-family controls, and shift/drift controls are implemented with deterministic controls and benchmark guardrails       | No explicit hard-task/adversarial profile suite                                                                                                                      | RD-004, RD-005                         |
| Causal structural integrity (hierarchical dependencies)             | `implemented` | Graph-driven node pipeline and multi-family function composition                                                                                                                                           | Deeper mechanism-family controls are not user-configurable                                                                                                           | RD-007, RD-011                         |
| Tabular realism (mixed type + postprocess hooks)                    | `partial`     | Numeric/categorical converters, E.13 postprocessing, configurable missingness mechanisms, explicit noise-family controls, and fixed-layout generation are implemented                                      | High-cardinality/many-class limits remain conservative                                                                                                               | RD-007                                 |
| PFN task coverage (classification, regression, time-series)         | `partial`     | Classification and regression generation pipelines are fully supported with deterministic seeds and diagnostics/benchmark workflows                                                                        | No time-series generation mode, temporal metadata contract, or temporal diagnostics/guardrails                                                                       | RD-013                                 |
| Staged complexity scaling (features/nodes/samples)                  | `retired`     | Historical staged-complexity implementation (RD-006) has been retired in favor of explicit split sizing and fixed-layout generation                                                                        | Not active                                                                                                                                                           | RD-006                                 |
| Hardware-native performance (Torch + hardware-aware tuning)         | `partial`     | Torch CPU/CUDA/MPS path, hardware detection, coarse profile-based tuning, and benchmark suite                                                                                                              | Stage-level bottleneck attribution/telemetry is not implemented; hardware-adaptive autotuning is not implemented; parallel/distributed generation is not implemented | RD-014, RD-010, RD-009                 |
| Parallel streaming Parquet sharding                                 | `partial`     | Streaming Parquet writing exists                                                                                                                                                                           | Writing is currently single-process sequential                                                                                                                       | RD-009                                 |

## Current Implementation Baseline

This section captures the current implementation baseline. For control/data-flow
walkthroughs, see `docs/how-it-works.md`.

### Source of Truth

- Normative behavior: `reference/TabICLv2.pdf` Appendix E (`E.2`-`E.14`).
- Clarification-only sources:
  - `reference/A Closer Look at TabPFN v2.pdf`
  - `reference/Accurate predictions on small data with a tabular foundation model.pdf`

### Public Interfaces

#### Python API

- `generate_one(config: GeneratorConfig, *, seed: int | None = None, device: str | None = None) -> DatasetBundle`
- `generate_batch(config: GeneratorConfig, *, num_datasets: int, seed: int | None = None, device: str | None = None) -> list[DatasetBundle]`
- `write_packed_parquet_shards_stream(bundles, out_dir, shard_size, compression="zstd")`
- `DatasetConfig` missingness controls:
  - `missing_rate`
  - `missing_mechanism` (`none|mcar|mar|mnar`)
  - `missing_mar_observed_fraction`
  - `missing_mar_logit_scale`
  - `missing_mnar_logit_scale`

#### CLI

- `dagzoo generate --config ... --num-datasets ... --device cuda --seed ...`
- `dagzoo generate --missing-rate ... --missing-mechanism ... --missing-mar-observed-fraction ... --missing-mar-logit-scale ... --missing-mnar-logit-scale ...`
- `dagzoo benchmark --suite standard --preset all --baseline ... --fail-on-regression`

#### Output Contract

See `docs/output-format.md` for the DatasetBundle field spec, Parquet layout,
metadata JSON contract, and DAG lineage schema.

### Runtime Profiles

- `configs/default.yaml`: balanced local development profile.
- `configs/benchmark_cpu.yaml`: CPU benchmark profile.
- `configs/benchmark_cuda_desktop.yaml`: desktop CUDA benchmark profile.
- `configs/benchmark_cuda_h100.yaml`: H100 CUDA benchmark profile.
- `configs/preset_cuda_h100.yaml`: high-throughput datacenter preset.
- `configs/preset_missingness_mcar.yaml`: MCAR missingness preset.
- `configs/preset_missingness_mar.yaml`: MAR missingness preset.
- `configs/preset_missingness_mnar.yaml`: MNAR missingness preset.
- `configs/preset_lineage_benchmark_smoke.yaml`: CPU smoke benchmark preset for
  lineage export guardrail checks.
- Runtime currently applies coarse profile-tier overrides from GPU FLOPS lookup
  and fallback behavior; adaptive autotuning is tracked in RD-010.

### Module Mapping (Appendix E)

- `sampling/correlated.py`: correlated scalar sampler (`E.2`)
- `core/dataset.py`: dataset orchestration entrypoint (`E.3`)
- `core/layout.py`: dataset layout, graph sampling, and node assignments
  (`E.3`, `E.4`)
- `graph/dag_sampler.py`: latent variable DAG sampling (`E.4`)
- `core/node_pipeline.py`: per-node flow (`E.5`)
- `converters/numeric.py`, `converters/categorical.py`: converters (`E.6`)
- `functions/multi.py`: concatenation vs per-parent aggregation (`E.7`)
- `functions/random_functions.py`: NN/tree/discretization/GP/linear/quadratic/EM/product (`E.8`)
- `functions/activations.py`: fixed + parametric activations (`E.9`)
- `linalg/random_matrices.py`: five matrix families and postprocessing (`E.10`)
- `sampling/random_weights.py`: positive normalized weights (`E.11`)
- `sampling/random_points.py`: base distributions + random function transform
  (`E.12`)
- `postprocess/postprocess.py`: cleanup, scaling, class/index permutation
  (`E.13`)
- `filtering/extra_trees_filter.py`: CPU ExtraTrees OOB filter (`E.14`)

### Performance Strategy

1. Current generator path runs Torch on all devices (CPU/CUDA/MPS); diagnostics
   extraction converts bundles to CPU before computing metrics (see
   `docs/how-it-works.md` for diagnostics data flow).
1. Keep kernels batch-oriented with vectorized torch operations and avoid Python
   loops in inner math paths.
1. Use optional filtering (`E.14`) behind config flags to avoid CPU bottlenecks
   in throughput benchmarks.
1. Profile with `bench/throughput.py` and track JSON baseline regressions by
   preset.
1. Missingness-enabled benchmark runs include acceptance/runtime guardrails
   against missingness-off controls.
1. Benchmark profile summaries include lineage-export persistence overhead
   guardrails (`lineage_guardrails`) against lineage-stripped control
   persistence runs.
1. Next hardware-aware step is bounded adaptive autotuning with explicit
   telemetry/guardrails (RD-010).
1. Next roadmap step for throughput is controlled multi-worker execution
   (RD-009) while preserving seeded behavior.

### Reproducibility Strategy

1. Global run seed -> per-dataset seed -> per-component derived seeds.
1. Central RNG utilities provide deterministic seed derivation and
   `torch.Generator` helpers.
1. Document expected backend variation (best effort, not strict bitwise
   determinism).

### Validation and Benchmarks

#### Correctness

- Unit invariants for ranges, shapes, DAG validity, converter class ranges, and
  matrix normalization.
- Unit/integration coverage for missingness mask invariants, deterministic
  behavior, and end-to-end metadata emission.
- Integration tests for end-to-end classification/regression paths.

#### Reproducibility

- Fixed seed should reproduce metadata exactly and numeric outputs within
  tolerance.

#### Performance

- Benchmark suites: `smoke`, `standard`, `full`.
- Artifacts: JSON + Markdown summaries under
  `benchmarks/results/<timestamp>/`.
- Soft regression gate: warn at configurable threshold, fail only on severe
  regression with `--fail-on-regression`.

## Roadmap Items

### RD-001: Ground-Truth DAG Artifact Export

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#44`, `#45`, `#46`, `#47`, `#48`)
- Mission alignment: causal discovery
- Pillar alignment: causal structural integrity
- Goal: persist full adjacency matrix and node assignment lineage as stable dataset artifacts.
- Repo touchpoints: `src/dagzoo/core/dataset.py`, `src/dagzoo/core/layout.py`, `src/dagzoo/io/parquet_writer.py`, `src/dagzoo/types.py`
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
- Linear tracking: epic `BL-50`; dependency chain `BL-67 -> BL-68 -> BL-69 -> BL-70`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/cli.py`
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
  - `DatasetConfig` supports missingness controls (`missing_rate`, mechanism, MAR/MNAR scales). See [docs/how-it-works.md](../how-it-works.md) for MCAR/MAR/MNAR mechanism definitions.
  - `dagzoo generate` supports missingness CLI overrides.
  - Generation path injects deterministic missingness masks and emits per-bundle metadata.
  - Benchmark profiles emit `missingness_guardrails` including metadata coverage, realized-rate accuracy, and runtime degradation checks.
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/sampling/missingness.py`, `src/dagzoo/postprocess/postprocess.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/cli.py`, `src/dagzoo/bench/suite.py`
- Completion evidence:
  - Config and CLI support opt-in mechanism selection and missing rate controls.
  - Tests validate expected missing-rate and dependency behavior.
  - Benchmark summaries include missingness guardrail metrics and status.

### RD-004: Shift-Aware SCM Generation

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#64`, `#72`, `#73`, `#74`, and `#75`)
- Mission alignment: robustness testing, causal discovery
- Pillar alignment: causal structural integrity, tabular realism
- Goal: introduce controlled distribution-shift/drift modes in graph and mechanism sampling.
- GitHub tracking: epic `#64`; dependency chain `#72 -> #73 -> #74 -> #75`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/core/shift.py`, `src/dagzoo/diagnostics/`, `src/dagzoo/bench/`, `configs/preset_shift_*.yaml`
- Delivered scope:
  - Shift controls are integrated into graph/mechanism/noise sampling with deterministic seeded behavior.
  - Per-bundle metadata and diagnostics expose resolved shift settings and observability signals.
  - Discoverable shift presets are available for generation and benchmark smoke workflows.
  - Benchmark profiles emit `shift_guardrails` with runtime, metadata-coverage, and directional checks against shift-disabled controls.
- Completion evidence:
  - Shift workflows are runnable directly from preset configs and documented in user-facing guides.
  - Integration tests cover shift metadata/diagnostics propagation and preset/CLI execution paths.
  - Benchmark summaries include `shift_guardrails` alongside existing guardrail families.

### RD-005: Robustness Stress Profiles (Hard-Task/Adversarial Regimes)

- Status: `research`
- Milestone: `Next`
- Mission alignment: robustness testing
- Pillar alignment: tabular realism
- Goal: define reproducible stress presets (low-SNR, class imbalance, harder interactions).
- Linear tracking: epic `BL-48`; dependency chain `BL-59 -> BL-62 -> BL-61 -> BL-60`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/functions/random_functions.py`, `src/dagzoo/postprocess/postprocess.py`, `src/dagzoo/bench/`
- Exit criteria:
  - Presets are selectable via config/CLI and remain opt-in.
  - Benchmarks and diagnostics confirm regimes differ from baseline in intended directions.
  - Reproducibility tests pass for fixed seed runs.

### RD-006: Staged Complexity Scaling (Features + Graph)

- Status: `retired`
- Milestone: `Now` (completed via epics/issues `#50`, `#51`, `#90`, `#52`, `#53`)
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism
- Goal: historical; staged complexity controls have been removed in favor of
  explicit split sizing and fixed-layout generation.
- GitHub tracking: epic `#49`; dependency chain `#50 -> #51 -> #90 -> #52 -> #53`
- Repo touchpoints (historical): `src/dagzoo/config.py`,
  `src/dagzoo/core/dataset.py`

### RD-007: Many-Class and High-Cardinality Expansion

- Status: `research`
- Milestone: `Next`
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism, causal structural integrity
- Goal: assess feasibility and, only if viable, raise practical class/cardinality limits while preserving filter quality.
- Current rollout envelope (`BL-18`): enforce `dataset.n_classes_max <= 32` as the narrow-go safety cap until follow-on hardening (`BL-19`-`BL-21`) lands.
- `BL-20` scope: class-aware RF filter thresholding plus metadata diagnostics so many-class accept/reject behavior remains interpretable.
- Linear tracking: epic `BL-17`; research gate `BL-31`; conditional chain `BL-18 -> BL-19 -> BL-20 -> BL-21`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/converters/categorical.py`, `src/dagzoo/filtering/extra_trees_filter.py`
- Exit criteria:
  - Feasibility report produces explicit `go` / `narrow-go` / `no-go` decision under predefined thresholds.
  - If `go` or `narrow-go`: implementation chain (`BL-18`-`BL-21`) lands with backward-compatible defaults.
  - If `no-go`: roadmap records deferral or narrowed scope with rationale.

### RD-008: Meta-Feature Coverage Steering

- Status: `retired`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism
- Goal: feature retired; diagnostics target bands remain as reporting-only metadata.
- Repo touchpoints: `src/dagzoo/diagnostics/coverage.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/cli.py`
- Notes:
  - Steering selection logic was removed to simplify generation semantics.
  - `diagnostics.meta_feature_targets` remains supported for coverage summaries.

### RD-009: Parallel and Distributed Generation/Writing

- Status: `research`
- Milestone: `Next`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: support multi-worker generation and shard writing with deterministic seed partitioning.
- Linear tracking: epic `BL-49`; dependency chain `BL-63 -> BL-64 -> BL-65 -> BL-66`
- Repo touchpoints: `src/dagzoo/core/dataset.py`, `src/dagzoo/io/parquet_writer.py`, `src/dagzoo/cli.py`
- Planned sequencing:
  - T1: internal GPU micro-batching for dataset generation hot paths (no new user toggles in first rollout).
  - T2: bounded producer/consumer overlap between generation and shard writing.
  - T3: multi-worker seed partitioning and deterministic shard assignment.
- Exit criteria:
  - Worker-aware config/API is backward-compatible and opt-in.
  - Multi-worker mode matches single-worker outputs for fixed seed equivalence checks.
  - Throughput improves on supported hardware without violating fail threshold regressions.

### RD-010: Hardware-Adaptive Autotuning Beyond Coarse FLOPs Tiers

- Status: `planned`
- Milestone: `Later`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: evolve hardware-aware scaling from static coarse profile tiers to bounded adaptive tuning based on observed throughput/memory behavior when throughput/cost becomes a practical bottleneck.
- Linear tracking: epic `BL-42`; dependency chain `BL-43 -> BL-44 -> BL-53 -> BL-45 -> BL-54 -> BL-46`
- Repo touchpoints: `src/dagzoo/hardware.py`, `src/dagzoo/config.py`, `src/dagzoo/cli.py`, `src/dagzoo/bench/suite.py`, `src/dagzoo/bench/report.py`
- Exit criteria:
  - Adaptive mode improves throughput versus profile baseline on at least one CUDA hardware class without violating memory guardrails.
  - Unknown CUDA devices can run adaptive tuning without relying only on static fallback tiers.
  - Fixed seed + fixed hardware signature reproduces selected tuning settings within declared deterministic behavior.
  - Opt-out mode preserves current profile-only behavior.

### RD-011: Mechanism Family Mix Expansion (BNN/GP Kernels/Interactions)

- Status: `planned`
- Milestone: `Next`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: causal structural integrity, tabular realism
- Goal: add explicit mechanism-family mix controls and broaden function families (including BNN/GP-kernel/interactions) to increase structural diversity that materially affects generated datasets for PFN pretraining.
- Linear tracking: epic `BL-26`; dependency chain `BL-27 -> BL-28 -> BL-51 -> BL-52 -> BL-29 -> BL-30`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/functions/random_functions.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/bench/suite.py`
- Exit criteria:
  - Config supports explicit mechanism-family mix controls with backward-compatible defaults.
  - Expanded mechanism families are selectable and covered by unit/integration tests.
  - Metadata/diagnostics expose effective family mix in generated bundles and benchmark runs.
  - Coverage metrics show measurable diversity gain versus baseline mechanism sampling.
  - Presets/docs/bench guardrails exist for contributors.

### RD-012: Noise Family Diversification for Synthetic Generation

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#24`, `#25`, `#26`, `#27`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: complete lean, low-complexity integration of explicit noise-family controls and mixtures to diversify residual/noise behavior without broad generator refactors.
- GitHub tracking: epic `#24`; dependency chain `#25 -> #26 -> #27` (completed)
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/sampling/`, `src/dagzoo/core/dataset.py`, `src/dagzoo/bench/suite.py`
- Delivered scope:
  - Config supports `gaussian`, `laplace`, `student_t`, and `mixture` families with safety validation.
  - Runtime sampling and generation metadata report requested/effective family settings.
  - Benchmark guardrails include metadata validation and runtime delta checks versus gaussian-noise controls.
  - Presets/docs/tests cover family-specific generation and benchmark workflows.
- Completion evidence:
  - All delivery issues in the chain are closed (`#25`, `#26`, `#27`).
  - End-user docs include noise workflow guidance and benchmark examples.

### RD-013: Time-Series Generation Tracks for PFN Pretraining

- Status: `research`
- Milestone: `Next`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: add an opt-in temporal generation track for sequence datasets so PFN pretraining workflows cover classification/regression/time-series under one reproducible generator framework.
- Linear tracking: epic `BL-73`; dependency chain `BL-74 -> BL-75 -> BL-76 -> BL-77`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/diagnostics/`, `src/dagzoo/bench/`, `docs/`
- Exit criteria:
  - Temporal mode is opt-in and backward-compatible (`off` by default).
  - Fixed seed + config reproducibility is preserved for temporal generation.
  - Sequence metadata/diagnostics contracts are emitted and test-covered.
  - Presets/docs/bench guardrails provide discoverable temporal workflows.

### RD-014: Run-Time Bottleneck Observability and Telemetry

- Status: `research`
- Milestone: `Next`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: hardware-native performance
- Goal: define and validate an opt-in observability path that attributes wall-time/memory bottlenecks to generation and benchmark stages across CPU/CUDA/MPS runs.
- GitHub tracking: epic TBD; dependency chain TBD
- Repo touchpoints: `src/dagzoo/bench/`, `src/dagzoo/core/dataset.py`, `src/dagzoo/cli.py`, `src/dagzoo/hardware.py`, `docs/`
- Exit criteria:
  - Bottleneck observability mode is opt-in and backward-compatible (`off` by default).
  - Benchmark/report artifacts expose stage-level timing and bottleneck attribution instead of only total runtime.
  - Observability outputs are stable enough for CI comparison and regression triage.
  - Instrumentation overhead is bounded, documented, and does not invalidate existing benchmark guardrails.
  - User-facing docs describe how to enable and interpret bottleneck outputs.

## Milestone Board

### Implemented

- RD-001 ground-truth DAG artifact export, completed via `#44`, `#45`, `#46`, `#47`, and `#48`
- RD-003 missingness generation (MCAR/MAR/MNAR), completed via `#17` and `#18`
- RD-008 meta-feature coverage steering (retired)
- RD-006 staged complexity scaling (retired), completed via `#49`, `#50`, `#51`, `#90`, `#52`, and `#53`
- RD-004 shift-aware SCM generation, completed via `#64`, `#72`, `#73`, `#74`, and `#75`
- RD-012 noise family diversification, completed via `#24`, `#25`, `#26`, and `#27`

### Now

- No active `Now` lane items; implemented items are tracked above for traceability.

### Next

- RD-011 mechanism family mix expansion
- RD-013 time-series generation tracks
- RD-014 bottleneck observability and telemetry
- RD-007 many-class/high-cardinality expansion (research-gated)
- RD-005 robustness stress profiles
- RD-009 parallel/distributed generation

### Later

- RD-002 interventional and counterfactual generation modes
- RD-010 hardware-adaptive autotuning

## Dependencies and Sequencing

- RD-003 and RD-008 are implemented and provide a stronger baseline for remaining realism and robustness roadmap work.
- RD-007 is explicitly research-gated by `BL-17` / `BL-31` before conditional rollout issues (`BL-18`-`BL-21`) proceed.
- RD-011 is the primary near-term diversity lever and is tracked in epic `BL-26`.
- RD-012 is implemented via epic `#24` and provides explicit noise-family controls for stress-profile composition.
- RD-013 introduces sequence/temporal generation coverage and is tracked in epic `BL-73`.
- RD-014 should precede or run in parallel with RD-010 so adaptive tuning decisions are guided by stage-level bottleneck evidence.
- RD-014 should inform RD-009 worker/sharding design by identifying true single-process bottlenecks first.
- RD-014 remains observability-only and must not change default runtime behavior when disabled.
- RD-005 now primarily depends on RD-004 for shift/drift controls, and can build on existing RD-003 missingness infrastructure.
- RD-005 can consume implemented RD-012 noise-family controls while RD-011 mechanism-family controls remain in flight.
- RD-004/RD-005/RD-009/RD-002 now have explicit epic trackers and PR-scoped delivery chains.
- RD-002 builds on completed RD-001 lineage artifacts for intervention metadata extensions.
- RD-013 should define temporal schema + metadata contracts before adding temporal stress-profile variants.
- RD-009 should land after interface contracts for RD-001/RD-006 are stable to avoid repeated parallelism refactors.
- RD-010 remains opt-in and benchmark-guarded, but is sequenced later because it is throughput/cost oriented rather than a direct diversity expansion.

## Guardrails

- All new behavior is opt-in by default.
- Existing config files remain valid unless explicitly versioned.
- Reproducibility expectations are mandatory for every roadmap item.
- Benchmark warn/fail thresholds remain the performance gate.
