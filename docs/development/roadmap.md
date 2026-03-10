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

- primary: filtered dataset throughput and acceptance yield while preserving or improving effective diversity for classification/regression corpora
- secondary: concise request-file handoff so downstream repos such as `tab-foundry` can request corpora without depending on the full internal config surface
- deferred: closed-loop feedback from downstream model predictions back into generation policy after the one-way handoff is stable

## Canonical Priority Queue

Lower rank means higher priority. Rank `0` is reserved for completed items retained for traceability.

| Rank | Roadmap ID | Item                                                                   | Status      | Milestone | Tracker Links                                                               |
| ---- | ---------- | ---------------------------------------------------------------------- | ----------- | --------- | --------------------------------------------------------------------------- |
| 0    | RD-001     | Ground-truth DAG artifact export                                       | implemented | Now       | `#44 -> #45 -> #46 -> #47 -> #48` (completed)                               |
| 0    | RD-003     | Missingness generation (MCAR/MAR/MNAR)                                 | implemented | Now       | `#15 -> #17 -> #18` (completed)                                             |
| 0    | RD-004     | Shift-aware SCM generation                                             | implemented | Now       | `#64 -> #72 -> #73 -> #74 -> #75` (completed)                               |
| 0    | RD-006     | Curriculum complexity scaling (features + graph)                       | retired     | Now       | `#49 -> #50 -> #51 -> #90 -> #52 -> #53` (historical), `#142` (replacement) |
| 0    | RD-007     | Many-class rollout envelope (`<=32` classes)                           | implemented | Now       | `BL-17 -> BL-18 -> BL-19 -> BL-20 -> BL-21` (completed), `BL-31` (closure)  |
| 0    | RD-008     | Meta-feature coverage steering (retired)                               | retired     | Now       | `#9` (historical)                                                           |
| 0    | RD-012     | Noise family diversification for synthetic generation                  | implemented | Now       | `#24 -> #25 -> #26 -> #27` (completed)                                      |
| 0    | RD-014     | Stage-level benchmark observability and telemetry                      | implemented | Now       | `BL-82` (completed historical delivery under `BL-49`)                       |
| 0    | RD-015     | Keyed RNG semantic reproducibility                                     | implemented | Now       | `BL-90 -> BL-133 -> BL-134 -> BL-135 -> BL-136 -> BL-137`                   |
| 1    | RD-009     | Filtered dataset throughput and deferred-filter scaling                | planned     | Now       | `BL-49 -> BL-148 -> BL-149 -> BL-150 -> (BL-84 -> BL-85 later)`             |
| 2    | RD-016     | Concise request-file contract and one-way `tab-foundry` handoff        | planned     | Now       | `BL-143 -> BL-144 -> BL-145 -> BL-146 -> BL-147`                            |
| 3    | RD-011     | Mechanism diversity expansion with measurable effective-diversity gain | planned     | Now       | `BL-26 -> BL-28 -> BL-51 -> BL-52 -> BL-29 -> BL-30`                        |
| 8    | RD-005     | Robustness stress profiles (hard-task/adversarial regimes)             | research    | Later     | `BL-48 -> BL-59 -> BL-62 -> BL-61 -> BL-60`                                 |
| 9    | RD-013     | Time-series generation tracks for PFN pretraining                      | research    | Later     | `BL-73 -> BL-74 -> BL-75 -> BL-76 -> BL-77`                                 |
| 10   | RD-002     | Interventional and counterfactual generation modes                     | research    | Later     | `BL-50 -> BL-67 -> BL-68 -> BL-69 -> BL-70`                                 |
| 11   | RD-010     | Hardware-adaptive autotuning beyond coarse FLOPs tiers                 | planned     | Later     | `BL-42 -> BL-43 -> BL-44 -> BL-53 -> BL-45 -> BL-54 -> BL-46`               |

## Current Capability Matrix

| README Mission/Pillar Claim                                         | Current State | Evidence in Repo                                                                                                                                                                                                         | Gap                                                                                                                                          | Roadmap IDs                    |
| ------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Foundation model pretraining with diverse structural priors         | `partial`     | Canonical fixed-layout generation, `dataset.rows`, deferred filtering, effective-diversity audits, diagnostics coverage aggregation, explicit noise/shift controls, and stage-level throughput metrics are implemented   | Concise downstream handoff, new mechanism families with measurable diversity lift, and time-series generation are not implemented end-to-end | RD-009, RD-011, RD-013, RD-016 |
| Causal discovery with ground-truth DAGs and interventional datasets | `partial`     | DAG lineage metadata is emitted per dataset and persisted as compact shard-level artifacts with schema validation and benchmark guardrails                                                                               | Interventional/counterfactual generation semantics are not implemented                                                                       | RD-002                         |
| Robustness testing with hard tasks, shifts, adversarial regimes     | `partial`     | Deferred filtering, diagnostics proxies, missingness mechanisms, explicit noise-family controls, and shift/drift controls are implemented with deterministic controls and benchmark guardrails                           | No explicit hard-task/adversarial profile suite                                                                                              | RD-004, RD-005                 |
| Causal structural integrity (hierarchical dependencies)             | `implemented` | Graph-driven node pipeline, canonical fixed-layout execution, DAG lineage artifacts, and keyed RNG semantic reproducibility are implemented                                                                              | Remaining work is diversity-oriented expansion, not structural correctness                                                                   | RD-011                         |
| Tabular realism (mixed type + postprocess hooks)                    | `partial`     | Numeric/categorical converters, postprocess hooks, many-class rollout within the current `<=32` class envelope, configurable missingness, explicit noise families, and canonical fixed-layout generation are implemented | Broader mechanism diversity and optional future expansion beyond the current many-class envelope remain deferred                             | RD-007, RD-011                 |
| PFN task coverage (classification, regression, time-series)         | `partial`     | Classification and regression generation pipelines are fully supported with deterministic seeds, keyed replay metadata, and benchmark workflows                                                                          | No time-series generation mode, temporal metadata contract, or temporal diagnostics/guardrails                                               | RD-013                         |
| Staged complexity scaling (features/nodes/samples)                  | `retired`     | Historical staged-complexity implementation (RD-006) has been retired in favor of explicit split sizing and fixed-layout generation                                                                                      | Not active                                                                                                                                   | RD-006                         |
| Hardware-native performance (Torch + hardware-aware tuning)         | `partial`     | Torch CPU/CUDA/MPS path, hardware detection, coarse profile-based tuning, benchmark suite, and stage-level generation/write/filter metrics are implemented                                                               | Filtered-corpus throughput is still bottlenecked by deferred filtering, and hardware-adaptive autotuning is not implemented                  | RD-009, RD-010, RD-014         |
| Downstream synthetic-corpus handoff                                 | `partial`     | Effective config artifacts, deferred-filter manifests, and reproducible output directories already exist                                                                                                                 | No concise request-file contract, handoff manifest, or documented `dagzoo -> tab-foundry` smoke workflow                                     | RD-009, RD-016                 |

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
- `generate_batch_iter(config: GeneratorConfig, *, num_datasets: int, seed: int | None = None, device: str | None = None) -> Iterator[DatasetBundle]`
- `write_packed_parquet_shards_stream(bundles, out_dir, shard_size, compression="zstd")`
- `DatasetConfig` missingness controls:
  - `missing_rate`
  - `missing_mechanism` (`none|mcar|mar|mnar`)
  - `missing_mar_observed_fraction`
  - `missing_mar_logit_scale`
  - `missing_mnar_logit_scale`

#### CLI

- `dagzoo generate --config ... --rows ... --num-datasets ... --device cuda --seed ...`
- `dagzoo generate --missing-rate ... --missing-mechanism ... --missing-mar-observed-fraction ... --missing-mar-logit-scale ... --missing-mnar-logit-scale ...`
- `dagzoo filter --in <shard_dir> --out <filter_dir> [--curated-out <accepted_shards_dir>] [--n-jobs ...]`
- `dagzoo benchmark --suite standard --preset all --baseline ... --fail-on-regression`
- `dagzoo effective-diversity --config ... --suite ... --baseline ... --fail-on-regression`

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
  plus explicit hardware-policy transforms; adaptive autotuning is tracked in
  RD-010.

### Module Mapping (Appendix E)

- `sampling/correlated.py`: correlated scalar sampler (`E.2`)
- `core/dataset.py`: dataset orchestration entrypoint (`E.3`)
- `core/layout.py`: dataset layout, graph sampling, and node assignments
  (`E.3`, `E.4`)
- `graph/dag_sampler.py`: latent variable DAG sampling (`E.4`)
- `core/fixed_layout_batched.py`: typed plan sampling and batched node execution
- `core/node_pipeline.py`: isolated node-plan helper used by tests and microbenchmarks
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
1. Next roadmap step for throughput is filtered-corpus scaling: improve
   deferred-filter throughput and acceptance yield first, then revisit writer
   overlap or accelerator micro-batching only if filter is no longer the
   dominant bottleneck (RD-009).

### Reproducibility Strategy

1. Global run seed -> per-dataset seed -> per-component derived seeds.
1. `KeyedRng` provides the semantic RNG contract for generation/runtime code,
   while preserving deterministic child-seed derivation for replay and
   compatibility workflows.
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
- Milestone: `Later`
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

### RD-007: Many-Class Rollout Envelope (`<=32` Classes)

- Status: `implemented`
- Milestone: `Now` (completed via `BL-17`, `BL-18`, `BL-19`, `BL-20`, `BL-21`, with `BL-31` retained as closure note)
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism, causal structural integrity
- Goal: land a stable many-class rollout envelope while keeping filter behavior and label handling interpretable.
- Linear tracking: historical epic `BL-17`; completion chain `BL-18 -> BL-19 -> BL-20 -> BL-21`; closure note `BL-31`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/converters/categorical.py`, `src/dagzoo/filtering/extra_trees_filter.py`, `docs/features/many-class.md`
- Delivered scope:
  - `dataset.n_classes_max <= 32` is enforced as the supported rollout envelope.
  - Converter and postprocess paths are hardened for the current many-class range.
  - Deferred filter diagnostics make accept/reject behavior interpretable within the current envelope.
  - Discoverable many-class generation and benchmark smoke workflows are documented.
- Completion evidence:
  - Config validation and user docs align on the supported envelope.
  - Integration and CLI tests cover the current many-class workflows.
  - Broader expansion beyond the current envelope is not an active roadmap item.

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

### RD-009: Filtered Dataset Throughput and Deferred-Filter Scaling

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: improve accepted-corpus throughput on the canonical `generate -> filter` pipeline while preserving or improving effective diversity.
- Linear tracking: epic `BL-49`; active chain `BL-148 -> BL-149 -> BL-150`; lower-priority follow-ons `BL-84 -> BL-85`; historical completed work `BL-82`, `BL-83`, `BL-86`
- Repo touchpoints: `src/dagzoo/filtering/deferred_filter.py`, `src/dagzoo/bench/stage_metrics.py`, `src/dagzoo/bench/suite.py`, `src/dagzoo/cli.py`, `src/dagzoo/io/parquet_writer.py`
- Planned sequencing:
  - T1: optimize deferred-filter replay throughput on canonical shard metadata.
  - T2: make filter-enabled benchmark runs report accepted-corpus rate and rejection yield as first-class outputs.
  - T3: add diversity-aware filter calibration guardrails so throughput work does not collapse functional diversity.
  - T4: revisit writer overlap or accelerator micro-batching only if filter is no longer the dominant bottleneck.
- Exit criteria:
  - Filter-enabled benchmark flows show a measured filtered-corpus throughput improvement on at least one canonical CPU workload.
  - Benchmark artifacts surface generation, write, and filter stage rates plus acceptance-yield signals.
  - No new public worker-count or worker-index surface is added.
  - Diversity guardrails are available for filter-tuning work.

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

### RD-011: Mechanism Diversity Expansion With Measurable Effective-Diversity Gain

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: causal structural integrity, tabular realism
- Goal: build on the already-landed mechanism-family mix surface by adding genuinely new families/variants and proving they increase emitted diversity without unacceptable throughput or filter-yield regressions.
- Linear tracking: epic `BL-26`; remaining chain `BL-28 -> BL-51 -> BL-52 -> BL-29 -> BL-30`; completed plumbing `BL-27`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/functions/random_functions.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/bench/suite.py`
- Exit criteria:
  - New mechanism families or variants are selectable through the existing family-mix surface.
  - Expanded mechanism families are selectable and covered by unit/integration tests.
  - Metadata, diagnostics, or diversity-audit artifacts expose realized mechanism-family coverage beyond config echo.
  - Effective-diversity or equivalent audit outputs show measurable diversity gain versus the current baseline.
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
- Milestone: `Later`
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

### RD-014: Stage-Level Benchmark Observability and Telemetry

- Status: `implemented`
- Milestone: `Now` (completed historically via `BL-82`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: hardware-native performance
- Goal: expose stage-level throughput telemetry for canonical benchmark runs so bottlenecks can be attributed before further runtime work is prioritized.
- Linear tracking: historical delivery `BL-82` under `BL-49`
- Repo touchpoints: `src/dagzoo/bench/stage_metrics.py`, `src/dagzoo/bench/suite.py`, `src/dagzoo/cli.py`, `docs/features/benchmark-guardrails.md`
- Delivered scope:
  - Benchmark/report artifacts expose generation, write, and optional filter stage throughput instead of only total runtime.
  - Filter rejection and retry signals are surfaced for filter-enabled benchmark runs.
  - Stage metrics are stable enough to use in regression triage and backlog prioritization.
- Completion evidence:
  - Benchmark CLI summaries print stage-level throughput fields.
  - Benchmark JSON and Markdown artifacts persist stage-level metrics.
  - Subsequent throughput planning now relies on those stage metrics rather than inferred bottlenecks.

### RD-015: Keyed RNG Semantic Reproducibility

- Status: `implemented`
- Milestone: `Now` (completed via `BL-90`, `BL-133`, `BL-134`, `BL-135`, `BL-136`, and `BL-137`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: causal structural integrity, hardware-native performance
- Goal: replace order-coupled ambient RNG usage with keyed semantic namespaces so regrouping, retries, and scalar-vs-batched path changes preserve the intended reproducibility contract.
- Linear tracking: epic `BL-90`; dependency chain `BL-133 -> BL-134 -> BL-135 -> BL-136 -> BL-137`
- Repo touchpoints: `src/dagzoo/rng.py`, `src/dagzoo/core/`, `src/dagzoo/postprocess/`, `src/dagzoo/sampling/`, `src/dagzoo/bench/`, `docs/`
- Delivered scope:
  - Core generation/runtime randomness is keyed by semantic namespace rather than draw order or offset-only coupling.
  - `KeyedRng` is the semantic RNG surface used by runtime code.
  - Retrying one stage does not perturb sibling-stage randomness.
  - Scalar and batched typed-plan execution preserve semantic equivalence under the keyed contract.
  - Reproducibility docs describe `keyed_replay` and the intended non-goals of the contract.
- Completion evidence:
  - Repo code and tests exercise `KeyedRng` across generation, postprocess, missingness, and benchmark flows.
  - Generated bundle metadata includes keyed replay paths for canonical batch replay.
  - The Linear implementation chain is closed and moved to historical traceability.

### RD-016: Concise Request-File Contract and One-Way `tab-foundry` Handoff

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance, tabular realism
- Goal: let downstream repos specify what to generate and how many datasets to generate through a concise request file, then consume a filtered corpus plus machine-readable handoff metadata without depending on the full internal config surface.
- Linear tracking: epic `BL-143`; dependency chain `BL-144 -> BL-145 -> BL-146 -> BL-147`
- Repo touchpoints: `src/dagzoo/config.py`, `src/dagzoo/cli.py`, `src/dagzoo/filtering/deferred_filter.py`, `docs/`, downstream `tab-foundry`
- Exit criteria:
  - A versioned request-file schema captures a useful generation intent with a small field set.
  - Request execution resolves into canonical `generate -> filter` runs with effective-config traceability.
  - A machine-readable handoff manifest exposes filtered-corpus paths, accepted/rejected counts, stage-throughput context, and diversity-artifact paths.
  - Docs include at least one reproducible `dagzoo -> tab-foundry` smoke workflow.
  - Closed-loop feedback ingestion remains explicitly out of scope until the one-way handoff is stable.

## Milestone Board

### Implemented

- RD-001 ground-truth DAG artifact export, completed via `#44`, `#45`, `#46`, `#47`, and `#48`
- RD-003 missingness generation (MCAR/MAR/MNAR), completed via `#17` and `#18`
- RD-004 shift-aware SCM generation, completed via `#64`, `#72`, `#73`, `#74`, and `#75`
- RD-007 many-class rollout envelope, completed via `BL-17`, `BL-18`, `BL-19`, `BL-20`, and `BL-21`
- RD-008 meta-feature coverage steering (retired)
- RD-006 staged complexity scaling (retired), completed via `#49`, `#50`, `#51`, `#90`, `#52`, and `#53`
- RD-012 noise family diversification, completed via `#24`, `#25`, `#26`, and `#27`
- RD-014 stage-level benchmark observability and telemetry, completed via `BL-82`
- RD-015 keyed RNG semantic reproducibility, completed via `BL-90`, `BL-133`, `BL-134`, `BL-135`, `BL-136`, and `BL-137`

### Now

- RD-009 filtered dataset throughput and deferred-filter scaling
- RD-016 concise request-file contract and one-way `tab-foundry` handoff
- RD-011 mechanism diversity expansion with measurable effective-diversity gain

### Later

- RD-005 robustness stress profiles
- RD-013 time-series generation tracks
- RD-002 interventional and counterfactual generation modes
- RD-010 hardware-adaptive autotuning

## Dependencies and Sequencing

- RD-014 is implemented and provides the stage-level evidence that now justifies centering RD-009 on deferred-filter throughput rather than speculative public parallelism.
- RD-015 is implemented and provides the semantic RNG contract that active throughput or handoff work must preserve.
- RD-009 is the immediate bottleneck-oriented priority because deferred filtering is slower than generation on current filter-enabled benchmark probes.
- RD-016 depends on the canonical `generate -> filter` pipeline from RD-009 and should not introduce a parallel configuration surface that the repo has already removed.
- RD-011 is the active diversity lever, but it should be evaluated against RD-009's acceptance-yield and diversity guardrails rather than in isolation.
- RD-012 is implemented and provides explicit noise-family controls that RD-005 can consume later for stress-profile composition.
- RD-005 now primarily depends on RD-004, RD-003, and the filter/density observability already in the repo.
- RD-013 remains later because the near-term downstream contract is one-way tabular corpus handoff, not temporal generation.
- RD-002 builds on completed RD-001 lineage artifacts for intervention metadata extensions.
- RD-010 remains opt-in and benchmark-guarded, but is sequenced later because filtered-corpus throughput and downstream handoff are more urgent than adaptive tuning.

## Guardrails

- All new behavior is opt-in by default.
- Existing config files remain valid unless explicitly versioned.
- Reproducibility expectations are mandatory for every roadmap item.
- Benchmark warn/fail thresholds remain the performance gate.
