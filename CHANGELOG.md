# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-03-03

### Changed

- Added internal typed layout contracts in `core/layout_types.py`:
  `LayoutPayload`, `FeatureType`, `ConverterKind`, `MechanismFamily`, and
  `AggregationKind`.
- Updated generation/layout/metadata wiring to use typed layout payloads
  instead of untyped `dict[str, Any]` signatures.
- Node pipeline now exposes explicit feature-key parsing (`parse_feature_key`)
  and uses helper boundaries for latent-dimension resolution and latent padding.
- Multi-parent function composition now supports optional explicit aggregation
  selection while preserving random-selection default behavior.
- Clarified DAG adjacency direction contract in generation/docs:
  `adjacency[src, dst]` means `src -> dst`; parent lookup is column-based.
- Standardized seed derivation callsites in `core/dataset.py` behind named
  helper functions for attempts, node-spec sampling, and split permutations.
- Internal naming cleanup across layout/dataset/node codepaths for clearer
  node/feature/dimension semantics.
- No CLI flag changes, no lineage schema version changes, and no dataset
  metadata contract changes.

## [0.3.1] - 2026-03-03

### Changed

- Added a shared config-resolution engine (`core/config_resolution.py`) used by both `generate` and benchmark profile execution paths.
- Added per-run field-level effective config trace artifacts:
  - `generate`: `effective_config_trace.yaml`
  - `benchmark`: `effective_configs/*_trace.yaml`
- Added CLI flags `--print-resolution-trace` for `generate` and `benchmark`.
- Benchmarks now include per-profile `effective_config_trace` payloads in summary output.
- Added canonical precedence documentation: `docs/config-resolution.md`.
- **Breaking:** `dataset.categorical_ratio_min` and `dataset.categorical_ratio_max` now validate strictly in `[0, 1]` at config load time.
- Updated shipped config presets to use in-range categorical ratio bounds (`0.0..1.0`).

## [0.3.0] - 2026-03-03

### Changed

- **Rename: `cauchy-generator` is now `dagsynth`.** Package name, CLI entrypoint, Python import paths, and all internal references updated. The CLI command is now `dagsynth` (was `cauchy-gen`). Python imports change from `from cauchy_generator.…` to `from dagsynth.…`.
- **Breaking:** Lineage schema name changed from `cauchy_generator.dag_lineage` to `dagsynth.dag_lineage`. Existing Parquet files with embedded lineage metadata will fail validation against the new schema name.
- `sample_cauchy_dag` renamed to `sample_dag`; `graph/cauchy_graph.py` renamed to `graph/dag_sampler.py`.
- Noise rollout presets for generate/benchmark smoke workflows (`configs/preset_noise_*.yaml`)
- Noise workflow script wrapper (`scripts/generate-noise.sh`)
- Noise benchmark guardrail reporting (`noise_guardrails`) for runtime delta vs gaussian control and metadata validity
- Noise feature guide (`docs/features/noise.md`) and usage/workflow documentation updates
- Benchmark profile summaries now surface noise guardrail status in CLI and Markdown reports
- **Breaking:** removed `noise.family=legacy`; default noise family is now explicit `gaussian`.
- **Breaking:** removed config/runtime compatibility knobs `runtime.hardware_aware`, CLI `--no-hardware-aware`, and graph aliases `graph.n_nodes_log2_min/max`.
- **Breaking:** `shift.profile` no longer accepts boolean aliases; only explicit string values are allowed.
- Hardware detection and policy are now separated: `hardware.py` handles detection and `hardware_policy.py` handles explicit scaling policy.
- Added hardware policy registry APIs (`apply_hardware_policy`, `register_hardware_policy`, `list_hardware_policies`) with immutable policy application semantics.
- Added CLI `--hardware-policy` and `--print-effective-config` options; generate/benchmark runs now persist resolved effective config artifacts.
- Fixed-layout plan schema now requires a non-null `compatibility_snapshot` payload.

## [0.2.0] - 2026-03-02

### Added

- Shift profile schema, sampling integration, and metadata/diagnostics observability across generation and benchmarks
- Configurable noise family schema/samplers (`legacy`, `gaussian`, `laplace`, `student_t`, `mixture`) with profile/preset coverage
- Many-class generation envelope and class-aware filtering/diagnostics presets for benchmark and generate workflows
- Fixed-layout batch generation APIs with emitted schema stability guarantees
- Curriculum shell runner (`scripts/generate-curriculum.sh`) for staged dataset batches with manifest output
- Additional guidance docs and feature references (`docs/how-it-works.md`, `docs/design-decisions.md`, `docs/glossary.md`, `docs/output-format.md`, `docs/usage-guide.md`)

### Changed

- Dataset generation flow now favors fixed-layout execution paths instead of curriculum staging APIs
- Tiny split/postprocess and DAG sampling steps run on CPU to avoid underutilized GPU kernels
- Benchmark/docs workflows now include shift/noise/many-class/curriculum shell guidance and presets

### Fixed

- Noise sampling stability improvements, including Student-t support on MPS and tighter mixture reproducibility guards
- Fixed-layout compatibility validation hardening for resolved device checks and feature/node bounds
- Curriculum shell manifest accounting/status fixes for aborted and dry-run chunk handling

### Removed

- Legacy steering hard-fail path and associated deprecated hard-fail tests

## [0.1.14] - 2026-02-26

### Added

- Curriculum stage schema/validation and stagewise runtime scaling for feature/node/depth/structure complexity
- Curriculum benchmark presets/guardrails and curriculum metadata diagnostics (`metadata.curriculum.realized_complexity`, `stage_bounds`, `monotonicity_axes`)
- DAG lineage artifact pipeline with compact shard outputs (`adjacency.bitpack.bin`, `adjacency.index.json`) and benchmark lineage guardrails (`lineage_guardrails`)
- Lineage benchmark smoke preset: `configs/preset_lineage_benchmark_smoke.yaml`

### Changed

- Diagnostics extraction now normalizes bundles to CPU and delegates metric computation to `core/steering_metrics.py`, avoiding non-CPU float64 failures (notably MPS)
- Internal refactors removed dataset/benchmark compatibility shims and consolidated duplicated helper/constants wiring across CLI/core modules
- README/roadmap/implementation docs expanded lineage/curriculum workflows and benchmark guardrail guidance

### Fixed

- Curriculum validation hardening across stage bounds, node/depth consistency, singleton DAG handling, and stratified split retry sizing
- Lineage artifact correctness/resilience: assignment alignment with postprocessed features, adjacency type validation, bounded blob descriptors, and robust index flushing on failures
- Benchmark lineage guardrail robustness for non-runtime probe handling and undersized runtime gating

### Removed

- `RuntimeConfig` keys: `generation_engine`, `gpu_name_hint`, `gpu_memory_gb_hint`, `peak_flops_hint`
- `SeedManager.numpy_rng`

### Compatibility

- Configs that still set `runtime.generation_engine` now fail fast during config load (`TypeError`).

## [0.1.13] - 2026-02-24

### Added

- Successful dataset generation now emits versioned DAG lineage payloads (`metadata.lineage`) with adjacency/assignment metadata derived from seeded layout sampling

## [0.1.12] - 2026-02-23

### Added

- Versioned DAG lineage schema validator for `metadata.lineage` (`schema_name=dagsynth.dag_lineage`, `schema_version=1.0.0`) with strict adjacency/assignment checks and backward-compatible optional metadata mode

## [0.1.11] - 2026-02-23

### Added

- Missingness CLI overrides for `generate`: `--missing-rate`, `--missing-mechanism`, `--missing-mar-observed-fraction`, `--missing-mar-logit-scale`, `--missing-mnar-logit-scale`
- Missingness presets: `configs/preset_missingness_mcar.yaml`, `configs/preset_missingness_mar.yaml`, `configs/preset_missingness_mnar.yaml`
- Missingness wrapper script: `scripts/generate-missingness.sh`
- Missingness benchmark guardrails (runtime vs missingness-off control + acceptance checks) surfaced in profile summaries and regression issues

## [0.1.10] - 2026-02-23

### Added

- End-to-end missingness injection in generation/postprocess path with compact metadata summaries for configured and realized rates

## [0.1.9] - 2026-02-23

### Added

- Deterministic MCAR/MAR/MNAR missingness mask samplers with typed config validation coverage

## [0.1.8] - 2026-02-23

### Added

- Architecture documentation with mermaid diagrams (`docs/architecture.md`)
- Torch-native steering metric extractor (`core/steering_metrics.py`) to avoid NumPy conversion during candidate scoring

### Fixed

- Diagnostics coverage aggregation now uses deterministic reservoir sampling per metric with configurable retention cap (`diagnostics.max_values_per_metric`) to bound memory on long runs

## [0.1.7] - 2026-02-22

### Changed

- Eliminated NumPy bottlenecks across generation pipeline: `functions/`, `linalg/`, and converters now use torch-native implementations
- Steering candidate scoring uses torch-native metric path and torch softmax selection to avoid CPU/NumPy round trips on accelerator runs

### Fixed

- Correlated beta sampling no longer reseeds process-wide Torch RNG state; draws now use a local seed-derived generator for reproducibility isolation

## [0.1.6] - 2026-02-22

### Added

- Opt-in soft meta-feature steering with bounded deterministic candidate selection
- New generation CLI controls: `--diagnostics`, `--steer-meta`, and repeatable `--meta-target key=min:max[:weight]`
- Steering metadata payload propagation on generated bundles when steering is enabled
- Benchmark diagnostics collection controls: `dagsynth benchmark --diagnostics [--diagnostics-out-dir ...]`
- Per-profile benchmark diagnostics artifacts and summary pointers under `diagnostics/<sanitized_profile_key>_<hash>/`
- New presets: `configs/preset_diagnostics_on.yaml` and `configs/preset_steering_conservative.yaml`

### Changed

- Added top-level `GeneratorConfig.meta_feature_targets` and `SteeringConfig` with disabled-by-default safe defaults
- Target resolution now merges legacy `diagnostics.meta_feature_targets` with top-level targets (top-level precedence)
- Benchmark markdown reports now surface diagnostics state and artifact pointers per profile
- Script/README benchmark workflow examples now include diagnostics and conservative steering presets

## [0.1.5] - 2026-02-22

### Added

- Diagnostics coverage aggregation reports with bin target-band analysis
- Dataset-level diagnostics metric extractors
- Staged curriculum CLI workflow with presets and documentation
- Three-stage curriculum core implementation
- CI/CD workflow, AGENTS.md docs, and config/test improvements
- Torch determinism tests and tightened postprocess assertions
- Expanded unit coverage and CI coverage checks

### Changed

- Switched to torch-first runtime with streaming generation
- Removed sklearn dependency; replaced with torch-native implementations

### Fixed

- Bin target-band coverage on configured range
- Coverage robustness and band accounting
- Null diagnostics coverage config values handling

## 0.1.0 – 0.1.4

Early development releases. No structured changelog was maintained for these versions.
