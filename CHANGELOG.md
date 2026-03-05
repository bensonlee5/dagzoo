# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2026-03-05

### Added

- Added worker-partition runtime schema fields:
  `runtime.worker_count` and `runtime.worker_index`.
- Added deterministic round-robin worker partitioning for
  `dagzoo generate --no-dataset-write`, where `num_datasets` remains the global
  dataset-index space and each worker covers a disjoint subset.

### Breaking

- **BREAKING:** `dagzoo generate` now rejects write-enabled runs when
  `runtime.worker_count > 1`. Multi-worker mode is currently supported only with
  `--no-dataset-write` until shard-writing coordination lands.
- **BREAKING:** `dagzoo benchmark` and `dagzoo diversity-audit` now reject
  configs with `runtime.worker_count > 1` until those flows become
  partition-aware.

## [0.5.2] - 2026-03-05

### Fixed

- Benchmark suite generation paths now consistently use a filter-disabled
  runtime config across throughput, latency, reproducibility, microbenchmarks,
  lineage guardrails, and missingness/shift/noise control runs. Presets with
  `filter.enabled=true` no longer crash benchmark execution after throughput.
- Benchmark `filter_retry_dataset_rate` now uses a denominator from the same
  sampled population as replayed filter rejections when deferred filter-stage
  replay metrics are active, preventing systematic underreporting on sampled
  suites.

## [0.5.1] - 2026-03-05

### Added

- Added a new deferred filtering CLI stage:
  `dagzoo filter --in <shard_dir> --out <out_dir>` with optional
  accepted-only curated shard output via `--curated-out`.
- Added deferred filter manifest (`filter_manifest.ndjson`) and summary
  (`filter_summary.json`) artifacts for acceptance auditing and throughput
  reporting.

### Changed

- Generated bundle metadata now always marks deferred filter state using
  `metadata.filter.mode` and `metadata.filter.status` (`not_run` immediately
  after generation).
- Benchmark filter stage throughput now replays deferred filtering over sampled
  bundles and reports replay-derived filter attempts/rejections when the filter
  stage is enabled.

### Breaking

- **BREAKING:** Inline filtering was removed from generation. `dagzoo generate`
  now errors when `filter.enabled=true`; users must run `dagzoo filter` as a
  separate stage.

## [0.4.9] - 2026-03-04

### Fixed

- Benchmark runtime guardrail control runs now execute the same per-bundle
  callback instrumentation (stage-sample update + throughput-pressure
  collection) as the primary throughput run, removing callback-overhead skew
  from missingness/shift/noise runtime degradation comparisons.
- Benchmark stage-sample bundles are now released immediately after write/filter
  stage probes, reducing avoidable bundle/tensor retention before latency
  collection and guardrail control runs.
- Filter stage throughput now uses benchmark-only runtime timing captured on
  pre-filter tensors (`bundle.runtime_metrics.filter_elapsed_seconds`), keeping
  emitted metadata deterministic while avoiding replay skew from postprocessing
  and missingness-transformed outputs.

## [0.4.8] - 2026-03-04

### Fixed

- Scalar RNG draws (`torch.randint`, `torch.rand`) now explicitly target the
  generator's device via new `rand_scalar`/`randint_scalar` helpers, fixing
  failures when the generator lives on a non-CPU device.
- Generation retry exception logging now includes the exception message for
  easier debugging.

### Changed

- Benchmark suite summaries now emit stage-level throughput metrics for
  generation, parquet writing, and filter replay:
  `generation_datasets_per_minute`, `write_datasets_per_minute`, and
  `filter_datasets_per_minute` (nullable when filtering is disabled).
- Benchmark suite summaries now emit throughput-pressure metrics required for
  end-to-end interpretation, including attempt pressure and both filter
  rejection views:
  `filter_rejection_rate_attempt_level` and `filter_retry_dataset_rate`.
- Benchmark baseline defaults now gate on stage throughput metrics in addition
  to `datasets_per_minute`.
- Benchmark CLI single-line preset summaries and markdown reports now include
  stage throughput and filter rejection telemetry columns.
- Added RTX 3060 to the hardware peak FLOPS lookup table.

### Breaking

- Generated bundle metadata now includes a new additive
  `metadata.generation_attempts` object. Consumers with strict metadata schemas
  must allow this field.

## [0.4.7] - 2026-03-04

### Fixed

- Generate config resolution now applies CLI `--rows` overrides before final
  post-policy validation, preventing premature failures when CUDA tiered
  policies raise `dataset.n_test`.
- Benchmark preset resolution now rejects `dataset.rows` before hardware-policy
  transforms, so users consistently receive the intended unsupported-feature
  error message.

## [0.4.6] - 2026-03-04

### Added

- Added unified total-row control for generation via `dataset.rows` and
  `dagzoo generate --rows ...`.
- `dataset.rows` supports:
  - fixed totals (for example `1024`)
  - ranges (for example `400..60000`)
  - choice lists (for example `1024,2048,4096` or YAML list form)
- Variable rows specs now resolve deterministically per dataset seed lineage.

### Changed

- When `dataset.rows` is active, split resolution keeps `dataset.n_test` fixed
  and derives `n_train = total_rows - n_test`.
- Fixed-layout APIs now require fixed row modes; variable `dataset.rows`
  (range/choices) fail fast with clear compatibility errors.
- Benchmark config resolution now rejects `dataset.rows` (benchmark paths remain
  on explicit `dataset.n_train` / `dataset.n_test` for now).
- Generated metadata config snapshots now reflect realized split sizes for each
  emitted bundle.

### Removed

- Retired curriculum shell workflow: removed
  `scripts/generate-curriculum.sh` and its test coverage.

### Breaking

- **BREAKING:** Curriculum shell users must migrate to `dataset.rows` /
  `--rows` workflows for staged row-envelope generation.

## [0.5.0] - 2026-03-03

### Changed

- **BREAKING:** Renamed package from `dagsynth` to `dagzoo`. All imports change
  from `from dagsynth ...` to `from dagzoo ...`.
- **BREAKING:** Lineage schema name changed from `dagsynth.dag_lineage` to
  `dagzoo.dag_lineage`. Existing Parquet files with the old schema name will not
  validate against the new constant.

## [0.4.5] - 2026-03-04

### Fixed

- Effective-diversity baseline regression comparison now fails fast on
  incompatible baselines (schema/version/suite/arm-set/config-fingerprint
  mismatch and missing evaluated-arm baseline entries) instead of silently
  passing.
- Effective-diversity scale audit now builds combined ablation arms from
  runtime-applicable mappings only, so unavailable activations do not force
  combined-arm skips.

## [0.4.4] - 2026-03-04

### Changed

- Added a first-class `dagzoo diversity-audit` CLI command that runs local
  overlap checks and dataset-scale overlap-impact ablations with reproducible
  run artifacts.
- Effective-diversity local auditing now reads activation names from runtime
  source-of-truth (`fixed_activation_names`) to avoid stale claim coverage when
  activation pools change.
- Added scale-impact baseline persistence and regression comparison workflow for
  repeated effective-diversity audits as the repository evolves.

## [0.4.3] - 2026-03-03

### Fixed

- Corrected mechanism-family sampling to iterate only across positive-probability
  families so `mechanism.function_family_mix` remains a strict hard mask even
  when RNG draws are exactly zero.

## [0.4.2] - 2026-03-03

### Changed

- Added `mechanism.function_family_mix` config support with strict
  normalization/validation and active runtime enforcement:
  unspecified families are excluded from sampling, and shift
  `mechanism_scale` reweights only within enabled families.
- Added `swiglu` to the global random activation pool as a shape-preserving
  variant (`x * silu(x)`).

### Breaking

- Generated bundle `metadata.config` now includes the top-level `mechanism`
  section (persisted config-schema extension). Consumers with strict
  schema expectations must allow this new section.

## [0.4.1] - 2026-03-03

### Changed

- Stabilized smoke benchmark CI behavior by disabling lineage runtime-overhead
  guardrail gating for smoke-size samples while retaining lineage metadata
  coverage validation.
- Lineage runtime degradation guardrails remain enabled for larger suites (for
  example `standard`) where sample sizes are sufficient for stable runtime
  comparisons.

## [0.4.0] - 2026-03-03

### Changed

- Renamed shift terminology in config/runtime/metadata:
  - `shift.mode` (was `shift.profile`)
  - `shift.variance_scale` (was `shift.noise_scale`)
  - `variance_sigma_multiplier` (was `noise_sigma_multiplier`) in emitted shift metadata
- Renamed noise amplitude key to `noise.base_scale` (was `noise.scale`) and standardized emitted dataset metadata under `noise_distribution` (was `noise`).
- Renamed benchmark preset terminology across config/CLI/summary payloads:
  - `benchmark.preset_name` (was `benchmark.profile_name`)
  - `benchmark.presets` (was `benchmark.profiles`)
  - summary keys `preset_results` / `preset_key` (was `profile_results` / `profile_key`)
- Renamed benchmark script/docs terminology from profile-centric wording to preset/tier wording.
- Replaced internal layout dict contracts with `LayoutPlan` dataclass wiring in generation/layout/metadata paths.
- Added explicit device-resolution metadata fields on generated bundles:
  - `requested_device`
  - `resolved_device`
  - `device_fallback_reason`
- Renamed diagnostics target bridge module:
  - `meta_targets.py` -> `diagnostics_targets.py`
  - `build_coverage_aggregation_config` -> `build_diagnostics_aggregation_config`

### Breaking

- `dagzoo benchmark --profile ...` was removed; use `--preset ...`.
- `dagzoo generate --no-write` was removed; use `--no-dataset-write`.
- Config files using `shift.profile`, `shift.noise_scale`, `noise.scale`, `benchmark.profile_name`, or `benchmark.profiles` must be migrated to the new keys above.
- Dataset metadata consumers must read `metadata.noise_distribution` (not `metadata.noise`) and `metadata.shift.mode` (not `metadata.shift.profile`).

## [0.3.3] - 2026-03-03

### Changed

- Refactored generation config validation into explicit staged validators in `src/dagzoo/config.py`:
  stage 1 field normalization/typing, stage 2 cross-field constraints, and stage 3 post-override revalidation.
- `resolve_generate_config()` and `resolve_benchmark_profile_config()` now rely on `GeneratorConfig.validate_generation_constraints()` as the single post-override revalidation pass.
- Classification configs now enforce split feasibility for both `dataset.n_classes_min` and `dataset.n_classes_max` (previously only `n_classes_min` was checked).
- `runtime.device: null` is normalized to `auto` during staged validation.
- Added/updated tests for ordering-sensitive constraints and override-driven revalidation behavior.
- Documented validation stages in `docs/config-resolution.md`.

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

- **Rename: `cauchy-generator` is now `dagzoo`.** Package name, CLI entrypoint, Python import paths, and all internal references updated. The CLI command is now `dagzoo` (was `cauchy-gen`). Python imports change from `from cauchy_generator.…` to `from dagzoo.…`.
- **Breaking:** Lineage schema name changed from `cauchy_generator.dag_lineage` to `dagzoo.dag_lineage`. Existing Parquet files with embedded lineage metadata will fail validation against the new schema name.
- `sample_cauchy_dag` renamed to `sample_dag`; `graph/cauchy_graph.py` renamed to `graph/dag_sampler.py`.
- Noise rollout presets for generate/benchmark smoke workflows (`configs/preset_noise_*.yaml`)
- Noise workflow script wrapper (`scripts/generate-noise.sh`)
- Noise benchmark guardrail reporting (`noise_guardrails`) for runtime delta vs gaussian control and metadata validity
- Noise feature guide (`docs/features/noise.md`) and usage/workflow documentation updates
- Benchmark profile summaries now surface noise guardrail status in CLI and Markdown reports
- **Breaking:** removed `noise.family=legacy`; default noise family is now explicit `gaussian`.
- **Breaking:** removed config/runtime compatibility knobs `runtime.hardware_aware`, CLI `--no-hardware-aware`, and graph aliases `graph.n_nodes_log2_min/max`.
- **Breaking:** `shift.mode` no longer accepts boolean aliases; only explicit string values are allowed.
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

- Versioned DAG lineage schema validator for `metadata.lineage` (`schema_name=dagzoo.dag_lineage`, `schema_version=1.0.0`) with strict adjacency/assignment checks and backward-compatible optional metadata mode

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
- Benchmark diagnostics collection controls: `dagzoo benchmark --diagnostics [--diagnostics-out-dir ...]`
- Per-profile benchmark diagnostics artifacts and summary pointers under `diagnostics/<sanitized_preset_key>_<hash>/`
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
