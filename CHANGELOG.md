# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Architecture documentation with mermaid diagrams (`docs/architecture.md`)
- Torch-native steering metric extractor (`core/steering_metrics.py`) — avoids NumPy conversion during candidate scoring
- Deterministic MCAR/MAR/MNAR missingness mask samplers with typed config validation coverage
- End-to-end missingness injection in generation/postprocess path with compact metadata summaries for configured and realized rates
- Missingness CLI overrides for `generate`: `--missing-rate`, `--missing-mechanism`, `--missing-mar-observed-fraction`, `--missing-mar-logit-scale`, `--missing-mnar-logit-scale`
- Missingness presets: `configs/preset_missingness_mcar.yaml`, `configs/preset_missingness_mar.yaml`, `configs/preset_missingness_mnar.yaml`
- Missingness wrapper script: `scripts/generate-missingness.sh`
- Missingness benchmark guardrails (runtime vs missingness-off control + acceptance checks) surfaced in profile summaries and regression issues
- Versioned DAG lineage schema validator for `metadata.lineage` (`schema_name=cauchy_generator.dag_lineage`, `schema_version=1.0.0`) with strict adjacency/assignment checks and backward-compatible optional metadata mode
- Benchmark lineage-export guardrails (`lineage_guardrails`) that compare shard persistence throughput against lineage-stripped controls and surface warn/fail issues in regression summaries
- Lineage benchmark smoke preset: `configs/preset_lineage_benchmark_smoke.yaml`
- Curriculum metadata diagnostics payload (`metadata.curriculum.realized_complexity`, `stage_bounds`, `monotonicity_axes`) for staged complexity tracking across tasks

### Changed

- Eliminated NumPy bottlenecks across generation pipeline: `functions/`, `linalg/`, `converters/` now use torch-native implementations
- Steering candidate scoring uses torch-native metric path and torch softmax selection to avoid CPU/NumPy round trips on accelerator runs
- Diagnostics coverage aggregation now uses deterministic reservoir sampling per metric with configurable retention cap (`diagnostics.max_values_per_metric`) to bound memory on long runs
- Root docs and script docs now include recommended missingness generation and benchmark commands
- Roadmap/backlog/implementation docs now track RD-010 hardware-adaptive autotuning beyond coarse FLOPs-tier profile tuning
- Successful dataset generation now emits versioned DAG lineage payloads (`metadata.lineage`) with adjacency/assignment metadata derived from seeded layout sampling
- Parquet shard persistence now rewrites dense DAG lineage adjacency into shard-level bit-packed artifacts (`adjacency.bitpack.bin` + `adjacency.index.json`) and stores compact lineage pointers in per-dataset `metadata.json`
- Benchmark CLI/report outputs now include lineage guardrail status in per-profile summaries
- README/roadmap/implementation docs now document compact lineage artifact workflows and lineage-overhead benchmark guardrails

### Fixed

- Correlated beta sampling no longer reseeds process-wide Torch RNG state; draws now use a local seed-derived generator for reproducibility isolation

## [0.1.6] - 2026-02-22

### Added

- Opt-in soft meta-feature steering with bounded deterministic candidate selection
- New generation CLI controls: `--diagnostics`, `--steer-meta`, and repeatable `--meta-target key=min:max[:weight]`
- Steering metadata payload propagation on generated bundles when steering is enabled
- Benchmark diagnostics collection controls: `cauchy-gen benchmark --diagnostics [--diagnostics-out-dir ...]`
- Per-profile benchmark diagnostics artifacts and summary pointers under `diagnostics/<profile>/`
- New presets: `configs/preset_diagnostics_on.yaml` and `configs/preset_steering_conservative.yaml`

### Changed

- Added top-level `GeneratorConfig.meta_feature_targets` and `SteeringConfig` with disabled-by-default safe defaults
- Target resolution now merges legacy `diagnostics.meta_feature_targets` with top-level targets (top-level precedence)
- Benchmark markdown reports now surface diagnostics state and artifact pointers per profile
- Script/README benchmark workflow examples now include diagnostics and conservative steering presets

## [0.1.5] - 2025-02-22

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
