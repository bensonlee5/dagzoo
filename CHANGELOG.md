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

### Changed

- Eliminated NumPy bottlenecks across generation pipeline: `functions/`, `linalg/`, `converters/` now use torch-native implementations
- Steering candidate scoring uses torch-native metric path and torch softmax selection to avoid CPU/NumPy round trips on accelerator runs
- Diagnostics coverage aggregation now uses deterministic reservoir sampling per metric with configurable retention cap (`diagnostics.max_values_per_metric`) to bound memory on long runs
- Root docs and script docs now include recommended missingness generation and benchmark commands

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
