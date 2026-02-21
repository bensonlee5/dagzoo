# Literature-Driven Improvement Backlog (2026Q1)

This document is a docs-only backlog for improving synthetic data quality in `cauchy-generator`.

Scope:
- In scope: generator pipeline and diagnostics.
- Out of scope (this phase): implementation changes in `src/`.

Primary ranking objective:
- Expected downstream model quality gain.

Ranking method:
- Defined in `docs/backlog_decision_rules.md`.

## Current Gap Snapshot

Observed from current code/config surface:
- Class count support is capped (`n_classes_max=10`) in `src/cauchy_generator/config.py`.
- Missingness generation is not exposed in config or postprocessing.
- Meta-feature diagnostics/coverage steering module is not present.
- Noise family controls are not explicit in config.
- Shift-aware SCM variants are not represented in graph/node pipeline interfaces.

## Prioritized Backlog

### Near-Term Track (High Confidence)

| Rank | Improvement | Expected Quality Impact | Effort | Risk | Literature Evidence | Repo Touchpoints | Candidate Interface Additions (future) |
|---|---|---|---|---|---|---|---|
| 1 | Meta-feature coverage diagnostics + steering | High | Medium | Medium | A Closer Look at TabPFN v2 (`2502.17361`) | `src/cauchy_generator/config.py`, new `src/cauchy_generator/diagnostics/`, `src/cauchy_generator/filtering/`, benchmark reports | `GeneratorConfig.meta_feature_targets`, per-run coverage report artifact |
| 2 | Missingness generation (MCAR/MAR/MNAR) | High | Low-Medium | Low | TabPFN v2 (Nature), A Closer Look (`2502.17361`) | `src/cauchy_generator/postprocess/postprocess.py`, `src/cauchy_generator/config.py` | `DatasetConfig.missing_rate`, `DatasetConfig.missing_mechanism` |
| 3 | Curriculum scheduling over complexity | High | Medium | Low-Medium | TabICL (`2502.05564`), TabICLv2 (`2602.11139`) | `src/cauchy_generator/config.py`, `configs/`, generation entrypoints in `src/cauchy_generator/core/dataset.py` | `GeneratorConfig.curriculum_stage`, staged preset configs |
| 4 | Expand many-class support beyond 10 classes | High | Low-Medium | Medium | TabPFN Unleashed (`2502.02527`) | `src/cauchy_generator/config.py`, `src/cauchy_generator/converters/categorical.py`, `src/cauchy_generator/filtering/torch_rf_filter.py` | Wider `n_classes` ranges and class-aware filter thresholds |
| 5 | Noise family diversification | Medium-High | Low | Low | TabPFN v2 (Nature), Causal PFN (`2506.10914`) | `src/cauchy_generator/core/node_pipeline.py`, `src/cauchy_generator/config.py` | `GeneratorConfig.noise_distribution` (`gaussian/laplace/student_t/mixture`) |
| 6 | Mechanism family expansion (BNN/GP kernels/interactions) | Medium-High | Medium | Medium | TabPFN v2 (Nature), Causal PFN (`2506.10914`) | `src/cauchy_generator/functions/random_functions.py`, `src/cauchy_generator/functions/multi.py`, `src/cauchy_generator/config.py` | `function_family_mix` with explicit kernel/mechanism weights |

### Research Track (Higher Uncertainty)

| Rank | Improvement | Expected Quality Impact | Effort | Risk | Literature Evidence | Repo Touchpoints | Candidate Interface Additions (future) |
|---|---|---|---|---|---|---|---|
| 7 | 2nd-order / shift-aware SCM generation | Medium-High | High | Medium-High | Drift-Resilient TabPFN (`2411.10634`) | `src/cauchy_generator/sampling/random_weights.py`, `src/cauchy_generator/core/node_pipeline.py`, `src/cauchy_generator/config.py` | `GeneratorConfig.shift_profile`, latent shift variable propagation |
| 8 | Robustness-oriented hard-task/adversarial synthetic regimes | Medium | Medium-High | High | Robust tabular FM direction (`2512.03307`), TabPFGen (`2406.05216`) | `src/cauchy_generator/functions/`, `src/cauchy_generator/postprocess/`, benchmark suites | `difficulty_profile` and stress-test presets |
| 9 | Larger context/size regime expansion for pretraining realism | Medium | Medium | Medium | Scaling TabPFN (`2311.10609`), TabDPT (`2410.18164`) | `src/cauchy_generator/config.py`, benchmark configs, throughput harness | Wider dataset-size distributions with throughput guardrails |

## Public Interface Additions (Planning-Only)

No API changes are implemented in this phase. Candidate additions for future implementation:
- `DatasetConfig`:
  - `missing_rate: float`
  - `missing_mechanism: str`
  - expanded class and sample-size ranges
- `GeneratorConfig`:
  - `curriculum_stage: str | int`
  - `meta_feature_targets: dict[str, tuple[float, float]]`
  - `noise_distribution: str`
  - `shift_profile: str | dict[str, float]`
- New diagnostics artifacts:
  - meta-feature summary per generation run
  - target-vs-observed coverage report

## Acceptance Scenarios For Future Implementation

1. Reproducibility:
   fixed seeds preserve deterministic metadata lineage when new knobs are disabled.
2. Coverage:
   generated corpus meets configured target bands for selected meta-features.
3. Missingness:
   MCAR/MAR/MNAR mechanisms match expected missing-rate and dependency patterns.
4. Many-class robustness:
   high-class-count datasets are generated without excessive filter rejection.
5. Shift-aware behavior:
   configured drift modes produce measurable, intended distribution shift.
6. Performance guardrail:
   benchmark regression remains within configured warn/fail thresholds.

## Citations

- TabPFN v2 (Nature): https://doi.org/10.1038/s41586-024-08328-6
- A Closer Look at TabPFN v2: https://arxiv.org/abs/2502.17361
- Scaling TabPFN: https://arxiv.org/abs/2311.10609
- TabPFGen: https://arxiv.org/abs/2406.05216
- TabDPT: https://arxiv.org/abs/2410.18164
- Drift-Resilient TabPFN: https://arxiv.org/abs/2411.10634
- TabPFN Unleashed: https://arxiv.org/abs/2502.02527
- TabICL: https://arxiv.org/abs/2502.05564
- Foundation Models for Causal Inference via PFNs: https://arxiv.org/abs/2506.10914
- TabICLv2: https://arxiv.org/abs/2602.11139
- Robust tabular foundation model direction: https://arxiv.org/abs/2512.03307
