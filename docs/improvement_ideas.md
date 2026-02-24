# Literature-Driven Improvement Backlog (2026Q1)

This is the prioritized execution queue for roadmap items defined in `docs/roadmap.md`.

Scope:

- In scope: generator pipeline, diagnostics, and benchmark/throughput guardrails.
- Out of scope (this phase): implementation changes in `src/` and `configs/`.

Ranking method:

- `docs/backlog_decision_rules.md`

Related docs:

- Canonical roadmap: `docs/roadmap.md`
- Evidence appendix: `docs/literature_evidence_2026.md`
- Current implementation contract: `docs/implementation.md`

## Current Gap Snapshot

Observed from current code/config surface:

- Diagnostics extraction and coverage aggregation exist; soft steering is implemented.
- Class support defaults remain narrow (`n_classes_max=10`) and categorical cardinality defaults are conservative.
- Missingness mechanisms are configurable (MCAR/MAR/MNAR) with deterministic sampling, CLI overrides, and benchmark guardrails.
- Shift-aware SCM and interventional/counterfactual generation are not implemented.
- Curriculum stages scale row counts and split regime only.
- Hardware-aware scaling is profile-tier based and coarse; no bounded adaptive autotuning loop exists yet.
- Streaming Parquet writing is sequential, not multi-worker.

## Recently Implemented

- RD-003: Missingness generation (MCAR/MAR/MNAR), completed in epics/issues `#17` and `#18`:
  - typed config schema and strict validation
  - deterministic missingness mask samplers
  - end-to-end injection into generation/postprocess with metadata summaries
  - benchmark missingness acceptance/runtime guardrails
- RD-008: Meta-feature coverage steering (soft candidate selection with configurable target bands).

## Prioritized Queue

| Rank          | Roadmap ID | Item                                             | Status      | Milestone | Mission Alignment                    | Expected Impact | Effort      | Risk        | Key Repo Touchpoints                                                                                                                                                                                               |
| ------------- | ---------- | ------------------------------------------------ | ----------- | --------- | ------------------------------------ | --------------- | ----------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0 (completed) | RD-003     | Missingness generation (MCAR/MAR/MNAR)           | implemented | Now       | pretraining, robustness testing      | High            | Low-Medium  | Low         | `src/cauchy_generator/config.py`, `src/cauchy_generator/sampling/missingness.py`, `src/cauchy_generator/postprocess/postprocess.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/bench/suite.py` |
| 1             | RD-007     | Many-class and high-cardinality expansion        | planned     | Now       | pretraining                          | High            | Low-Medium  | Medium      | `src/cauchy_generator/config.py`, `src/cauchy_generator/converters/categorical.py`, `src/cauchy_generator/filtering/torch_rf_filter.py`                                                                            |
| 2             | RD-006     | Curriculum complexity scaling (features + graph) | planned     | Now       | pretraining                          | Medium-High     | Medium      | Medium      | `src/cauchy_generator/config.py`, `src/cauchy_generator/core/dataset.py`                                                                                                                                           |
| 3             | RD-001     | Ground-truth DAG artifact export                 | implemented | Now       | causal discovery                     | Medium-High     | Medium      | Low-Medium  | `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/io/parquet_writer.py`, `src/cauchy_generator/types.py`                                                                                               |
| 4             | RD-004     | Shift-aware SCM generation                       | planned     | Next      | robustness testing, causal discovery | Medium-High     | High        | Medium-High | `src/cauchy_generator/sampling/random_weights.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/config.py`                                                                                        |
| 5             | RD-010     | Hardware-adaptive autotuning beyond coarse tiers | planned     | Next      | pretraining throughput               | Medium          | Medium      | Medium      | `src/cauchy_generator/hardware.py`, `src/cauchy_generator/config.py`, `src/cauchy_generator/cli.py`, `src/cauchy_generator/bench/suite.py`, `src/cauchy_generator/bench/report.py`                                 |
| 6             | RD-005     | Robustness hard-task/adversarial profiles        | research    | Next      | robustness testing                   | Medium          | Medium-High | High        | `src/cauchy_generator/functions/`, `src/cauchy_generator/postprocess/`, `src/cauchy_generator/bench/`                                                                                                              |
| 7             | RD-009     | Parallel/distributed generation                  | research    | Next      | pretraining throughput               | Low-Medium      | High        | Medium      | `src/cauchy_generator/core/`, `src/cauchy_generator/io/parquet_writer.py`, `src/cauchy_generator/cli.py`                                                                                                           |
| 8             | RD-002     | Interventional and counterfactual generation     | research    | Later     | causal discovery                     | Medium          | High        | High        | `src/cauchy_generator/core/node_pipeline.py`, `src/cauchy_generator/core/dataset.py`, `src/cauchy_generator/config.py`                                                                                             |

## Interface Additions (Implemented + Planning)

Implemented interface additions:

- RD-003 (implemented via `#17`/`#18`):
  - `DatasetConfig.missing_rate: float`
  - `DatasetConfig.missing_mechanism: Literal["none", "mcar", "mar", "mnar"]`
  - `DatasetConfig.missing_mar_observed_fraction: float`
  - `DatasetConfig.missing_mar_logit_scale: float`
  - `DatasetConfig.missing_mnar_logit_scale: float`
  - `cauchy-gen generate` missingness CLI overrides
  - benchmark output `missingness_guardrails`

Candidate future additions by roadmap item:

- RD-004:
  - `GeneratorConfig.shift_profile: str | dict[str, float]`
- RD-006:
  - per-stage feature/node/depth range controls under curriculum config
- RD-007:
  - expanded `DatasetConfig.n_classes_*` and categorical cardinality ranges
- RD-002:
  - `GeneratorConfig.intervention_mode: Literal["off", "do", "counterfactual"]`
- RD-009:
  - `GeneratorConfig.num_workers: int`
- RD-010:
  - `RuntimeConfig.autotune_mode: Literal["off", "profile", "adaptive"]`
  - `RuntimeConfig.autotune_max_trials: int`
  - `RuntimeConfig.autotune_time_budget_sec: float`
  - benchmark summary `autotune` telemetry fields

## Acceptance Scenarios (Execution Contract)

1. RD-001 ground-truth DAG artifacts (completed via `#44`, `#45`, `#46`, `#47`, `#48`):
   generated outputs include adjacency and node assignment lineage with schema validation tests and compact shard artifact persistence.
1. RD-002 intervention/counterfactual consistency:
   intervention mode preserves causal truncation behavior under fixed intervention sets.
1. RD-003 missingness mechanisms:
   implemented in epics/issues `#17` and `#18`; MCAR/MAR/MNAR targets match expected missing-rate and dependency behavior with benchmark guardrails.
1. RD-004 shift-aware behavior:
   enabled shift profile produces measurable drift; disabled profile matches baseline behavior.
1. RD-005 stress profile validity:
   hard-task presets produce intended diagnostics deltas versus baseline.
1. RD-006 curriculum monotonicity:
   later stages have equal or greater feature and graph complexity than earlier stages.
1. RD-007 many-class robustness:
   high-class datasets generate without excessive filter rejection and preserve label validity.
1. RD-009 parallel determinism:
   multi-worker mode reproduces single-worker outputs for fixed seeds within declared tolerance.
1. RD-010 autotune uplift and safety:
   adaptive mode improves throughput versus profile baseline on supported hardware without violating memory/runtime guardrails.

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
