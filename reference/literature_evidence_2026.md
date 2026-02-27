# Literature Evidence Appendix (2026Q1)

This appendix links roadmap items in `docs/roadmap.md` to primary sources and current repo gaps.

Related docs:

- Canonical roadmap: `docs/roadmap.md`
- Decision rubric: `docs/backlog_decision_rules.md`
- Implementation baseline: `docs/roadmap.md` (`Current Implementation Baseline`)

Conventions:

- Confidence: `high`, `medium`, or `low`.
- Lane:
  roadmap milestone lane supported by this evidence (`Now`, `Next`, `Later`).

## Evidence-to-Roadmap Mapping

| Source                                                                                        | Key Claim Used                                                                                      | Roadmap IDs            | Lane             | Confidence  |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------- | ---------------- | ----------- |
| TabPFN v2 (Nature, 2024) https://doi.org/10.1038/s41586-024-08328-6                           | Prior design realism (including missingness/noise diversity) materially affects tabular performance | RD-003, RD-005, RD-007 | Now, Next        | high        |
| A Closer Look at TabPFN v2 (2502.17361) https://arxiv.org/abs/2502.17361                      | Coverage of weak meta-feature regimes improves reliability                                          | RD-008, RD-003         | Now              | high        |
| TabPFN Unleashed (2502.02527) https://arxiv.org/abs/2502.02527                                | Many-class settings need dedicated handling and scaling                                             | RD-007                 | Now              | medium-high |
| TabICL (2502.05564) https://arxiv.org/abs/2502.05564                                          | Curriculum and staged complexity improve optimization behavior                                      | RD-006                 | Now              | high        |
| TabICLv2 (2602.11139) https://arxiv.org/abs/2602.11139                                        | Synthetic prior quality and scale remain central to tabular FM performance                          | RD-006, RD-007, RD-008 | Now              | high        |
| Drift-Resilient TabPFN (2411.10634) https://arxiv.org/abs/2411.10634                          | Shift-aware constructions help drift robustness                                                     | RD-004                 | Next             | medium-high |
| Foundation Models for Causal Inference via PFNs (2506.10914) https://arxiv.org/abs/2506.10914 | Broader SCM/noise families and interventions support causal fidelity                                | RD-002, RD-004, RD-007 | Later, Next, Now | medium-high |
| TabPFGen (2406.05216) https://arxiv.org/abs/2406.05216                                        | Hard-regime synthetic generation can improve robustness behavior                                    | RD-005                 | Next             | medium      |
| Scaling TabPFN (2311.10609) https://arxiv.org/abs/2311.10609                                  | Scaling regimes require broader size/diversity coverage and throughput-conscious design             | RD-009, RD-010, RD-006 | Next, Now        | medium-high |
| TabDPT (2410.18164) https://arxiv.org/abs/2410.18164                                          | Broader synthetic diversity and scale can improve pretraining realism                               | RD-006, RD-007         | Now              | medium      |
| Robust tabular FM direction (2512.03307) https://arxiv.org/abs/2512.03307                     | Robustness-oriented training benefits from systematic stress regimes                                | RD-005                 | Next             | medium      |

## Per-Item Evidence Notes

### RD-001: Ground-Truth DAG Artifact Export

- Evidence type: architectural necessity for causal mission claims.
- Literature support: indirect from causal PFN framing and causal discovery use case.
- Confidence: medium.

### RD-002: Interventional and Counterfactual Generation Modes

- Strongest source: Causal PFN (`2506.10914`).
- Risk note: higher implementation and validation complexity than observational tracks.
- Confidence: medium-high.

### RD-003: Missingness Generation (MCAR/MAR/MNAR)

- Strongest sources: TabPFN v2 (Nature) and A Closer Look.
- Implementation status (as of 2026-02-24): implemented in repo via epics/issues `#17` and `#18`.
- Confidence: high.

### RD-004: Shift-Aware SCM Generation

- Strongest source: Drift-Resilient TabPFN (`2411.10634`).
- Confidence: medium-high.

### RD-005: Robustness Stress Profiles

- Strongest sources: TabPFGen and robust tabular FM direction.
- Confidence: medium.

### RD-006: Curriculum Complexity Scaling

- Strongest sources: TabICL and TabICLv2.
- Confidence: high.

### RD-007: Many-Class and High-Cardinality Expansion

- Strongest source: TabPFN Unleashed.
- Supporting signal: TabPFN v2 and TabICLv2.
- Confidence: medium-high.

### RD-008: Meta-Feature Coverage Steering

- Strongest source: A Closer Look at TabPFN v2.
- Confidence: high.

### RD-009: Parallel and Distributed Generation/Writing

- Evidence type: scaling/throughput requirement for pretraining data factories.
- Strongest source: Scaling TabPFN.
- Confidence: medium.

### RD-010: Hardware-Adaptive Autotuning Beyond Coarse FLOPs Tiers

- Evidence type: throughput scaling requirement plus engineering observability for hardware heterogeneity.
- Strongest source: Scaling TabPFN (`2311.10609`) with implementation-driven constraints from current coarse-tier behavior.
- Confidence: medium.

## Evidence Limits and Assumptions

- This is a planning artifact, not a reproduction benchmark.
- Mappings are design-level and must be validated against repo benchmarks/tests.
- Lane assignments reflect current repo context and can be revised after implementation learning.
