# Literature Evidence Appendix (2026Q1)

This appendix links backlog items to primary sources and current repo gaps.

Conventions:

- Confidence: `high`, `medium`, or `low`.
- Mapping: how the evidence translates to `cauchy-generator` changes.

## 1) TabPFN v2 (Nature, 2024)

Source:

- https://doi.org/10.1038/s41586-024-08328-6

Claim relevant here:

- Prior design details strongly affect tabular generalization; realism features such as missingness/noise diversity matter for robustness.

Mapping to current repo:

- Supports adding explicit missingness and broader noise controls in config and generation flow.

Confidence:

- High.

Backlog links:

- Missingness generation.
- Noise family diversification.
- Mechanism family expansion.

## 2) A Closer Look at TabPFN v2 (arXiv:2502.17361)

Source:

- https://arxiv.org/abs/2502.17361

Claim relevant here:

- Performance is sensitive to dataset meta-features; identifying and covering weak meta-feature regimes improves reliability.

Mapping to current repo:

- Justifies a diagnostics module and coverage-steering loop against configured meta-feature targets.

Confidence:

- High.

Backlog links:

- Meta-feature coverage diagnostics + steering.
- Missingness and class-imbalance coverage planning.

## 3) TabPFN Unleashed (arXiv:2502.02527)

Source:

- https://arxiv.org/abs/2502.02527

Claim relevant here:

- Multi-class scaling remains a challenge, with practical adaptations needed for larger class spaces.

Mapping to current repo:

- Current class range cap (`<=10`) is a likely coverage gap; class-aware converter/filter behavior should be reviewed.

Confidence:

- Medium-High.

Backlog links:

- Expanded many-class support.

## 4) TabICL (arXiv:2502.05564)

Source:

- https://arxiv.org/abs/2502.05564

Claim relevant here:

- Curriculum/progressive complexity is useful for stable optimization and better final performance.

Mapping to current repo:

- Supports staged config presets and curriculum controls in generator config.

Confidence:

- High.

Backlog links:

- Curriculum scheduling over complexity.

## 5) TabICLv2 (arXiv:2602.11139)

Source:

- https://arxiv.org/abs/2602.11139

Claim relevant here:

- Scaling and improved synthetic prior design remain central to high-performing tabular foundation models.

Mapping to current repo:

- Confirms that complexity schedules and broader size/meta-feature coverage are relevant next steps.

Confidence:

- High.

Backlog links:

- Curriculum scheduling.
- Context/size regime expansion.

## 6) Drift-Resilient TabPFN (arXiv:2411.10634)

Source:

- https://arxiv.org/abs/2411.10634

Claim relevant here:

- Shift-aware SCM constructions improve temporal/distribution-shift resilience.

Mapping to current repo:

- Motivates adding latent shift variables and shift-dependent edge/mechanism controls.

Confidence:

- Medium-High.

Backlog links:

- 2nd-order / shift-aware SCM generation.

## 7) Foundation Models for Causal Inference via PFNs (arXiv:2506.10914)

Source:

- https://arxiv.org/abs/2506.10914

Claim relevant here:

- Broader SCM/noise/mechanism families improve causal-faithful synthetic distributions.

Mapping to current repo:

- Supports adding richer noise and mechanism families, with optional interventional data tracks later.

Confidence:

- Medium-High.

Backlog links:

- Noise family diversification.
- Mechanism family expansion.

## 8) TabPFGen (arXiv:2406.05216)

Source:

- https://arxiv.org/abs/2406.05216

Claim relevant here:

- PFN-based synthetic generation can be used to target hard regimes and improve conditional generation quality.

Mapping to current repo:

- Motivates a research track for stress-test/hard-task generation presets.

Confidence:

- Medium.

Backlog links:

- Robustness-oriented hard-task generation.

## 9) Scaling TabPFN (arXiv:2311.10609)

Source:

- https://arxiv.org/abs/2311.10609

Claim relevant here:

- Scaling to larger tables requires careful context/feature handling; data priors should reflect those size regimes.

Mapping to current repo:

- Supports expanding generated sample-size distributions and validating throughput impact.

Confidence:

- Medium-High.

Backlog links:

- Larger context/size regime expansion.

## 10) TabDPT (arXiv:2410.18164)

Source:

- https://arxiv.org/abs/2410.18164

Claim relevant here:

- Real-data and larger-scale pretraining dynamics suggest broader synthetic diversity and size regimes are beneficial.

Mapping to current repo:

- Reinforces size/coverage expansion and benchmarking guardrails.

Confidence:

- Medium.

Backlog links:

- Larger context/size regime expansion.

## 11) Robust Tabular Foundation Model Direction (arXiv:2512.03307)

Source:

- https://arxiv.org/abs/2512.03307

Claim relevant here:

- Robustness-focused evaluation/training suggests value in systematically generating challenging regimes.

Mapping to current repo:

- Supports explicit stress-test dataset modes (adversarial corruption, hard interactions, low-SNR zones).

Confidence:

- Medium.

Backlog links:

- Robustness-oriented hard-task generation.

## Evidence Limits and Assumptions

- This appendix is a planning artifact, not a full reproduction study.
- Claims are mapped at design level; implementation decisions should be validated with repo benchmarks and tests.
- Where papers target downstream training methods rather than generator internals, mappings are marked as medium confidence.
