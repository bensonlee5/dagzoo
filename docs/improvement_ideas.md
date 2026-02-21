# Improving Model Quality via Reference Document Insights

## Context

The `cauchy-generator` repo faithfully implements TabICLv2 Appendix E (sections E.2-E.14) for generating synthetic tabular datasets used to train tabular foundation models. The `reference/` directory contains 13 PDFs and 1 markdown index covering related work. This document identifies concrete improvements to the synthetic data generation pipeline, drawn from those papers, that could improve the quality of downstream models.

---

## High-Impact Improvements

### 1. Curriculum Learning / Progressive Complexity (from TabICL)
**Paper:** `TabICL_2502.05564.pdf`
**Insight:** TabICL trains with a curriculum -- starting with simple datasets (few features, few samples, linear functions) and progressively increasing complexity. This significantly improved convergence and final model quality.
**What to change:**
- Add a `difficulty` or `curriculum_stage` parameter to `GeneratorConfig` (`src/cauchy_generator/config.py`)
- Implement staged config presets (e.g., `configs/curriculum_easy.yaml`, `configs/curriculum_hard.yaml`) that control:
  - Number of features (`n_features` range)
  - Number of samples (`n_train`, `n_test`)
  - Function complexity (restrict to linear/quadratic early, add neural nets/trees later)
  - DAG density (fewer edges early, more later)
- Alternatively, expose a utility that generates a sequence of configs with linearly interpolated complexity

**Files:** `src/cauchy_generator/config.py`, `configs/`, new curriculum utility module

### 2. Meta-Feature Distribution Matching (from A Closer Look at TabPFN v2)
**Paper:** `A Closer Look at TabPFN v2.pdf`
**Insight:** The paper shows that specific meta-features of datasets (class imbalance, feature correlations, dimensionality, sample size) predict whether PFN-style models succeed or fail. Synthetic data that doesn't cover the meta-feature space of real-world datasets leaves blind spots.
**What to change:**
- Add a meta-feature computation module that measures generated datasets' properties (class balance, feature correlation structure, effective dimensionality, noise level)
- Add optional rejection sampling or config steering to ensure generated datasets cover target meta-feature distributions
- Use this as a diagnostic tool: given a collection of generated datasets, report coverage gaps vs. real-world benchmarks (e.g., OpenML CC-18)

**Files:** New `src/cauchy_generator/diagnostics/` module, `src/cauchy_generator/filtering/`

### 3. Expanded Class Count Support (from TabPFN Unleashed)
**Paper:** `TabPFN_Unleashed_2502.02527.pdf`
**Insight:** TabPFN struggles with >10 classes. The BETA method uses Error-Correcting Output Codes (ECOC) at inference time, but the root cause is that the synthetic prior doesn't generate enough high-class-count datasets. Generating more datasets with 10-100 classes during training would improve native multi-class performance.
**What to change:**
- Extend `n_classes` range in config to support up to 100 classes
- Adjust categorical converter logic in `src/cauchy_generator/converters/categorical.py` to handle many-class scenarios without degenerate distributions
- Ensure the ExtraTrees OOB filter (`src/cauchy_generator/filtering/extratrees_filter.py`) doesn't over-reject high-class datasets (may need relaxed thresholds for many classes)

**Files:** `src/cauchy_generator/config.py`, `src/cauchy_generator/converters/categorical.py`, `src/cauchy_generator/filtering/extratrees_filter.py`

### 4. Richer SCM Structures / 2nd-Order SCMs (from Drift Resilient TabPFN)
**Paper:** `Drift_Resilient_TabPFN_2411.10634.pdf`
**Insight:** Introduces 2nd-order SCMs where edge weights themselves are functions of a latent variable (e.g., time). This creates datasets with distribution shifts, teaching models to be robust. The paper also uses Time2Vec positional encodings for temporal features.
**What to change:**
- Add an optional "shift" mechanism to the DAG pipeline: edge weights in `src/cauchy_generator/sampling/random_weights.py` become functions of a shift variable (could be time, group index, etc.)
- This requires modifying `src/cauchy_generator/core/node_pipeline.py` to propagate the shift variable through the DAG computation
- Add a `temporal_shift` config option that controls whether and how distribution shifts are introduced

**Files:** `src/cauchy_generator/sampling/random_weights.py`, `src/cauchy_generator/core/node_pipeline.py`, `src/cauchy_generator/config.py`

### 5. More Function Family Diversity (from TabPFN v2 / CausalFM)
**Paper:** `Accurate predictions on small data with a tabular foundation model.pdf`, `Foundation_Models_for_Causal_Inference_2506.10914.pdf`
**Insight:** TabPFN v2 uses Bayesian neural network (BNN) priors with weight uncertainty. CausalFM uses SCMs with flexible noise distributions and non-linear mechanisms drawn from GPs with diverse kernels. Both suggest that richer function families in the prior improve downstream performance.
**What to change:**
- Add BNN-style random functions: neural nets where weights are sampled from distributions with varying scale (already partially covered by the existing neural net function, but could add explicit weight uncertainty / dropout during generation)
- Add GP-based random functions with diverse kernel choices (RBF, Matern, periodic, polynomial)
- These would be new entries in `src/cauchy_generator/functions/random_functions.py`

**Files:** `src/cauchy_generator/functions/random_functions.py`, `src/cauchy_generator/config.py`

---

## Medium-Impact Improvements

### 6. Larger Dataset Sizes (from Scaling TabPFN, TabDPT, LoCalPFN)
**Papers:** `Scaling_TabPFN_2311.10609.pdf`, `TabDPT_2410.18164.pdf`, `LoCalPFN_Retrieval_FineTuning_2406.05207.pdf`
**Insight:** Multiple papers show PFN models improve with larger context sizes. Current defaults (768 train / 256 test) limit what the trained model can handle. Generating with larger sample counts (up to 10K+) during training is needed for models that serve larger real-world datasets.
**What to change:**
- Increase default and max `n_train` / `n_test` in configs
- Profile and optimize the pipeline for larger datasets (the PyTorch/CUDA backend should help)
- Consider variable-size generation where each dataset has a randomly sampled size from a wide range

**Files:** `src/cauchy_generator/config.py`, `configs/*.yaml`

### 7. Noise Model Diversity (from TabPFN v2, CausalFM)
**Papers:** `Accurate predictions on small data...`, `Foundation_Models_for_Causal_Inference_2506.10914.pdf`
**Insight:** TabPFN v2 and CausalFM use diverse noise distributions in their SCMs (Gaussian, Laplace, Student-t, mixture). The current generator uses fixed noise addition. More diverse noise would make trained models more robust.
**What to change:**
- Add configurable noise distributions in the node pipeline (`src/cauchy_generator/core/node_pipeline.py`)
- Support Gaussian, Laplace, Student-t, uniform, and mixtures thereof
- Add a `noise_distribution` config parameter

**Files:** `src/cauchy_generator/core/node_pipeline.py`, `src/cauchy_generator/config.py`

### 8. Feature Interaction Patterns (from TabICL, TabPFN v2)
**Insight:** Real-world tabular data often has specific interaction patterns (multiplicative interactions, threshold effects, XOR-like patterns). The current generator handles multi-input composition (E.7) but could benefit from explicitly generating common interaction types.
**What to change:**
- Add interaction-specific function compositions in `src/cauchy_generator/functions/multi.py`
- Include explicit multiplicative, threshold/step, and modular arithmetic interactions

**Files:** `src/cauchy_generator/functions/multi.py`

### 9. Missing Value Generation (from TabPFN v2, A Closer Look)
**Insight:** TabPFN v2 explicitly handles missing values in its prior by randomly masking features during synthetic data generation. This teaches the model to handle missingness natively. The current generator does not introduce missing values.
**What to change:**
- Add a post-processing step in `src/cauchy_generator/postprocess/` that randomly masks feature values with NaN
- Support multiple missingness mechanisms: MCAR (random), MAR (dependent on observed features), MNAR (dependent on the missing value itself)
- Add `missing_rate` and `missing_mechanism` config parameters

**Files:** `src/cauchy_generator/postprocess/`, `src/cauchy_generator/config.py`

---

## Lower-Impact / Specialized Improvements

### 10. Interventional / Counterfactual Data (from CausalFM)
For models targeting causal inference tasks, generate interventional distributions by clamping specific nodes in the DAG during generation.

### 11. Energy-Based Conditional Generation (from TabPFGen)
Use the synthetic data generator as a conditional generative model by training an energy-based formulation on top of PFN embeddings.

### 12. Permutation Equivariance Awareness (from EquiTabPFN)
Ensure generated datasets don't have artifacts from column/row ordering. The current postprocessing includes column permutation (`src/cauchy_generator/postprocess/permutation.py`), which partially addresses this.

### 13. Retrieval-Aware Data (from LoCalPFN)
Generate datasets that are pre-organized for kNN-based context selection, ensuring local neighborhoods in feature space have consistent label structure.

---

## Recommended Priority Order

| Priority | Improvement | Effort | Expected Impact |
|----------|------------|--------|-----------------|
| 1 | Curriculum learning (#1) | Medium | High -- directly improves training convergence |
| 2 | Missing value generation (#9) | Low | High -- real data almost always has missingness |
| 3 | More function families (#5) | Medium | High -- richer priors = better generalization |
| 4 | Expanded class counts (#3) | Low-Medium | High -- removes a known failure mode |
| 5 | Noise model diversity (#7) | Low | Medium-High -- cheap robustness gain |
| 6 | Meta-feature matching (#2) | Medium-High | Medium-High -- diagnostic + steering value |
| 7 | Larger dataset sizes (#6) | Low | Medium -- config change + profiling |
| 8 | 2nd-order SCMs (#4) | High | Medium -- specialized but impactful |
| 9 | Feature interactions (#8) | Low-Medium | Medium |
