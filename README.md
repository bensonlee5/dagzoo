# cauchy-generator

**A high-performance causal engine for generating high-quality synthetic tabular data.**

`cauchy-generator` provides the structural priors necessary for training and evaluating next-generation Tabular Foundation Models. It bridges the gap between simple statistical noise and the complex, hierarchical dependencies found in real-world data by combining **Structural Causal Models (SCMs)** with a **hardware-native execution pipeline**.

______________________________________________________________________

## Core Mission

The project is designed to serve as a high-throughput data factory for researchers and engineers working on:

- **Foundation Model Pretraining:** Providing diverse, structural priors to improve model generalization.
- **Causal Discovery:** Generating "ground-truth" DAGs and interventional datasets for algorithm validation.
- **Robustness Testing:** Systematically creating challenging "hard-task" regimes, distribution shifts, and adversarial data conditions.

______________________________________________________________________

## Strategic Pillars

### 1. Causal Structural Integrity

Unlike traditional synthetic generators that rely on independent feature sampling, `cauchy-generator` builds datasets through **Cauchy DAG-based execution**.

- **Hierarchical Dependencies:** Features are generated as nodes in a graph, where child values are non-linear functional transforms of their parents.
- **Deep Mechanism Families:** Functional relationships incorporate Neural Networks, Tree ensembles, GPs, and parametric activations, ensuring complex multi-order interactions.

### 2. Tabular Realism

The generator incorporates the "messiness" of real-world tabular data directly into its priors:

- **Mixed-Type Converters:** Native support for high-cardinality categorical features and multi-scale numeric data.
- **Post-processing Hooks:** Real-world scaling, outlier clipping, and class-permutation logic to mimic benchmark data (e.g., OpenML, Kaggle).
- **Complexity Curriculum:** Optional staged generation that scales dataset complexity (features, nodes, samples) to support progressive learning.

### 3. Hardware-Native Performance

Performance is a first-class citizen. The engine is built for datacenter-scale throughput:

- **Torch-Native Pipeline:** Optimized for NVIDIA CUDA and Apple Silicon (MPS), minimizing Python-level bottlenecks.
- **Hardware-Aware Scaling:** Auto-detects GPU capabilities (e.g., H100 vs. RTX 4090) to tune generation parameters and maximize utilization.
- **Parallel Streaming:** Native Parquet sharding for efficient data loading during model training.

______________________________________________________________________

## Quick Start

### Installation

```bash
uv sync --group dev
```

### Basic Generation

```bash
# Generate 10 datasets using the default high-quality prior
uv run cauchy-gen generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

```bash
# Enable diagnostics artifacts and opt-in meta-feature steering
uv run cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --steer-meta \
  --meta-target linearity_proxy=0.25:0.75:1.5
```

```bash
# Use discoverable presets for diagnostics-only and conservative steering runs
uv run cauchy-gen generate --config configs/preset_diagnostics_on.yaml --num-datasets 25 --diagnostics --out data/run_diag
uv run cauchy-gen generate --config configs/preset_steering_conservative.yaml --num-datasets 25 --diagnostics --out data/run_steering
```

```bash
# Missingness presets (MCAR/MAR/MNAR)
uv run cauchy-gen generate --config configs/preset_missingness_mcar.yaml --num-datasets 25 --out data/run_missing_mcar
uv run cauchy-gen generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
uv run cauchy-gen generate --config configs/preset_missingness_mnar.yaml --num-datasets 25 --out data/run_missing_mnar
```

```bash
# Override missingness controls directly from CLI with strict validation
uv run cauchy-gen generate \
  --config configs/default.yaml \
  --num-datasets 25 \
  --device cpu \
  --missing-rate 0.25 \
  --missing-mechanism mar \
  --missing-mar-observed-fraction 0.6 \
  --missing-mar-logit-scale 1.4 \
  --out data/run_missing_cli_mar
```

### Benchmarking Performance

```bash
# Run the standard benchmark suite for the CPU profile
uv run cauchy-gen benchmark --suite standard --profile cpu
```

```bash
# Collect diagnostics during benchmark runs and emit artifact pointers in summary outputs
uv run cauchy-gen benchmark \
  --suite smoke \
  --profile cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

```bash
# Benchmark a missingness-enabled custom config with runtime + acceptance guardrails
uv run cauchy-gen benchmark \
  --config configs/preset_missingness_mar.yaml \
  --profile custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_missing_mar
```

When missingness is enabled, benchmark summaries include `missingness_guardrails` per profile,
and guardrail warnings/failures are reflected in overall regression status.

______________________________________________________________________

## Research & Roadmap

The development of `cauchy-generator` is strictly driven by recent literature in Tabular Deep Learning.

- **Meta-Feature Diagnostics:** A diagnostics module computes 15 structural metrics per dataset and aggregates coverage across generation runs. Soft steering is available to bias selection toward under-represented target bands.
- **Missingness Generation:** Configurable MAR/MCAR/MNAR mechanisms with CLI overrides and benchmark guardrails.
- **Shift-Aware SCMs:** Expanding the graph pipeline to support distribution shift and temporal drift.

See [docs/improvement_ideas.md](docs/improvement_ideas.md) for the prioritized research backlog.

______________________________________________________________________

## Theoretical Foundations

- **TabICLv2 (2026):** Core generation prior and Appendix E implementation.
- **TabPFN v2 (2025):** Insights into meta-feature coverage and tabular foundation model sensitivity.
- **TabICL (2025):** Methodology for complexity-based curriculum scheduling.

Detailed implementation notes and paper mappings can be found in [docs/implementation.md](docs/implementation.md).
