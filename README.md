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

Unlike traditional synthetic generators that rely on independent feature sampling, `cauchy-generator` builds datasets through **Cauchy DAG-based execution** (Directed Acyclic Graph).

- **Hierarchical Dependencies:** Features are generated as nodes in a graph, where child values are non-linear functional transforms of their parents.
- **Deep Mechanism Families:** Functional relationships incorporate Neural Networks, Tree ensembles, and parametric activations, ensuring complex multi-order interactions.

### 2. Tabular Realism

The generator incorporates the "messiness" of real-world tabular data directly into its priors:

- **Mixed-Type Converters:** Native support for high-cardinality categorical features and multi-scale numeric data.
- **Post-processing Hooks:** Real-world scaling, outlier clipping, and class-permutation logic to mimic benchmark data (e.g., OpenML, Kaggle).
- **Complexity Curriculum:** Optional staged generation that scales dataset complexity (features, nodes, samples) to support progressive learning.

### 3. Hardware-Native Performance

Performance is a first-class citizen. The engine is built for datacenter-scale throughput:

- **Torch-Native Pipeline:** Optimized for NVIDIA **CUDA** and Apple Silicon **MPS**, minimizing Python-level bottlenecks.
- **Hardware-Aware Scaling:** Auto-detects GPU capabilities (e.g., H100 vs. RTX 4090) to tune generation parameters and maximize utilization.
- **Parallel Streaming:** Native Parquet sharding for efficient data loading during model training.

______________________________________________________________________

## How cauchy-generator Works

The current baseline generation flow is:

1. Load config and resolve hardware-aware runtime defaults.
1. Derive deterministic run, dataset, and component seeds.
1. Sample curriculum stage, layout, and DAG structure.
1. Execute node pipelines in topological order to generate latent signals.
1. Convert node outputs into observable features and targets.
1. Apply optional filtering, postprocessing, and missingness injection.
1. Emit `DatasetBundle` outputs, then optionally persist Parquet shards and diagnostics artifacts.

Read the full walkthrough and terminology guide: [docs/how-it-works.md](docs/how-it-works.md).

______________________________________________________________________

## Quick Start

### Installation

```bash
uv tool install cauchy-generator
```

### Basic Generation

```bash
# Generate 10 datasets using the default high-quality prior
cauchy-gen generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

### Steering & Realism

```bash
# Enable meta-feature steering for specific linearity and SNR (Signal-to-Noise Ratio) targets
cauchy-gen generate \
  --config configs/default.yaml \
  --steer-meta \
  --meta-target linearity_proxy=0.25:0.75:1.0 \
  --meta-target snr_proxy_db=10:30:1.5 \
  --num-datasets 25
```

```bash
# Inject 20% MAR (Missing At Random) values with custom logit scale
cauchy-gen generate \
  --missing-rate 0.2 \
  --missing-mechanism mar \
  --missing-mar-logit-scale 1.5 \
  --num-datasets 10
```

### Benchmarking

```bash
# Run a quick smoke benchmark on CPU
cauchy-gen benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
```

### Many-Class Workflows

```bash
# Generate many-class datasets within the current rollout envelope (<=32 classes)
cauchy-gen generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 10 --out data/run_many_class

# Benchmark many-class smoke performance and guardrails
cauchy-gen benchmark --config configs/preset_many_class_benchmark_smoke.yaml --profile custom --suite smoke --no-memory --out-dir benchmarks/results/smoke_many_class
```

### Inspect Hardware

```bash
# Show detected compute backend, GPU model, and performance profile
cauchy-gen hardware
```

______________________________________________________________________

## Documentation Map

- [docs/how-it-works.md](docs/how-it-works.md): System walkthrough and terminology.
- [docs/usage-guide.md](docs/usage-guide.md): End-user workflows for generation and benchmark runs.
- [docs/output-format.md](docs/output-format.md): Output schema and persistence contract.
- [docs/design-decisions.md](docs/design-decisions.md): Architecture decisions and evolution policy.
- [docs/roadmap.md](docs/roadmap.md): Canonical roadmap and current implementation baseline.
- [docs/backlog_decision_rules.md](docs/backlog_decision_rules.md): Ranking rubric and go/no-go implementation gates.
- [reference/literature_evidence_2026.md](reference/literature_evidence_2026.md): Evidence appendix mapped to roadmap items.

______________________________________________________________________

## Python API

```python
from cauchy_generator import GeneratorConfig, generate_one

config = GeneratorConfig.from_yaml("configs/default.yaml")
bundle = generate_one(config, seed=42)

print(bundle.X_train.shape)      # (n_train, n_features)
print(bundle.feature_types)      # ["num", "cat", "num", ...]
```

For bulk generation, use `generate_batch` (returns an eager list) or `generate_batch_iter` (returns a lazy iterator):

```python
from cauchy_generator import GeneratorConfig, generate_batch, generate_batch_iter

config = GeneratorConfig.from_yaml("configs/default.yaml")

bundles = generate_batch(config, num_datasets=50, seed=0)           # list[DatasetBundle]
for bundle in generate_batch_iter(config, num_datasets=500, seed=0):  # lazy iterator
    process(bundle)
```

______________________________________________________________________

## Configuration Presets

| Category    | Example file                                     | What it controls                                                 |
| ----------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| Default     | `configs/default.yaml`                           | Balanced baseline for classification                             |
| Curriculum  | `configs/preset_curriculum_auto_staged.yaml`     | Staged difficulty (features, nodes, depth)                       |
| Missingness | `configs/preset_missingness_mcar.yaml`           | Synthetic missing-data injection (**MCAR**/MAR/MNAR - see below) |
| Shift       | `configs/preset_shift_mixed_generate_smoke.yaml` | Opt-in graph/mechanism/noise shift workflows                     |
| Steering    | `configs/preset_steering_conservative.yaml`      | Soft selection toward meta-feature target bands                  |
| Many-class  | `configs/preset_many_class_generate_smoke.yaml`  | High-class-count generation within 32-class envelope             |
| Benchmark   | `configs/benchmark_cpu.yaml`                     | Hardware-specific benchmark parameters                           |

Presets follow the `preset_<category>_<variant>.yaml` naming convention. You can compose presets by layering CLI `--config` with flag overrides.

______________________________________________________________________

## Feature Highlights

- **Configurable Missingness:** Native support for **MCAR** (Missing Completely At Random), **MAR** (Missing At Random), and **MNAR** (Missing Not At Random) mechanisms with deterministic seeded behavior and benchmark guardrails.
- **Complexity Curriculum:** Stage-aware scaling of row counts, feature counts, node counts, and graph depth to support progressive model training.
- **Configurable Shift/Drift:** Opt-in graph/mechanism/noise shift profiles with interpretable scale semantics and deterministic seed behavior.
- **Meta-Feature Steering:** soft-steering loop that biases generation toward target meta-feature bands (e.g., specific linearity or SNR ranges) using under-coverage reweighting.
- **Lineage Integrity:** Every dataset includes a versioned DAG lineage artifact with adjacency matrices and feature-to-node assignments.

______________________________________________________________________

## Theoretical Foundations

- **TabICLv2 (2026):** Core generation prior and Appendix E implementation.
- **TabPFN v2 (2025):** Insights into meta-feature coverage and tabular foundation model sensitivity.
- **TabICL (2025):** Methodology for complexity-based curriculum scheduling.

Implementation baseline and paper mappings can be found in [docs/roadmap.md](docs/roadmap.md) and [reference/literature_evidence_2026.md](reference/literature_evidence_2026.md).
