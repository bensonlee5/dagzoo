# Implementation Details

## Objective
Build a Python repository that generates synthetic tabular datasets according to TabICLv2 Appendix E (`E.2`-`E.14`) with high throughput on NVIDIA CUDA GPUs and deterministic-seed best effort behavior.

## Source of Truth
- Normative behavior: `reference/TabICLv2.pdf` Appendix E.
- Clarification-only sources:  
  `reference/A Closer Look at TabPFN v2.pdf` and  
  `reference/Accurate predictions on small data with a tabular foundation model.pdf`.

## Public Interfaces
### Python API
- `generate_one(config: GeneratorConfig, seed: int, device: str) -> DatasetBundle`
- `generate_batch(config: GeneratorConfig, num_datasets: int, seed: int, device: str) -> list[DatasetBundle]`
- `write_parquet_shards(bundles, out_dir, shard_size, compression="zstd")`

### CLI
- `cauchy-gen generate --config ... --num-datasets ... --device cuda --seed ...`
- `cauchy-gen benchmark --suite standard --profile all --baseline ... --fail-on-regression`

### Output Contract
Each `DatasetBundle` contains:
- `X_train`, `y_train`, `X_test`, `y_test`
- `feature_types` (`"num"` or `"cat"`)
- metadata (seed lineage, graph stats, function selections, filter decision)

Persist generated outputs as Parquet shards with a sidecar metadata JSON per shard.

## Runtime Profiles
- `configs/default.yaml`: balanced local development profile.
- `configs/benchmark_cpu.yaml`: CPU benchmark profile.
- `configs/benchmark_cuda_desktop.yaml`: desktop CUDA benchmark profile.
- `configs/benchmark_cuda_h100.yaml`: H100 CUDA benchmark profile.
- `configs/preset_cuda_h100.yaml`: high-throughput datacenter preset.
- Runtime can auto-tune from a GPU FLOPS lookup table with unknown-device fallback behavior.

## Module Plan (Appendix Mapping)
- `sampling/correlated.py`: correlated scalar sampler (`E.2`)
- `core/dataset.py`: dataset orchestration (`E.3`)
- `graph/cauchy_graph.py`: random Cauchy DAG (`E.4`)
- `core/node_pipeline.py`: per-node flow (`E.5`)
- `converters/numeric.py`, `converters/categorical.py`: converters (`E.6`)
- `functions/multi.py`: concatenation vs per-parent aggregation (`E.7`)
- `functions/random_functions.py`: NN/tree/discretization/GP/linear/quadratic/EM/product (`E.8`)
- `functions/activations.py`: fixed + parametric activations (`E.9`)
- `linalg/random_matrices.py`: five matrix families and postprocessing (`E.10`)
- `sampling/random_weights.py`: positive normalized weights (`E.11`)
- `sampling/random_points.py`: base distributions + random function transform (`E.12`)
- `postprocess/postprocess.py`: cleanup, scaling, class/index permutation (`E.13`)
- `filtering/torch_rf_filter.py`: Torch-native RF OOB filter (`E.14`)

## Performance Strategy
1. Current generator path runs Torch on all devices (CPU/CUDA/MPS) and uses NumPy only for postprocessing contracts.
2. Keep kernels batch-oriented with vectorized NumPy operations and avoid Python loops in inner math paths.
3. Use optional filtering (`E.14`) behind config flags to avoid CPU bottlenecks in throughput benchmarks.
4. Profile with `bench/throughput.py` and track JSON baseline regressions by preset.
5. Next optimization step is lifting node/function kernels from NumPy to Torch/CUDA while preserving seeded behavior.

## Reproducibility Strategy
1. Global run seed -> per-dataset seed -> per-component derived seeds.
2. Central RNG utilities wrap Python/NumPy/Torch RNGs.
3. Document expected backend variation (best effort, not strict bitwise determinism).

## Validation and Benchmarks
### Correctness
- Unit invariants for ranges, shapes, DAG validity, converter class ranges, and matrix normalization.
- Integration tests for end-to-end classification/regression paths.

### Reproducibility
- Fixed seed should reproduce metadata exactly and numeric outputs within tolerance.

### Performance
- Benchmark suites: `smoke`, `standard`, `full`.
- Artifacts: JSON + Markdown summaries under `benchmarks/results/<timestamp>/`.
- Soft regression gate: warn at configurable threshold, fail only on severe regression with `--fail-on-regression`.

## Delivery Phases
1. Scaffold package/config/CLI + minimal generation loop.
2. Implement `E.2`-`E.7` plus unit tests.
3. Implement `E.8`-`E.10` with GPU-first tensor kernels.
4. Implement `E.11`-`E.14`, parquet writing, and integration tests.
5. Add benchmark harness, tune bottlenecks, and lock baseline.
