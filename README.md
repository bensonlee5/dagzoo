# cauchy-generator

High-performance synthetic tabular data generator scaffold targeting TabICLv2 Appendix E (`E.2`-`E.14`).

## Quickstart (uv)

```bash
uv sync --group dev
uv run pre-commit install
uv run cauchy-gen generate --config configs/default.yaml --out data/run1 --num-datasets 10
uv run cauchy-gen benchmark --suite smoke --profile cpu
uv run cauchy-gen benchmark --suite standard --profile cpu --out-dir benchmarks/results/latest
```

## Accessible Scripts

Use the wrappers in `scripts/` for common generation workflows:

```bash
./scripts/generate-default.sh
./scripts/generate-h100.sh 500 cuda data/run_h100_500
./scripts/generate-from-config.sh configs/default.yaml 25 auto data/run_custom 42
./scripts/generate-smoke.sh configs/default.yaml 3 cpu
```

See `scripts/README.md` for argument details.

## Presets

- `configs/preset_cuda_h100.yaml`: high-throughput profile for H100-class GPUs.
- `configs/benchmark_cpu.yaml`: benchmark profile for CPU-only runs.
- `configs/benchmark_cuda_desktop.yaml`: benchmark profile for desktop CUDA GPUs.
- `configs/benchmark_cuda_h100.yaml`: benchmark profile for H100-class GPUs.

## Config Notes

- Filter config uses RF field names: `filter.n_trees` and `filter.depth`.
- Legacy keys (`filter.n_estimators`, `filter.max_depth`) are intentionally rejected.

## Benchmark Suite

Run profile-specific or matrix benchmark suites:

```bash
uv run cauchy-gen benchmark --suite smoke --profile cpu
uv run cauchy-gen benchmark --suite standard --profile all
uv run cauchy-gen benchmark --config configs/benchmark_cuda_desktop.yaml --profile custom --baseline benchmarks/baselines/desktop.json --fail-on-regression
```

`--profile all` includes CUDA profiles and will hard-fail on machines without CUDA.

Each run writes `summary.json` and `summary.md` under `benchmarks/results/<timestamp>/` by default.

CI automation is configured in `.github/workflows/benchmark.yml`:
- PRs: smoke benchmark (`cpu`) with sticky PR comment and artifact upload.
- Schedule/manual: standard benchmark (`cpu`) with baseline regression checks and artifact upload.

## References

- Existing core papers are stored in `reference/`.
- Additional curated papers and rationale are indexed in `reference/ADDITIONAL_PAPERS.md`.
- Re-fetch the additional papers with `./scripts/fetch-additional-references.sh`.

## Hardware Awareness

`cauchy-gen` uses a FLOPS lookup table (model-name matching) to classify GPUs and auto-tune profile settings.  
Unknown GPUs fall back safely (`peak_flops=inf`) so utilization metrics do not report misleading values.

## Status

This repository currently provides:
- package/CLI/config scaffolding
- Torch-required seeded batch generation (CPU/CUDA/MPS)
- Parquet writing interface
- benchmark harness and test scaffolding

See `docs/implementation.md` for the full Appendix E implementation plan.
