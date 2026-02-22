# Benchmark Agent Notes

## Goals

- Measure throughput/latency/reproducibility without changing generation semantics.
- Keep benchmark runs comparable across CI and local environments.

## Rules

- Prefer streaming generation in benchmark execution to avoid OOM at scale.
- Keep regression gating metric directions accurate (`higher_is_better` vs `lower_is_better`).
- Do not silently swallow benchmark failures caused by explicit unavailable devices.

## Validation

- Run benchmark unit tests in `tests/test_benchmark_*.py`.
- Ensure summary JSON remains serializable with finite values or explicit `null`.
