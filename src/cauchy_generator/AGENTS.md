# Package Agent Notes

## Architecture

- `core/`: orchestration for dataset generation.
- `sampling/`, `functions/`, `converters/`, `linalg/`: Appendix E math kernels.
- `postprocess/`, `filtering/`: post-generation constraints and quality checks.
- `io/`: output serialization.
- `bench/`: benchmark harness and reporting.
- Steering candidate scoring path: `core/steering_metrics.py` (torch-native subset).
- Full diagnostics extraction path: `diagnostics/metrics.py` (reporting-focused, NumPy-based).

## Guardrails

- Keep seed lineage stable (`SeedManager` paths must stay deterministic).
- Preserve metadata shape/keys unless intentionally versioned.
- Validate device semantics (`auto|cpu|cuda|mps`) and fail fast for explicit unavailable accelerators.
- Keep strict JSON output (`allow_nan=False` behavior must remain intact).

## Performance

- Keep hot paths vectorized.
- Avoid per-row Python loops in generation kernels.
- Prefer streaming iteration for large dataset counts.
