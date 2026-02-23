# Core Agent Notes

## Files

- `dataset.py`: device resolution, layout sampling, one/batch generation.
- `node_pipeline.py`: per-node execution pipeline.

## Rules

- Preserve deterministic behavior for `generate_one`, `generate_batch`, and `generate_batch_iter`.
- Keep explicit `cuda`/`mps` requests fail-fast when unavailable.
- Keep `generate_batch_iter` lazy; do not reintroduce full-batch materialization in CLI paths.
- Keep Torch tensor output contracts stable (shape/type/metadata keys).
- Keep steering candidate scoring on torch-native metric paths; do not reintroduce tensor CPU/NumPy round trips in selection flow.
- Preserve deterministic softmax candidate selection behavior for fixed seeds.

## Testing Focus

- device selection and fail-fast paths
- reproducibility for fixed seeds
- classification split validity constraints
