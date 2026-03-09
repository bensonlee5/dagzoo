# Keyed RNG Design For Semantic Reproducibility

This note defines the keyed RNG contract for `dagzoo` and the migration plan
for moving generation off order-coupled ambient `torch.Generator` usage.

Related docs:

- [`design-decisions.md`](design-decisions.md)
- [`roadmap.md`](roadmap.md)
- [`../how-it-works.md`](../how-it-works.md)

## Summary

`dagzoo` already has a strong deterministic seed-derivation baseline:
`SeedManager`, `derive_seed()`, and a small number of offset helpers isolate
major stages like dataset selection, fixed-layout planning, and missingness.

The remaining problem is semantic coupling inside those stages. Many helpers
still share one ambient `torch.Generator`, one batch generator, or one
offset-derived local seed. That means regrouping work, retrying one stage,
moving from scalar to batched execution, or changing an implementation detail
can perturb unrelated randomness even when the user-facing seed and config are
unchanged.

The goal of the keyed RNG epic is to preserve deterministic reproducibility
while making randomness follow semantic namespaces instead of call order.

## Target Contract

The migration target is **semantic reproducibility**:

- The same `(config, run seed, dataset index)` resolves the same semantic
  random substreams.
- Retrying one stage does not perturb sibling stages.
- Regrouping work for batching, converter grouping, or parent aggregation does
  not perturb unrelated randomness.
- Scalar and batched paths that implement the same typed plan consume the same
  semantic subkeys.
- Cross-device execution remains deterministic per device/backend contract, but
  bitwise equality across CPU/CUDA/MPS is not required.

This contract is stronger than the current “same draw order gives the same
result” baseline and is the compatibility target for `BL-134` through
`BL-137`.

## Current RNG Inventory

| Semantic stage                            | Current sites                                                                                                                                              | Current primitive                                                                                      | Current coupling risk                                                                          | Migration target                                                                                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run and dataset derivation                | `src/dagzoo/core/dataset.py`, `src/dagzoo/core/fixed_layout_runtime.py`, `src/dagzoo/config.py`                                                            | `SeedManager.child(...)`, `derive_seed(...)`                                                           | Good stage isolation today, but later helpers still mix keyed and offset-based derivation.     | Keep BLAKE2s seed derivation as the low-level primitive and expose keyed namespaces directly.                                                              |
| Attempt / node-spec / split helper seeds  | `src/dagzoo/core/generation_context.py`                                                                                                                    | `offset_seed32(...)`                                                                                   | Offset helpers encode sequencing assumptions and make namespace growth harder to reason about. | Replace new offset-only derivations with named keyed subpaths; keep `offset_seed32()` as legacy compatibility glue only.                                   |
| Layout and correlated layout sampling     | `src/dagzoo/core/layout.py`, `src/dagzoo/sampling/correlated.py`                                                                                           | Ambient `torch.Generator`, local CPU `manual_seed()` for Beta fallback                                 | Call-order coupling within layout sampling and correlated helper internals.                    | Layout-level keyed streams with explicit subkeys for assignments, cardinalities, and correlated helper fallback draws.                                     |
| Typed plan sampling                       | `src/dagzoo/core/execution_semantics.py`, `src/dagzoo/core/fixed_layout_batched.py`                                                                        | Ambient scalar draws on one generator, per-node plan/spec generators seeded from `SeedManager`         | Plan structure depends on local draw order inside each family and node planner.                | Key plan substreams by node, source, family, and nested component role.                                                                                    |
| Function / converter / node execution     | `src/dagzoo/functions/random_functions.py`, `src/dagzoo/functions/multi.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/core/fixed_layout_batched.py` | Shared ambient `torch.Generator` or `FixedLayoutBatchRng`                                              | Regrouping, stacking, or changing delegation paths can perturb unrelated randomness.           | Key execution draws by semantic role: root source, parent branch, function family component, converter spec, converter variant.                            |
| Noise-family selection and noise sampling | `src/dagzoo/core/noise_runtime.py`, `src/dagzoo/sampling/noise.py`                                                                                         | `SeedManager.torch_rng(...)`, ambient generator, local fallback CPU generator seeded from parent draws | Fallback sampling and mixture draws can remain tied to parent draw order.                      | Key runtime family selection separately from sample generation and make fallback subgenerators derive from explicit semantic subkeys.                      |
| Missingness and postprocess               | `src/dagzoo/sampling/missingness.py`, `src/dagzoo/postprocess/postprocess.py`, `src/dagzoo/core/generation_runtime.py`                                     | `SeedManager.torch_rng(...)`, split/postprocess generator seeded via `_split_permutation_seed(...)`    | Split, label remap, feature permutation, and missingness reuse a coarse stage seed.            | Split, postprocess, and missingness each get explicit keyed namespaces with train/test or mechanism subkeys.                                               |
| Benchmarks and reproducibility checks     | `src/dagzoo/bench/suite.py`, `src/dagzoo/bench/micro.py`                                                                                                   | `SeedManager`, `offset_seed32`, local manual seeds for microbenches                                    | Bench-specific offsets are deterministic but not semantically aligned with runtime namespaces. | Keep bench-only reproducibility independent from public generation, but move benchmark substreams onto named keys where they exercise generation behavior. |
| Diagnostics / audit utilities             | `src/dagzoo/diagnostics/effective_diversity.py`                                                                                                            | Local CPU generator factories and explicit `manual_seed()`                                             | Not part of the public generation contract, but still a source of duplicated RNG policy.       | Document as lower-priority follow-up unless it blocks the keyed runtime substrate.                                                                         |

## Proposed RNG Surface

`BL-134` should add a new preferred RNG substrate in `src/dagzoo/rng.py` while
keeping the hash primitive unchanged:

```python
@dataclass(slots=True, frozen=True)
class KeyedRng:
    seed: int
    path: tuple[str | int, ...] = ()

    def keyed(self, *components: str | int) -> "KeyedRng": ...
    def child_seed(self, *components: str | int) -> int: ...
    def torch_rng(
        self,
        *components: str | int,
        device: str = "cpu",
    ) -> torch.Generator: ...
```

Rules for the new surface:

- `derive_seed()` remains the hash-backed primitive.
- `KeyedRng.keyed(...)` is the preferred way to create semantic substreams.
- `SeedManager` remains as a compatibility wrapper and should delegate to
  `KeyedRng`, not remain a separate policy surface.
- `offset_seed32()` remains available only for existing compatibility seams and
  must not be used for new keyed namespaces.
- No new code may call `manual_seed()` on ad hoc local generators unless the
  seed came from an explicit keyed substream documented in the design.

## Namespace Tree

The keyed namespace tree for canonical public generation should be:

```text
run
└── dataset/{dataset_index}
    ├── rows
    ├── layout
    │   ├── graph
    │   ├── assignments
    │   ├── feature_types
    │   └── correlated
    ├── plan
    │   ├── node_spec/{node_index}
    │   └── node_plan/{node_index}
    ├── execution
    │   └── node/{node_index}
    │       ├── source
    │       ├── parent/{parent_index}
    │       ├── function
    │       ├── product/lhs
    │       ├── product/rhs
    │       └── converter/{spec_index}
    ├── split/{attempt_index}
    ├── postprocess/{attempt_index}
    │   ├── feature_permutation
    │   └── label_permutation
    ├── missingness/{attempt_index}
    │   ├── train
    │   └── test
    └── noise_runtime
        ├── family_selection
        └── samples
```

Benchmark and diagnostic namespaces should remain separate from public
generation:

```text
benchmark/{preset_key}/{phase}
diagnostics/{tool_name}/{arm_or_case}
```

These namespaces must not share generators with canonical generation paths.

## Stability And Non-Stability

The epic should preserve the following:

- Valid config behavior and default-off public interfaces.
- Emitted schema alignment, metadata shape, and replay metadata contracts.
- Deterministic output for the same post-migration keyed contract.
- Existing scalar-vs-batched typed-plan equivalence promises where tests
  already assert them.

The epic does **not** promise exact byte-for-byte parity with the legacy
generator-order implementation:

- Exact legacy numeric outputs may change when randomness moves to keyed
  substreams.
- Internal draw order inside plan sampling, grouped execution, or fallback
  sampling is not a compatibility target.
- Benchmark microbench helper seeds may move to named namespaces without
  preserving their current offset formulas.

Because the implementation tickets will change `src/dagzoo` behavior, merge
branches for `BL-134` and later must perform the repo-policy version bump and
`CHANGELOG.md` update.

## Ticket Breakdown

### BL-134: Keyed RNG substrate

- Add `KeyedRng` and delegation from `SeedManager`.
- Keep `derive_seed()` unchanged.
- Add RNG invariants for same-key stability and sibling-order independence.

### BL-135: Typed plan and executor migration

- Migrate `execution_semantics`, scalar helpers, node execution, and
  fixed-layout batch execution to semantic subkeys.
- Remove reliance on shared ambient draw order inside grouped execution.

### BL-136: Runtime and postprocess migration

- Migrate orchestration, noise selection, split permutation, missingness, and
  postprocess flows onto explicit keyed namespaces.
- Eliminate offset-only stage derivations from core runtime paths.

### BL-137: Hardening and docs

- Update user-facing reproducibility docs once the runtime contract is real.
- Add end-to-end regression coverage for regrouping, retries, replay, and
  benchmark reproducibility.

## Out Of Scope For BL-133

- No new config knobs.
- No changes to emitted metadata schema or artifact wire contracts.
- No runtime behavior changes in `src/dagzoo`.
- No attempt to migrate every diagnostics-only helper in the same slice; those
  should only move when they block the core keyed runtime substrate.
