# Design Decisions

Lightweight Architecture Decision Records (ADRs) for foundational choices
in the Cauchy generator.

______________________________________________________________________

## 1. Cauchy distribution for DAG edge probabilities

### Context

The generator builds random DAGs to define causal structure. Each potential
edge needs a probability. Standard Erdős–Rényi graphs sample each edge with
the same fixed probability, producing homogeneous structure with thin-tailed
degree distributions.

### Decision

Edge probabilities follow an additive logit model on the Cauchy distribution:

```
p_ij = sigmoid(A + B_i + C_j + edge_logit_bias)
```

where A is a scalar, B_i is per-source-node, and C_j is per-target-node, all
drawn from the standard Cauchy distribution. Only upper-triangular entries are
kept to enforce acyclicity.

This documents the current baseline implementation and may be extended via
future roadmap work while preserving backward-compatible defaults.

### Rationale

- **Heavy tails** — the Cauchy distribution produces occasional extreme
  values, creating natural variability in graph density within and across
  datasets. Some nodes become hubs; others stay sparse.
- **Node-level heterogeneity** — separate per-row (B_i) and per-column (C_j)
  terms let individual nodes have distinct connectivity profiles.
- **Global sparsity control** — the `edge_logit_bias` additive term shifts
  the entire probability surface up or down, which the curriculum system uses
  to produce sparser graphs at lower stages and denser ones at higher stages.
- **Theoretical grounding** — directly implements the mechanism described in
  TabICLv2 Appendix E.4.

### Alternatives considered

- **Erdős–Rényi (uniform p)** — too homogeneous; every node looks the same.
- **Power-law degree sequences** — adds structural diversity but requires a
  separate degree-to-DAG conversion step and does not naturally give per-edge
  control.
- **Normal distribution for logits** — lighter tails suppress the extreme hub
  / isolate patterns that create interesting downstream data.

______________________________________________________________________

## 2. Torch-only generation pipeline (no NumPy)

### Context

The data generation pipeline involves sampling random graphs, applying
function families, running converters, and postprocessing. These operations
could use NumPy, PyTorch, or a mix.

### Decision

All tensor computation in the generation pipeline uses PyTorch exclusively.
NumPy is used only at the I/O boundary (Parquet serialization).

This reflects the current baseline execution path; future roadmap work may
expand mechanism/noise controls (RD-011/RD-012) while keeping generation
backend strategy explicit.

### Rationale

- **Device-agnostic** — the same code path runs on CPU, CUDA, and MPS
  without if/else branching for backend.
- **No transfer bottlenecks** — a mixed pipeline requires CPU↔GPU copies at
  every NumPy↔Torch boundary. Keeping everything in Torch avoids this.
- **Single RNG system** — `torch.Generator` provides deterministic,
  device-aware random number generation. Mixing in `numpy.random` would
  require maintaining two RNG states and reasoning about their interaction.

### Alternatives considered

- **NumPy-first** — simpler for CPU-only use, but forfeits GPU acceleration
  entirely.
- **Mixed NumPy + Torch** — maximizes library ergonomics per operation but
  introduces transfer overhead and dual-RNG complexity.

______________________________________________________________________

## 3. BLAKE2s for seed derivation

### Context

The generator uses a tree of deterministic seeds: one base seed spawns child
seeds for each component (graph sampling, function selection, data generation,
steering, missingness, etc.). The derivation function must map
`(base_seed, component_path)` to a child seed without collisions.

### Decision

Child seeds are derived via BLAKE2s hashing: encode the base seed and path
components as UTF-8, hash them, and extract a 32-bit integer from the digest.

This is the current seeded reproducibility baseline and can be revisited only
with an explicit migration plan for compatibility.

### Rationale

- **No collisions** — a cryptographic hash ensures that nearby inputs (e.g.,
  seeds 1000 and 1001, or paths "feature/0" and "feature/1") produce
  uncorrelated outputs. Simple arithmetic (addition, XOR) leaks input
  structure into the derived seed.
- **Uniform output** — BLAKE2s output bits are statistically uniform, so
  derived seeds cover the full 32-bit range evenly.
- **Fast** — BLAKE2s is specifically optimized for short inputs. The overhead
  per derivation is negligible compared to the tensor operations that follow.
- **Deterministic** — given the same base seed and component path, the
  derived seed is always identical, which is the foundation of reproducibility.

### Alternatives considered

- **Linear combination / XOR** — fast but produces structured collisions
  (e.g., `seed ^ 1` and `(seed+1) ^ 0` can alias).
- **SHA-256** — equally correct but slower than BLAKE2s for the short-input
  use case.
- **Counter-based (Philox/Threefry)** — good for bulk random streams but
  less natural for tree-structured seed hierarchies with string-named
  components.

______________________________________________________________________

## 4. Softmax steering instead of hard rejection

### Context

Generated datasets need to cover target distributions of meta-features (e.g.,
linearity, class balance, feature correlations). One approach is to generate
candidates and reject those outside the target bands. The alternative is to
score candidates and select probabilistically.

### Decision

The steering system generates multiple candidates per dataset slot, scores
each candidate against target meta-feature bands, and selects via softmax
(temperature-scaled multinomial sampling) with under-coverage reweighting.

This describes the current selection policy and can evolve as long as
steering remains opt-in and benchmark-guarded.

### Rationale

- **No wasted budget** — hard rejection discards all work on "close but not
  quite" candidates. With softmax, every candidate has a nonzero selection
  probability proportional to how well it fits.
- **Exploration** — even a poor-scoring candidate can be selected, preventing
  the system from getting stuck in narrow meta-feature corridors.
- **Tunable greediness** — the temperature parameter controls the tradeoff:
  low temperature approximates argmin (greedy), high temperature approximates
  uniform random.
- **Adaptive reweighting** — metrics with low in-band coverage get
  progressively higher effective weight in the scoring function. This creates
  a feedback loop that pushes under-represented regions toward the target
  without explicit constraint solving.

### Alternatives considered

- **Hard accept/reject** — simple to implement but wasteful. At tight target
  bands, rejection rates can exceed 90%.
- **Constrained optimization** — could directly solve for parameters that
  produce target meta-features, but the generation pipeline is not
  differentiable end-to-end (discrete graph structure, categorical
  converters).
- **Importance sampling** — reweight after generation. This preserves all
  candidates but requires maintaining and normalizing weights, which is more
  complex than softmax selection.

______________________________________________________________________

## 5. `slots=True` dataclasses

### Context

The generator creates thousands of intermediate dataclass instances per batch
(dataset bundles, converter specs, seed managers, metrics containers, etc.).
Python dataclasses default to storing attributes in a per-instance `__dict__`.

### Decision

All dataclasses in the codebase use `@dataclass(slots=True)`.

This is a codebase-wide convention today, not a protocol contract.

### Rationale

- **Memory** — eliminates the per-instance `__dict__` allocation. When
  generating thousands of dataset bundles in a batch, the cumulative savings
  are meaningful.
- **Speed** — slot-based attribute access is a direct offset lookup instead
  of a dictionary lookup.
- **Safety** — prevents accidental attribute addition (e.g., typos like
  `bundle.metdata = ...`), signaling that these are structured containers
  with a fixed schema.

### Alternatives considered

- **NamedTuples** — immutable and memory-efficient, but lack default values
  and mutation support that dataclasses provide.
- **Plain dicts** — no attribute typo protection and no type annotation
  support.
- **`__slots__` without dataclasses** — equivalent runtime behavior but
  requires manual `__init__`, `__repr__`, etc.

______________________________________________________________________

## 6. Current function-family baseline

### Context

Each DAG node applies a random function to its parent values to produce a
latent representation. The choice of available function families determines
the space of possible data-generating processes.

### Decision

Current default family set includes eight families: neural network, tree ensemble, discretization (nearest
center), Gaussian process (random Fourier features), linear projection,
quadratic forms, EM-style soft assignment, and product (elementwise product
of two sub-families). Multi-parent nodes additionally use aggregation
strategies (sum, product, max, logsumexp).

Family and parameterization expansion is an explicit roadmap direction
(RD-011), with related noise-family expansion tracked separately (RD-012).

### Rationale

- **Theoretical grounding** — implements the mechanism families from
  TabICLv2 Appendix E.8.
- **Diversity across axes** — linear vs. nonlinear, smooth vs. piecewise,
  sparse vs. dense interactions. This diversity is necessary for the
  generated data to cover the space of real-world tabular relationships.
- **Multiplicative interactions** — the product family creates interaction
  effects not achievable by any single family, which is critical for
  modeling feature interactions in tabular data.
- **Composability** — multi-parent aggregation and the product family enable
  higher-order compositions without an exponential explosion in family count.

### Alternatives considered

- **Fewer families (e.g., NN-only)** — simpler but produces a narrower
  distribution of data characteristics. Neural networks alone struggle to
  produce sharp piecewise or nearest-neighbor-like structures.
- **More families** — diminishing returns on diversity; each new family adds
  code and testing surface. The current set covers the major structural
  archetypes.
- **Symbolic regression / GP trees** — expressive but hard to control for
  numerical stability and output distribution.

______________________________________________________________________

## 7. Noise family selection for RD-012 phase 1

### Context

Current generation uses implicit Gaussian-driven stochasticity throughout
matrix, weight, and point sampling, with optional global variance scaling via
`noise_sigma_multiplier` from shift controls. Epic `#24` and issue `#25`
introduce explicit user-facing noise-family configuration.

### Decision

Issue `#25` should start with five selectable modes:

- `legacy` (default compatibility mode)
- `gaussian`
- `laplace`
- `student_t`
- `mixture`

`mixture` is constrained to weighted combinations of Gaussian, Laplace, and
Student-t components only.

### Rationale

- **Backward compatibility first** — `legacy` default preserves existing config
  behavior and seeded reproducibility for users who do not opt in.
- **Coverage without excessive surface area** — Gaussian, Laplace, and
  Student-t provide progressively heavier tails with interpretable behavior.
- **Controlled flexibility** — `mixture` allows blended regimes without opening
  an unbounded family space in phase 1.
- **Numerical stability guardrails** — `student_t` should require `df > 2`
  to ensure finite variance and avoid unstable extreme draws.

### Alternatives considered

- **Expose Cauchy directly in phase 1** — rejected for now due infinite
  variance and high instability risk in downstream transforms.
- **Single-family only (Gaussian)** — rejected because it does not close the
  realism/robustness coverage gap targeted by RD-012.
- **Large family menu initially** — rejected to keep validation, docs, and
  benchmarking tractable for first delivery.

______________________________________________________________________

## Evolution Policy

- These ADRs document the current baseline implementation and rationale.
- Roadmap items may supersede ADR specifics; `development/roadmap.md` is authoritative
  for planned evolution.
- When roadmap delivery changes a decision detail materially, update this file
  in the same PR and record the rationale.
