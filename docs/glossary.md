# Glossary

Domain terminology for the Cauchy generator, grouped by category.

______________________________________________________________________

## Graph & Structure

**adjacency matrix** — a square binary matrix where entry (i, j) = 1 means
node i is a parent of node j. Only the strict upper triangle is populated
(enforcing DAG structure with topological order matching index order).

**DAG (directed acyclic graph)** — the causal structure underlying each
generated dataset. Nodes represent latent computations; directed edges define
parent-child data flow. Acyclicity guarantees a valid topological ordering
for sequential evaluation.

**edge logit bias** — a scalar added to all edge logit values before
applying the sigmoid. Positive values increase global edge probability
(denser graphs); negative values decrease it (sparser graphs). Used by the
curriculum system to control structural complexity per stage.

**topological order** — an ordering of DAG nodes such that every parent
appears before its children. In this generator, node indices *are* the
topological order: node i can only have parents with index < i.

______________________________________________________________________

## Generation Pipeline

**activation family** — the set of nonlinear functions applied
element-wise within function families (e.g., tanh, ReLU, SiLU, softplus,
sign, sin, and parametric variants like signed-power). Each node randomly
selects activations during generation.

**aggregation** — the method used to combine outputs from multiple parent
nodes before applying a function. Supported strategies: sum, product, max,
and logsumexp.

**converter / ConverterSpec** — the final transformation that maps a slice
of a node's latent representation into an observable feature or target.
Each converter has a kind ("num" for continuous, "cat" for categorical,
"target_cls" for classification target, "target_reg" for regression
target), a dimensionality, and optionally a cardinality. See also:
*numeric converter*, *categorical converter*.

**categorical converter** — converts a latent slice into integer category
labels. Uses either nearest-center assignment (with Lp distance) or
softmax-based multinomial sampling.

**function family** — a class of random functions that a DAG node can apply
to its inputs. The eight families are: neural network, tree ensemble,
discretization, Gaussian process, linear, quadratic, EM-style soft
assignment, and product. See the design decisions doc for rationale.

**layout** — the random assignment of features and the target to DAG nodes,
plus the sampling of row counts, feature counts, and graph structure for a
single dataset. The layout is determined before any data flows through the
graph.

**multi-function** — the mechanism for nodes with multiple parents.
Concatenates parent tensors and applies a single function, or transforms
each parent independently and then aggregates.

**node pipeline** — the full per-node computation: sample or receive parent
data, apply a random function, then run converters for each assigned
feature/target.

**numeric converter** — converts a latent slice into a continuous feature
value. Either passes through as identity or applies a Kumaraswamy-like
monotonic warping.

**ODT (oblivious decision tree)** — a decision tree variant where all nodes
at the same depth share the same splitting feature and threshold. Used in
the tree function family, where ensembles of oblivious trees use shared
per-depth, axis-aligned splits.

______________________________________________________________________

## Complexity Control

**curriculum** — a system that controls dataset complexity across a
generation run. Organizes generation into stages with increasing structural
and statistical complexity.

**curriculum stage** — one of three complexity tiers (1, 2, 3) or a mode
selector ("auto" for random per-dataset, "off" to disable). Stage 1
produces small, sparse datasets; stage 3 produces large, dense ones. Each
stage defines row count ranges, train/test split ratios, and an edge logit
bias.

**stage bounds** — per-stage constraints on layout parameters: min/max
feature count, min/max node count, and min/max graph depth. These bounds
are merged with base config bounds and enforced during layout sampling.

______________________________________________________________________

## Quality & Steering

**softmax selection** — the method for choosing among steering candidates.
Candidate scores are negated, divided by temperature, and passed through
softmax to produce selection probabilities. Lower scores (better fit) get
higher probability, but every candidate retains a nonzero chance.

**steering** — the system that guides the generated dataset distribution
toward target meta-feature bands. Generates multiple candidates per dataset
slot, scores each against target specifications, and selects
probabilistically via softmax.

**steering candidate** — one of several independently generated datasets
competing for a single slot in the output. Each candidate is scored against
the target bands; one is selected via softmax.

**target band** — a desired range [lo, hi] for a meta-feature, plus a
weight indicating the relative importance of coverage for that metric.
Candidates are scored by their distance to the band; values inside the band
score zero.

**temperature** — the softmax scaling parameter in steering selection.
Lower temperature favors the best-scoring candidate (more greedy); higher
temperature gives more uniform selection (more exploratory).

**under-coverage reweighting** — a feedback mechanism where metrics with low
in-band coverage across previously selected datasets receive progressively
higher effective weight in the scoring function. Drives the system toward
balanced coverage across all target bands.

______________________________________________________________________

## Filtering

**learnability filter** — a quality gate that checks whether a generated
dataset contains learnable signal. Trains a lightweight random forest on the
training split, computes out-of-bag predictions, and measures whether the
forest outperforms a baseline predictor via bootstrapped comparison.

**wins ratio** — the fraction of bootstrap resamples in which the random
forest's OOB predictions have lower MSE than the baseline (predicting the
training mean). A dataset passes the filter when its wins ratio exceeds a
configurable threshold (default 0.95).

______________________________________________________________________

## Missingness

**MAR (missing at random)** — a missingness mechanism where the probability
of a value being missing depends on other *observed* features in the same
row, but not on the missing value itself. Implemented via a random weight
matrix over a designated observed feature subset, with logit-scale
calibration to match the target rate.

**MCAR (missing completely at random)** — a missingness mechanism where
each cell is independently missing with a fixed probability equal to the
target rate. The simplest mechanism; missingness is unrelated to any data
values.

**missingness mask** — a binary tensor indicating which cells are missing.
Applied after generation by replacing masked cells with NaN.

**MNAR (missing not at random)** — a missingness mechanism where the
probability of a value being missing depends on the value itself. Each
feature gets a random weight; its own standardized value is scaled by that
weight to produce a logit, calibrated to match the target rate.

______________________________________________________________________

## Metrics

**categorical ratio** — the fraction of features that are categorical
(integer-coded) rather than continuous.

**class entropy** — Shannon entropy of the class distribution in a
classification dataset. Higher entropy indicates more balanced classes.

**coverage band** — synonymous with *target band*. The desired range for
a meta-feature that the steering system tries to cover.

**linearity proxy** — a ridge regression bootstrap wins ratio, measuring
how well a linear model predicts the target on the test split.

**meta-feature** — a scalar summary statistic computed on a generated
dataset (e.g., linearity proxy, class entropy, SNR proxy). Used by the
steering system to assess dataset characteristics and guide selection.

**nonlinearity proxy** — the gap between the random forest wins ratio and
the linearity proxy, floored at zero. Measures the degree of nonlinear
signal present beyond what a linear model captures.

**SNR proxy** — signal-to-noise ratio estimated from a ridge regression.
Computed as the ratio of prediction variance to residual variance,
expressed in decibels (dB).

______________________________________________________________________

## Reproducibility

**child seed** — a deterministic seed derived from a parent seed and a
component path via BLAKE2s hashing. Enables each component to have its own
independent RNG stream while remaining fully determined by the base seed.

**component path** — the sequence of string and integer identifiers passed
to the seed manager to derive a child seed (e.g., "dataset", 42, "layout").
Different paths produce uncorrelated seeds.

**seed derivation** — the process of deterministically computing a child
seed from a base seed and component path using BLAKE2s. See the design
decisions doc for why BLAKE2s.

**SeedManager** — the central object for deterministic randomness. Holds a
base seed and provides methods to derive child seeds and create seeded
`torch.Generator` instances for any component path.

______________________________________________________________________

## Output

**DatasetBundle** — the in-memory container for one generated dataset.
Holds train/test feature tensors, train/test target tensors, a
feature-type list, and a metadata dictionary. See the output format doc
for the full specification.

**lineage artifact** — on-disk record of a dataset's causal graph. Stores
the DAG adjacency matrix in bitpacked form with SHA-256 checksums, plus
the mapping from features/target to graph nodes.

**shard** — a directory containing a batch of generated datasets (default
128 per shard) plus shared lineage artifacts. Shards are the unit of
on-disk organization.

______________________________________________________________________

## Infrastructure

**hardware profile** — auto-detected classification of the compute
environment (e.g., "cuda_h100", "cuda_datacenter", "cuda_desktop", "cpu").
Used to set sensible defaults for batch sizes, feature counts, and other
parameters that should scale with available hardware.

**profile tier** — synonymous with *hardware profile*. The specific tier
name assigned based on detected GPU peak FLOPS and backend.
