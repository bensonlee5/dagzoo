# Dagsynth Functional Transforms (Canonical)

This document is the canonical mathematical specification for the generation transforms used by `dagsynth`.
Equations are implementation-faithful to the current runtime in `src/dagsynth`.

## Notation and Symbols

**Primary Variable:** Symbol map (this section's notation table).<br>
**Dependency Map:** all symbols used in later equations -> definitions in this table.<br>
**Path to Final Output:** unambiguous symbols -> unambiguous equations -> correct interpretation of how transforms produce `X`, `y`, and metadata.<br>

**Rationale.** This table fixes symbol meanings up front so equations stay unambiguous across sections. Symbol names mirror runtime naming in core modules, making the spec easier to trace back to code.

| Symbol                               | Meaning                                                                                                          |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| $N$                                  | Number of latent DAG nodes.                                                                                      |
| $n$                                  | Number of rows sampled for one dataset attempt ($n = n\_{\\text{train}} + n\_{\\text{test}}$).                   |
| $G \\in {0,1}^{N \\times N}$         | DAG adjacency matrix with convention `G[src, dst]`.                                                              |
| $i, j$                               | Node indices in ${0,\\dots,N-1}$.                                                                                |
| $\\operatorname{pa}(j)$              | Parent index set for node $j$: ${i : G\_{ij}=1}$.                                                                |
| $\\sigma(x)$                         | Logistic sigmoid $(1 + e^{-x})^{-1}$.                                                                            |
| $A, B_i, C_j$                        | DAG latent logits: $A$ is global edge propensity, $B_i$ is source-node effect, $C_j$ is destination-node effect. |
| $\\beta\_{\\text{edge}}$             | Edge logit bias applied in DAG sampling.                                                                         |
| $\\tau$                              | Mechanism logit tilt (`mechanism_logit_tilt`).                                                                   |
| $\\lambda_f$                         | Raw mechanism family weight for family $f$ from `mechanism.function_family_mix`.                                 |
| $\\pi_f^{(0)}$                       | Baseline mechanism-family probability before logit tilt.                                                         |
| $\\pi_f$                             | Final sampling probability for mechanism family $f$.                                                             |
| $\\ell_f$                            | Family base logit for family $f$ (from `MECHANISM_FAMILY_BASE_LOGITS`).                                          |
| $\\tilde{\\ell}\_f$                  | Centered family base logit: $\\ell_f - \\frac{1}{\\lvert\\mathcal{F}\\rvert}\\sum\_{g\\in\\mathcal{F}}\\ell_g$.  |
| $\\mathcal{F}$                       | Ordered mechanism family set: `(nn, tree, discretization, gp, linear, quadratic, em, product)`.                  |
| $Z_j \\in \\mathbb{R}^{n \\times d}$ | Node-$j$ latent tensor after node pipeline transforms.                                                           |
| $d\_{\\text{req}}$                   | Required latent width from converter specs: $\\sum_s \\max(1, d_s)$.                                             |
| $d\_{\\text{extra}}$                 | Additional sampled latent width: $\\max(1, \\lfloor \\operatorname{LogUniform}(1,32) \\rfloor)$.                 |
| $d$                                  | Node latent width used for mechanism output: $d = d\_{\\text{req}} + d\_{\\text{extra}}$.                        |
| $d_s$                                | Width requested by converter spec $s$ before max-with-1 clamp.                                                   |
| $w \\in \\mathbb{R}^d$               | Positive random weight vector (normalized to sum 1, then permuted).                                              |
| $X$                                  | Generic mechanism input matrix.                                                                                  |
| $Y$                                  | Generic mechanism output matrix.                                                                                 |
| $p\_{ij}$                            | DAG edge probability for ordered pair $(i,j)$ with $i < j$.                                                      |
| $\\gamma\_\\sigma$                   | Noise standard-deviation multiplier from shift runtime (`variance_sigma_multiplier`).                            |
| $\\gamma\_{\\text{var}}$             | Noise variance multiplier: $\\gamma\_{\\text{var}} = \\gamma\_\\sigma^2$.                                        |
| $\\mathcal{D}\_{\\text{noise}}$      | Active runtime noise family (`gaussian`, `laplace`, or `student_t` after runtime selection).                     |
| $C$                                  | Categorical cardinality for a converter target.                                                                  |
| $\\operatorname{LSE}$                | LogSumExp along parent axis.                                                                                     |
| $T$                                  | Number of trees in a sampled tree ensemble.                                                                      |
| $P$                                  | Number of random features in GP approximation (fixed at 256).                                                    |
| $s_j$                                | Final node-level multiplicative latent scale.                                                                    |
| $\\mathcal{O}$                       | Final attempt output tuple: $(X, y, \\text{metadata})$.                                                          |

### Operator Reference

| Operator                                 | Definition                                                                                 | Input/Output Shape Semantics                                                       | Where Used                        | Output Impact                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| $\\operatorname{pa}(j)$                  | Parent index set of node $j$: ${i: G\_{ij}=1}$.                                            | Input: scalar node index $j$ and adjacency $G$; Output: set of parent indices.     | Section 1 and Section 4.          | Determines which upstream node tensors are composed for node $j$.                      |
| $\\operatorname{Agg}(Y_1,\\dots,Y_k)$    | Multi-parent aggregation operator sampled from ${\\sum,\\prod,\\max,\\operatorname{LSE}}$. | Input: parent-aligned tensors $(n \\times d)$; Output: one $(n \\times d)$ tensor. | Section 4.2.                      | Changes how parent effects combine before downstream mechanism/converter steps.        |
| $\\operatorname{LSE}(Y_1,\\dots,Y_k)$    | LogSumExp over parent axis: $\\log\\sum_r e^{Y_r}$.                                        | Input: stacked parent outputs; Output: aggregated tensor $(n \\times d)$.          | Section 4.2.                      | Smoothly emphasizes larger parent responses while retaining contributions from others. |
| $\\arg\\min_k$                           | Index of minimum value over candidate index $k$.                                           | Input: candidate scores/distances; Output: integer index per row.                  | Sections 5.5 and 6.2.             | Produces center assignments that drive discrete or categorical structure.              |
| $\\sum,\\prod,\\max$                     | Sum/product/max reductions over specified axis.                                            | Input: tensors with reduction axis; Output: reduced tensor.                        | Sections 4.2, 4.3, 5.2, 5.4, 5.7. | Controls interaction form (additive, multiplicative, extremal) in latent construction. |
| $\\operatorname{Converter}\_s(V_s)$      | Converter function for spec $s$, returning transformed slice and extracted value.          | Input: latent view $V_s$; Output: $(X'\_s, v_s)$.                                  | Section 4.3 and Section 6.        | Directly emits feature/target values that become final `X` and `y`.                    |
| $\\operatorname{Linear}(\\cdot)$         | Random linear map to the configured output width.                                          | Input: matrix $(n \\times d\_{in})$; Output: matrix $(n \\times d\_{out})$.        | Sections 5.5 and 5.7.             | Projects intermediate representations into node latent channels used by converters.    |
| $\\operatorname{leaf}(X)$                | Oblivious-tree leaf index determined by sampled splits/thresholds.                         | Input: row vector(s) and tree splits; Output: leaf id(s).                          | Section 5.4.                      | Selects leaf values that define piecewise latent behavior.                             |
| $\\operatorname{logit}\_{ik}$            | Pre-softmax score for row $i$, component/class $k$.                                        | Input: row-to-center distances and scale terms; Output: score matrix.              | Section 5.7.                      | Determines assignment probabilities that shape EM-style latent outputs.                |
| $\\operatorname{softmax}\_k(\\cdot)$     | Normalize exponentiated logits over index $k$ to probabilities.                            | Input: logits per row; Output: probabilities summing to 1 across $k$.              | Sections 5.7 and 6.2.             | Converts scores into stochastic assignments that drive categorical/mixture behavior.   |
| $\\operatorname{sample_noise}(\\cdot)$   | Noise sampler with configurable family and scale.                                          | Input: shape + family params; Output: random tensor with requested shape.          | Section 7.1.                      | Injects stochastic variation into points, matrices, weights, and node outputs.         |
| $\\operatorname{clip}(x,a,b)$            | Clamp each entry of $x$ into $[a,b]$.                                                      | Input: tensor; Output: tensor with same shape.                                     | Section 4.3.                      | Prevents extreme values from destabilizing downstream standardization/conversion.      |
| $\\operatorname{nan_to_num}(x)$          | Replace NaN/Inf entries with finite values.                                                | Input: tensor; Output: finite tensor with same shape.                              | Section 4.3.                      | Stabilizes latent values before weighting and conversion.                              |
| $\\operatorname{standardize}(x)$         | Column-wise centering and scaling to unit variance (with epsilon guard).                   | Input: matrix $(n \\times d)$; Output: matrix $(n \\times d)$.                     | Sections 4.3 and 6.2.             | Controls scale so mechanism outputs/converters are numerically comparable.             |
| $\\operatorname{Cauchy}(0,1)$            | Standard Cauchy distribution.                                                              | Input: location/scale; Output: scalar or tensor draws.                             | Section 1.                        | Provides heavy-tailed latent logits for heterogeneous graph connectivity.              |
| $\\operatorname{Bernoulli}(p)$           | Bernoulli draw with success probability $p$.                                               | Input: probability $p \\in [0,1]$; Output: binary draw.                            | Section 1.                        | Converts edge probabilities into realized adjacency $G$.                               |
| $\\operatorname{Uniform}(a,b)$           | Continuous uniform distribution on $[a,b]$.                                                | Input: bounds $(a,b)$; Output: scalar/tensor draws.                                | Sections 4.1 and 7.1.             | Drives root sampling and inverse-CDF noise construction.                               |
| $\\operatorname{UniformInt}{m,\\dots,n}$ | Discrete uniform integer sampler on inclusive range.                                       | Input: integer bounds; Output: integer draws.                                      | Section 5.4.                      | Samples tree depths and related discrete structure choices.                            |
| $\\operatorname{LogUniform}(a,b)$        | Distribution with $\\log x$ uniform on $[\\log a,\\log b]$.                                | Input: positive bounds $(a,b)$; Output: positive scalar draws.                     | Sections 4.3, 5.5, 6.1, 6.2.      | Controls scales/exponents over multiple orders of magnitude.                           |
| $\\operatorname{Categorical}(p)$         | Discrete draw from class probabilities $p$.                                                | Input: probability vector per row; Output: class/label index.                      | Section 6.2.                      | Emits categorical feature/target labels that populate final outputs.                   |

Conventions:

- Unless noted otherwise, operations are row-wise over batch dimension $n$ and preserve row count.
- In $\\operatorname{Agg}$, the aggregation kind is sampled once per multi-parent composition call and then applied elementwise across parents.
- In noise sampling, `mixture` in Section 7.1 samples component assignment per element, while Section 7.2 resolves one family per dataset attempt before downstream draws.

### End-to-End Primary Variable Map

**Primary Variable:** $\\mathcal{O} = (X, y, \\text{metadata})$.<br>
**Dependency Map:** $s\_{\\text{shift}}, A, B_i, C_j, \\lambda_f, \\tilde{\\ell}_f, \\text{noise_spec}, \\epsilon, G, \\pi_f, Z_{\\text{base}}, Z\_{\\text{comp}}, Y, Z\_{\\text{post}}, (X'\_s, v_s) \\rightarrow \\mathcal{O}$.<br>
**Path to Final Output:** shift runtime values define graph, mechanism, and noise controls; these drive node latents and converter emissions that become $(X,y)$, while runtime selections and graph/mechanism/noise summaries become metadata in $\\mathcal{O}$.<br>

**Rationale.** This section is the single whole-picture view that ties section-level primary variables into one output contract. It complements the per-section formal details by making global dataflow explicit.

Pipeline skeleton:

$$s\_{\\text{shift}} \\rightarrow (\\beta\_{\\text{edge}}, \\tau, \\gamma\_{\\sigma}, \\gamma\_{\\text{var}})$$
$$\\left(\\beta\_{\\text{edge}}, A, B_i, C_j\\right) \\rightarrow G$$
$$\\left(\\lambda_f, \\tau, \\tilde{\\ell}_f\\right) \\rightarrow \\pi_f$$
$$\\text{noise_spec} \\rightarrow \\epsilon_{\\text{family}} \\rightarrow \\epsilon$$
$$\\left(G, \\pi_f, \\epsilon, Z\_{\\text{base}}, Z\_{\\text{comp}}, Y, Z\_{\\text{post}}\\right) \\rightarrow {Z_j}\_{j=0}^{N-1}$$
$$\\left({Z_j}, \\operatorname{Converter}\_s\\right) \\rightarrow {(X'\_s, v_s)}_s \\rightarrow (X, y)$$
$$\\left(G, \\pi_f, \\text{noise_spec}, s_{\\text{shift}}\\right) \\rightarrow \\text{metadata}$$
$$\\mathcal{O} = (X, y, \\text{metadata})$$

Operator role notes:

- $\\operatorname{Agg}$ is the bridge from multiple parent tensors to $Z\_{\\text{comp}}$ (Section 4.2).
- $\\operatorname{Converter}\_s$ is the bridge from latent state to emitted observable values (Section 4.3 and Section 6).

Primary variable crosswalk:

| Section                             | Primary Variable                      | Immediate Inputs                                                          | Immediate Output                                                              | Contribution to $\\mathcal{O}$                                                                       |
| ----------------------------------- | ------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1. DAG Structure Sampling           | $G$                                   | $A, B_i, C_j, \\beta\_{\\text{edge}}$                                     | Parent sets $\\operatorname{pa}(j)$                                           | Controls execution order and parent flow that shapes $(X,y)$ and graph metadata.                     |
| 2. Shift Runtime Parameters         | $s\_{\\text{shift}}$                  | Resolved shift config                                                     | $(\\beta\_{\\text{edge}}, \\tau, \\gamma\_{\\sigma}, \\gamma\_{\\text{var}})$ | Couples drift controls into graph density, mechanism usage, and noise magnitude in outputs/metadata. |
| 3. Mechanism Family Sampling        | $\\pi_f$                              | $\\lambda_f, \\tau, \\tilde{\\ell}\_f, \\mathcal{F}$                      | Family draw distribution per mechanism call                                   | Determines which transform family generates latent structure before conversion to $(X,y)$.           |
| 4. Node Generation Pipeline         | $Z_j$                                 | Parent/root inputs, width terms, sampled mechanisms, weights, scale       | Node-level latent tensor                                                      | Shared latent state from which converter slices emit observed features/targets.                      |
| 4.1 Root-node base sampling         | $Z\_{\\text{base}}$                   | Base-kind selection, distribution params, $\\mathcal{D}\_{\\text{noise}}$ | Initial latent input for root mechanisms                                      | Seeds root latent variability that propagates to downstream nodes and final observables.             |
| 4.2 Parent composition              | $Z\_{\\text{comp}}$                   | Parent tensors, concat/aggregation path, $\\operatorname{Agg}$            | Composed latent input for node transform                                      | Defines multi-parent interaction pattern that feeds mechanism outputs used in $(X,y)$.               |
| 4.3 Post-processing and slicing     | $Z\_{\\text{post}}$                   | Raw latent, sanitize/standardize, weighting, padding, slice writeback     | Converter-ready latent state                                                  | Directly feeds converter extraction and latent feedback that determines emitted values.              |
| 5. Mechanism Definitions            | $Y = f(X)$                            | Mechanism parameters and input $X$                                        | Mechanism output tensor                                                       | Family-level nonlinear map that populates latent channels later converted to outputs.                |
| 5.1 Linear                          | $Y\_{\\text{linear}}$                 | $X, M$                                                                    | Linear latent channels                                                        | Baseline low-complexity latent effects converted into numeric/categorical outputs.                   |
| 5.2 Quadratic                       | $Y\_{\\text{quad}}$                   | $X\_{\\text{sub}}, X\_{\\text{aug}}, M_t$                                 | Interaction-heavy latent channels                                             | Adds pairwise interaction structure reflected in emitted features/targets.                           |
| 5.3 Neural network                  | $Y\_{\\text{nn}}$                     | $X, {M\_{\\ell}}, {\\phi\_{\\ell}}$                                       | Deep nonlinear latent channels                                                | Produces high-capacity nonlinear signals passed through converters into $(X,y)$.                     |
| 5.4 Tree ensemble                   | $Y\_{\\text{tree}}$                   | Splits, thresholds, leaf values                                           | Piecewise latent channels                                                     | Introduces discontinuous region-based structure visible in output distributions.                     |
| 5.5 Discretization                  | $Y\_{\\text{disc}}$                   | Centers, distance exponent, nearest-center assignment                     | Clustered latent channels                                                     | Creates quantized/cluster-like effects in converted observables.                                     |
| 5.6 GP/RFF                          | $Y\_{\\text{gp}}$                     | $X\_{\\text{proj}}, \\Omega, b, V$                                        | Smooth kernel-like latent channels                                            | Adds smooth nonlinear components that converters emit as continuous structure.                       |
| 5.7 EM assignment                   | $Y\_{\\text{em}}$                     | Centers, scales, exponents, assignment probabilities                      | Mixture-like latent channels                                                  | Encodes local mixture behavior that propagates to final emitted values.                              |
| 5.8 Product family                  | $Y\_{\\text{prod}}$                   | Two component family outputs $f(X), g(X)$                                 | Multiplicative latent channels                                                | Adds higher-order multiplicative effects before converter extraction.                                |
| 6. Converter Definitions            | $(X'\_s, v_s)$                        | Latent slice, converter kind/params                                       | Updated latent slice plus extracted value                                     | Defines the emission boundary from latent state to observed feature/target values.                   |
| 6.1 Numeric converter               | $(X'_{\\text{num}}, v_{\\text{num}})$ | Numeric slice, warp params, branch choice                                 | Numeric extracted value and feedback slice                                    | Produces numeric entries in $(X,y)$ and latent feedback for downstream coupling.                     |
| 6.2 Categorical converter           | $(X'\_{\\text{cat}}, \\ell)$          | Method/variant, cardinality, logits/centers                               | Categorical label and feedback slice                                          | Produces categorical feature/class outputs and associated latent feedback.                           |
| 7. Noise Sampling and Selection     | $\\epsilon$                           | Family, scale, family-specific params                                     | Runtime noise draws                                                           | Injects stochastic variation throughout latent generation and converter inputs.                      |
| 7.1 Primitive family samplers       | $\\epsilon\_{\\text{family}}$         | Primitive family choice and parameters                                    | Family-specific noise samples                                                 | Supplies concrete perturbation draws used by samplers/mechanisms.                                    |
| 7.2 Dataset-level family resolution | `noise_spec`                          | Requested family, mixture weights, seeded RNG                             | One attempt-level family spec                                                 | Determines dataset-level noise identity reflected in outputs and metadata.                           |

## 1. DAG Structure Sampling

**Primary Variable:** $G$ (DAG adjacency matrix).<br>
**Dependency Map:** $A, B_i, C_j, \\beta\_{\\text{edge}} \\rightarrow p\_{ij} \\rightarrow G\_{ij}$.<br>
**Path to Final Output:** $G \\rightarrow \\operatorname{pa}(j)$ for each node -> node execution order and parent inputs -> latent tensors -> emitted `X`, `y`, and graph metadata.<br>

**Rationale.** The latent-logit Cauchy construction yields heterogeneous edge probabilities while strict upper-triangular masking guarantees acyclicity. Writing $p\_{ij}$ and $\\beta\_{\\text{edge}}$ explicitly keeps graph-drift behavior inspectable and testable.

`sample_dag` draws a strict upper-triangular adjacency matrix.

$$A \\sim \\operatorname{Cauchy}(0,1), \\quad B_i \\sim \\operatorname{Cauchy}(0,1), \\quad C_j \\sim \\operatorname{Cauchy}(0,1)$$

Interpretation:

- $A$: global intercept-like term shifting all edge logits together.
- $B_i$: source-node effect for node $i$ (how strongly node $i$ tends to emit edges).
- $C_j$: destination-node effect for node $j$ (how strongly node $j$ tends to receive edges).

For each $i < j$:

$$p\_{ij} = \\sigma(A + B_i + C_j + \\beta\_{\\text{edge}}), \\quad G\_{ij} \\sim \\operatorname{Bernoulli}(p\_{ij})$$

and for $i \\ge j$, $G\_{ij}=0$.

## 2. Shift Runtime Parameters

**Primary Variable:** $s\_{\\text{shift}} = (\\text{graph_scale}, \\text{mechanism_scale}, \\text{variance_scale})$.<br>
**Dependency Map:** $s\_{\\text{shift}} \\rightarrow (\\beta\_{\\text{edge}}, \\tau, \\gamma\_{\\sigma}, \\gamma\_{\\text{var}})$ via deterministic runtime mappings.<br>
**Path to Final Output:** $\\beta\_{\\text{edge}} \\rightarrow G$, $\\tau \\rightarrow \\pi_f$, and $\\gamma\_{\\sigma} \\rightarrow$ noise amplitude; these three paths converge in node generation and sampling to shape final `X`, `y`, and shift metadata.<br>

**Rationale.** Shift controls are defined as deterministic mappings from config scales to runtime coefficients so drift remains interpretable. Separating graph, mechanism, and variance mappings keeps each drift axis analyzable in isolation.

With resolved shift scales `(graph_scale, mechanism_scale, variance_scale)`:

$$\\beta\_{\\text{edge}} = \\ln(2) \\cdot \\text{graph_scale}$$
$$\\tau = \\text{mechanism_scale}$$
$$\\gamma\_\\sigma = \\exp\\left(\\frac{\\ln(2)}{2} \\cdot \\text{variance_scale}\\right), \\quad \\gamma\_{\\text{var}} = \\gamma\_\\sigma^2 = 2^{\\text{variance_scale}}$$

So positive `variance_scale` increases global noise variance multiplicatively.

## 3. Mechanism Family Sampling with Mix + Tilt

**Primary Variable:** $\\pi_f$ (mechanism-family sampling distribution).<br>
**Dependency Map:** $\\lambda_f, \\tau, \\tilde{\\ell}\_f, \\mathcal{F} \\rightarrow \\pi_f$.<br>
**Path to Final Output:** $\\pi_f \\rightarrow$ sampled mechanism family at each function draw -> latent transform type and nonlinearity -> converted observable values in `X`/`y` and nonlinear-mass-related metadata.<br>

**Rationale.** This parameterization separates family support control (`function_family_mix`) from directional drift (`mechanism_logit_tilt`). Centered logits make tilt a relative reweighting mechanism rather than an arbitrary global scaling change.

Let raw family weights be:

$$\\lambda_f = \\begin{cases} 1, & \\text{if no function_family_mix} \\ \\text{configured positive weight}, & \\text{otherwise} \\end{cases}$$

Normalize over positive-weight families:

$$\\pi_f^{(0)} = \\frac{\\lambda_f}{\\sum\_{g \\in \\mathcal{F}} \\lambda_g}$$

$\\pi_f^{(0)}$ is the baseline family distribution after applying `function_family_mix` support/weights and before any tilt.

If $\\tau \\le 0$, then $\\pi_f = \\pi_f^{(0)}$.
If $\\tau > 0$, logits are tilted using centered family base logits:

$$\\pi_f \\propto \\exp\\left(\\ln \\pi_f^{(0)} + \\tau \\tilde{\\ell}\_f\\right)$$

then normalized to sum to 1.

Configuration intuition:

- `mechanism.function_family_mix` controls support and baseline weights through $\\lambda_f$.
- `shift.mechanism_scale` sets $\\tau$ (Section 2), which reweights within that support.
- Increasing $\\tau$ shifts mass toward families with higher $\\tilde{\\ell}\_f$ and away from families with lower $\\tilde{\\ell}\_f$, changing mechanism usage frequencies that shape final `X`/`y`.
- Example: with mix `{nn: 0.7, linear: 0.3}`, $\\tau=0$ gives $\\pi^{(0)}_{nn}=0.7,\\pi^{(0)}_{linear}=0.3$; with $\\tau>0$, $nn$ gains additional mass because its centered base logit is higher than linear's.

Base logits used in $\\tilde{\\ell}\_f$ (**ONLY IF $\\tau > 0$**):

| Family $f$       | $\\ell_f$ |
| :--------------- | :-------- |
| `nn`             | 0.7       |
| `tree`           | 0.7       |
| `discretization` | 0.5       |
| `gp`             | 0.5       |
| `linear`         | -0.8      |
| `quadratic`      | -0.6      |
| `em`             | -0.3      |
| `product`        | 0.9       |

#### Rationale for Non-uniform Base Logits

The base logits define the **direction of mechanism drift**, not the default family distribution.

- When `mechanism_logit_tilt <= 0`, family probabilities are just normalized family weights (uniform when no `function_family_mix` is provided), so base logits have no effect.
- When `mechanism_logit_tilt > 0`, centered base logits reweight probabilities toward families with larger centered logits and away from families with smaller centered logits.
- In the current defaults, this shifts mass toward nonlinear families (`nn`, `tree`, `discretization`, `gp`, `product`) and away from simpler families (`linear`, `quadratic`), increasing nonlinear mechanism mass under positive tilt.
- `function_family_mix` still controls support and baseline weights; base logits only reweight **within enabled families**.

Implementation note: `product` is only valid when at least one of its component families is enabled (`tree`, `discretization`, `gp`, `linear`, `quadratic`).

## 4. Node Generation Pipeline

**Primary Variable:** $Z_j$ (node-$j$ latent tensor after pipeline transforms).<br>
**Dependency Map:** root/parent inputs, mechanism draws, width terms $(d\_{\\text{req}}, d\_{\\text{extra}})$, random weights $w$, and scale $s_j \\rightarrow Z_j$.<br>
**Path to Final Output:** $Z_j \\rightarrow$ converter slices and extracted values -> feature/target assignments -> final `X` and `y`.<br>

**Rationale.** Node generation is documented as an ordered contract because converter extraction and metadata semantics depend on stable execution order. The split into root sampling, parent composition, and latent refinement mirrors the actual runtime path.

### 4.1 Root-node base sampling

**Primary Variable:** $Z\_{\\text{base}}$.<br>
**Dependency Map:** base-kind choice + distribution parameters + active noise family $\\mathcal{D}_{\\text{noise}} \\rightarrow Z_{\\text{base}}$.<br>
**Path to Final Output:** $Z\_{\\text{base}} \\rightarrow f(Z\_{\\text{base}})$ (initial node latent) -> downstream node pipeline -> converter extraction -> `X`/`y`.<br>

**Rationale.** Multiple root base geometries increase latent diversity before any parent effects exist. Passing every sampled base through $f(\\cdot)$ keeps root and non-root nodes aligned under the same mechanism-family controls.

For root nodes ($\\operatorname{pa}(j) = \\varnothing$), one base kind is chosen uniformly from:

1. `normal`: $Z\_{\\text{base}} \\sim \\mathcal{D}\_{\\text{noise}}^{n \\times d}$.
1. `uniform`: elementwise $\\operatorname{Uniform}(-1,1)$.
1. `unit_ball`: for each row,
   $$v \\sim \\mathcal{N}(0, I_d), \\quad u \\sim \\operatorname{Uniform}(0,1), \\quad z = \\frac{v}{|v|\_2} u^{1/d}$$
1. `normal_cov`:
   $$E \\sim \\mathcal{D}_{\\text{noise}}^{n \\times d}, \\quad A \\sim \\mathcal{D}_{\\text{noise}}^{d \\times d}, \\quad Z\_{\\text{base}} = (E \\odot w^\\top) A^\\top$$
   where $w$ is sampled by `sample_random_weights` (a positive, sum-to-one, randomly permuted weight vector following a noisy power-law decay).

Then a random mechanism is applied:

$$Z_j = f(Z\_{\\text{base}})$$

### 4.2 Parent composition (non-root nodes)

**Primary Variable:** $Z\_{\\text{comp}}$ (composed parent latent input to node transform).<br>
**Dependency Map:** parent tensors ${Z\_{p_r}}$ + composition path choice + aggregation operator $\\operatorname{Agg} \\rightarrow Z\_{\\text{comp}}$.<br>
**Path to Final Output:** $Z\_{\\text{comp}} \\rightarrow$ node mechanism output -> updated $Z_j$ -> converter extraction -> `X`/`y`.<br>

**Rationale.** The 50/50 concat-versus-aggregation branch injects both feature-mixing and commutative interaction patterns. This broadens functional regimes without changing the node interface contract.

If $|\\operatorname{pa}(j)|=1$, apply one random mechanism to that parent tensor.

If $|\\operatorname{pa}(j)| \\ge 2$, choose one path with equal probability:

1. **Concatenation path**:
   $$Z\_{\\text{in}} = [Z\_{p_1} | Z\_{p_2} | \\dots | Z\_{p_k}], \\quad Z_j = f(Z\_{\\text{in}})$$
1. **Aggregation path**:
   $$Y_r = f_r(Z\_{p_r}), \\quad Z_j = \\operatorname{Agg}(Y_1, \\dots, Y_k)$$
   with $\\operatorname{Agg}$ sampled uniformly from:
   $$\\sum_r Y_r, \\quad \\prod_r Y_r, \\quad \\max_r Y_r, \\quad \\operatorname{LSE}(Y_1, \\dots, Y_k)$$

Concatenation semantics (implementation-faithful):

- If parent tensors are $Z\_{p_r} \\in \\mathbb{R}^{n \\times d_r}$, concatenation is along feature axis:
  $$Z\_{\\text{in}} \\in \\mathbb{R}^{n \\times \\sum\_{r=1}^k d_r}.$$
- The concatenated tensor is passed through one random mechanism $f(\\cdot)$ with node output width target $d$ (the node `total_dim` from Section 4.3), so:
  $$f: \\mathbb{R}^{n \\times \\sum_r d_r} \\rightarrow \\mathbb{R}^{n \\times d}.$$
- In this branch, aggregation kind is not used; `Agg` only applies in the non-concatenation branch.

Example:

- Three parents with shapes $(n,4)$, $(n,6)$, $(n,3)$ concatenate to $(n,13)$.
- If node target width is $d=9$, the mechanism maps $(n,13)\\rightarrow(n,9)$ before post-processing and converter slicing.

### 4.3 Latent post-processing and converter slicing

**Primary Variable:** $Z\_{\\text{post}}$ (post-processed latent tensor used for conversion).<br>
**Dependency Map:** raw node latent + sanitize/standardize operations + random weights $w$ + normalization + padding noise + slice/writeback updates $\\rightarrow Z\_{\\text{post}}$.<br>
**Path to Final Output:** $Z\_{\\text{post}} \\rightarrow$ per-spec extracted values $(v_s)$ and rewritten latent slices -> assigned feature/target outputs -> final `X`/`y`.<br>

**Rationale.** Sanitization, standardization, weighting, and norm scaling are applied before conversion to keep numerics stable across mechanism families. The slice-and-writeback loop formalizes how observable values are extracted while preserving latent continuity.

For each node, converter specs define required width:

$$d\_{\\text{req}} = \\sum_s \\max(1, d_s), \\quad d = d\_{\\text{req}} + d\_{\\text{extra}}, \\quad d\_{\\text{extra}} = \\max(1, \\lfloor \\operatorname{LogUniform}(1, 32) \\rfloor)$$

After mechanism output:

$$Z_j \\leftarrow \\operatorname{clip}(\\operatorname{nan_to_num}(Z_j), -10^6, 10^6)$$
$$Z_j \\leftarrow \\operatorname{standardize}(Z_j)$$
$$Z_j \\leftarrow Z_j \\odot w^\\top$$
$$Z_j \\leftarrow \\frac{Z_j}{\\max\\left(\\frac{1}{n} \\sum\_{r=1}^n |Z\_{j,r:}|\_2, 10^{-6}\\right)}$$

Converter loop semantics:

1. Slice per-spec view $V_s$ from current cursor over columns.
1. Pad missing columns with sampled noise if needed.
1. Apply converter: $(X'\_s, v_s) = \\operatorname{Converter}\_s(V_s)$.
1. Write $X'\_s$ back into the same slice (shape-padded/truncated as needed).
1. Store $v_s$ as extracted observable value for feature/target key.

Final node scaling:

$$Z_j \\leftarrow s_j Z_j, \\quad s_j \\sim \\operatorname{LogUniform}(0.1, 10)$$

## 5. Mechanism Family Definitions

**Primary Variable:** $Y = f(X)$ (family-specific mechanism output).<br>
**Dependency Map:** mechanism-family parameters + input $X \\rightarrow Y$.<br>
**Path to Final Output:** $Y \\rightarrow$ node latent updates and feedback slices -> converter-extracted observables -> final `X`/`y`.<br>

**Rationale.** The mechanism catalog intentionally spans smooth, piecewise, assignment-based, and compositional transforms to vary functional complexity under drift and mix controls. Each family maps to a concrete sampler implementation in `random_functions.py`.

Inputs to mechanism families are first sanitized and standardized.

### 5.1 Linear

**Primary Variable:** $Y\_{\\text{linear}}$.<br>
**Dependency Map:** $X, M \\rightarrow Y\_{\\text{linear}} = X M^\\top$.<br>
**Path to Final Output:** $Y\_{\\text{linear}} \\rightarrow$ latent columns for the node -> downstream converter extraction -> numeric/categorical outputs in `X` or `y`.<br>

**Rationale.** Linear maps act as a low-complexity baseline and calibration point for drift against nonlinear families. Row-normalized random matrices preserve directional mixing while limiting runaway scale.

$$Y = X M^\\top$$

where $M$ is a sampled random matrix.

### 5.2 Quadratic

**Primary Variable:** $Y\_{\\text{quad}}$.<br>
**Dependency Map:** $X\_{\\text{sub}}, X\_{\\text{aug}}, M_t \\rightarrow Y\_{\\text{quad}}$.<br>
**Path to Final Output:** $Y\_{\\text{quad}} \\rightarrow$ interaction-heavy latent signal -> converter extraction -> nonlinear feature/target structure in `X`/`y`.<br>

**Rationale.** Quadratic forms add pairwise interactions not available in pure linear maps. Subspace capping and bias augmentation balance expressivity with numerical stability and compute control.

Let $X\_{\\text{sub}}$ be either all columns or a random 20-column subset if input width exceeds 20.
Augment with bias column: $X\_{\\text{aug}} = [X\_{\\text{sub}}, \\mathbf{1}]$.
For each output channel $t$:

$$Y_t = \\operatorname{diag}(X\_{\\text{aug}}, M_t, X\_{\\text{aug}}^\\top)$$

or equivalently per row: $y_i = x\_{\\text{aug},i}^\\top M_t, x\_{\\text{aug},i}$ — a general (not necessarily symmetric) quadratic form. The bias augmentation means this captures quadratic + linear + constant terms.

### 5.3 Neural network

**Primary Variable:** $Y\_{\\text{nn}}$.<br>
**Dependency Map:** $X$, sampled layer matrices ${M\_\\ell}$, and sampled activations ${\\phi\_\\ell}$ $\\rightarrow Y\_{\\text{nn}}$.<br>
**Path to Final Output:** $Y\_{\\text{nn}} \\rightarrow$ high-capacity latent states -> converter extraction -> complex nonlinear observables in `X`/`y`.<br>

**Rationale.** Random-depth MLPs provide high-capacity nonlinear transforms with stochastic activation structure. Optional pre/post activation steps increase shape diversity while retaining a simple sampled-layer pipeline.

A random depth/width MLP with number of weight matrices sampled uniformly from ${1, 2, 3}$ (0–2 hidden layers) and hidden width from $\\lfloor \\operatorname{LogUniform}(1, 127) \\rfloor$:

$$Y^{(0)} = X \\text{ (optional pre-activation with prob 0.5)}$$
$$Y^{(\\ell+1)} = \\phi\_\\ell(Y^{(\\ell)} M\_\\ell^\\top)$$

with activation after hidden layers, optional final activation (prob 0.5).

Activation sampler includes fixed and parametric families.

- Fixed activations: `tanh`, `leaky_relu`, `elu`, `identity`, `silu`, `swiglu`, `relu`, `relu_sq`, `softplus`, `sign`, `gauss`, `exp`, `sin`, `square`, `abs`, `softmax`, `logsigmoid`, `logabs`, `sigmoid`, `round`, `mod1`, `selu`, `relu6`, `hardtanh`, `heaviside`, `indicator_01`, `onehot_argmax`, `argsort`, `rank`.
- Parametric families: `relu_pow`, `signed_pow`, `inv_pow`, `poly`.

### 5.4 Tree ensemble (oblivious decision trees)

**Primary Variable:** $Y\_{\\text{tree}}$.<br>
**Dependency Map:** split features, thresholds, leaf indices, and leaf values $\\rightarrow Y\_{\\text{tree}}$.<br>
**Path to Final Output:** $Y\_{\\text{tree}} \\rightarrow$ piecewise latent regions -> converter extraction -> discontinuous/tabular decision patterns in `X`/`y`.<br>

**Rationale.** Oblivious trees introduce thresholded piecewise behavior that complements smooth families. Variance-weighted split sampling biases random partitions toward informative dimensions without requiring supervised fitting.

Number of trees: $T = \\max(1, \\lfloor \\operatorname{LogUniform}(1, 32) \\rfloor)$.

For each tree $t$:

1. Depth $d_t \\sim \\operatorname{UniformInt}{1, \\dots, 7}$.
1. Sample split feature indices (variance-weighted probabilities when available).
1. Sample thresholds by indexing sampled training rows.
1. Compute leaf index via packed split bits.
1. Sample leaf values $V_t \\in \\mathbb{R}^{2^{d_t} \\times d\_{\\text{out}}}$ from noise sampler.

Prediction is averaged:

$$Y = \\frac{1}{T} \\sum\_{t=1}^T V_t[\\operatorname{leaf}(X)]$$

### 5.5 Discretization

**Primary Variable:** $Y\_{\\text{disc}}$.<br>
**Dependency Map:** sampled centers + distance exponent $p$ + nearest-center index + linear map $\\rightarrow Y\_{\\text{disc}}$.<br>
**Path to Final Output:** $Y\_{\\text{disc}} \\rightarrow$ clustered latent representation -> converter extraction -> quantized/cluster-like observables in `X`/`y`.<br>

**Rationale.** Nearest-center assignment creates clustered discontinuities and quantization effects that smooth mechanisms do not capture. A final linear map preserves shared output-shape contracts across families.

Number of centers: $K = \\operatorname{clamp}(\\lfloor \\operatorname{LogUniform}(2, 128) \\rfloor,, 2,, n)$.

Sample centers from rows, compute $L_p$-style distances (with sampled $p \\sim \\operatorname{LogUniform}(0.5, 4)$), assign nearest center, then apply linear map:

$$k^*(i) = \\arg\\min_k \\sum_q |X\_{iq} - c\_{kq}|^p, \\quad Y = \\operatorname{Linear}(c\_{k^*(i)})$$

### 5.6 GP/RFF approximation

**Primary Variable:** $Y\_{\\text{gp}}$.<br>
**Dependency Map:** $X\_{\\text{proj}}, \\Omega, b, V \\rightarrow \\Phi \\rightarrow Y\_{\\text{gp}}$.<br>
**Path to Final Output:** $Y\_{\\text{gp}} \\rightarrow$ smooth kernel-like latent behavior -> converter extraction -> continuous nonlinear structure in `X`/`y`.<br>

**Rationale.** Random Fourier features approximate kernel-style smooth nonlinearities at fixed computational cost. Dual frequency-sampling branches increase spectral variety while keeping the same output interface.

Uses $P=256$ random features and two branches for sampling frequency matrix $\\Omega$.
Core output:

$$\\Phi = \\cos(X\_{\\text{proj}} \\Omega^\\top + b), \\quad Y = \\frac{1}{\\sqrt{P}} \\Phi V^\\top$$

where $b$ is random phase and $V$ is sampled from noise.

### 5.7 EM-style assignment

**Primary Variable:** $Y\_{\\text{em}}$.<br>
**Dependency Map:** centers $c_k$, scales $\\sigma_k$, exponents $(p,q)$, and assignment probabilities $P\_{ik} \\rightarrow Y\_{\\text{em}}$.<br>
**Path to Final Output:** $Y\_{\\text{em}} \\rightarrow$ mixture-like local latent effects -> converter extraction -> locally structured observables in `X`/`y`.<br>

**Rationale.** Soft center assignments add mixture-like local behavior and scale-sensitive responses. The linear readout over assignment probabilities keeps compatibility with common downstream handling.

Sample centers $c_k$, scales $\\sigma_k$, and exponents $p, q$:

- Centers: $K = \\max(2, \\lfloor \\operatorname{LogUniform}(2, \\max(16,, 2 d\_{\\text{out}})) \\rfloor)$
- Scales: $\\sigma_k = \\exp(\\text{noise} \\cdot 0.1)$ (log-normal, concentrated near 1)
- Exponents: $p \\sim \\operatorname{LogUniform}(1, 4)$, $; q \\sim \\operatorname{LogUniform}(1, 2)$

$$d\_{ik} = \\left(\\sum_q |X\_{iq} - c\_{kq}|^p\\right)^{1/p}$$
$$\\operatorname{logit}_{ik} = -\\frac{1}{2} \\log(2\\pi\\sigma_k^2) - \\left(\\frac{d_{ik}}{\\max(\\sigma_k, 10^{-6})}\\right)^q$$
$$P\_{ik} = \\operatorname{softmax}_k(\\operatorname{logit}_{ik}), \\quad Y = \\operatorname{Linear}(P)$$

### 5.8 Product family

**Primary Variable:** $Y\_{\\text{prod}}$.<br>
**Dependency Map:** component family draws $f(X)$ and $g(X)$ from allowed support $\\rightarrow Y\_{\\text{prod}} = f(X)\\odot g(X)$.<br>
**Path to Final Output:** $Y\_{\\text{prod}} \\rightarrow$ multiplicative latent interactions -> converter extraction -> higher-order effects in final `X`/`y`.<br>

**Rationale.** Multiplying two component mechanisms introduces multiplicative interactions and sharper nonlinear responses than additive composition alone. Restricting component candidates to vetted families maintains stability under family-mix masking.

Pick two component families $f, g$ from allowed set
`(tree, discretization, gp, linear, quadratic)` (filtered by enabled family mix), then:

$$Y = f(X) \\odot g(X)$$

## 6. Converter Definitions

**Primary Variable:** $(X'\_s, v_s)$ (converter output state and extracted observable value for spec $s$).<br>
**Dependency Map:** latent slice, converter kind, and converter parameters $\\rightarrow (X'\_s, v_s)$.<br>
**Path to Final Output:** extracted $v_s$ values are assigned to feature/target slots and become emitted `X`/`y`; $X'\_s$ is written back to latent for downstream coupling.<br>

**Rationale.** Converters are separated from mechanism families because observable typing is an output-contract concern, not a latent-mechanism concern. This decoupling lets one latent pipeline serve multiple feature schemas.

### 6.1 Numeric converter

**Primary Variable:** $(X'_{\\text{num}}, v_{\\text{num}})$.<br>
**Dependency Map:** input slice $X$, warp parameters $(a,b)$, and identity/warp branch choice $\\rightarrow (X'_{\\text{num}}, v_{\\text{num}})$.<br>
**Path to Final Output:** $v\_{\\text{num}}$ is emitted as numeric feature/target value in `X`/`y`; $X'\_{\\text{num}}$ feeds back into node latent state.<br>

**Rationale.** The identity branch preserves raw latent structure when extra warping is unnecessary. The optional bounded Kumaraswamy-like warp provides controllable marginal shaping while staying numerically stable.

Given input slice $X$ and raw extracted scalar $v = X\_{:,0}$:

- with probability 0.5: identity output (no warp),
- otherwise sample $a,b \\sim \\operatorname{LogUniform}(0.2,5)$ and apply columnwise min-max + warp:

$$\\hat{X} = \\frac{X - X\_{\\min}}{\\max(X\_{\\max} - X\_{\\min}, 10^{-6})}$$
$$X' = 1 - \\left(1 - \\operatorname{clip}(\\hat{X}, 0, 1)^a\\right)^b$$

Converter returns $(X', v)$.

### 6.2 Categorical converter

**Primary Variable:** $(X'_{\\text{cat}}, \\ell)$ where $\\ell$ is categorical label output.<br>
**Dependency Map:** method/variant choice, cardinality $C$, centers or logits/probabilities $\\rightarrow (X'_{\\text{cat}}, \\ell)$.<br>
**Path to Final Output:** $\\ell$ is emitted as categorical feature/class target in `X`/`y`; $X'\_{\\text{cat}}$ is fed back into latent columns.<br>

**Rationale.** Joint method-variant sampling separates class assignment logic from latent representation feedback into node state. This increases categorical behavior diversity while preserving a consistent `(out, labels)` converter contract.

A joint `(method, variant)` pair is sampled from:

- `neighbor`: `input`, `index_repeat`, `center`, `center_random_fn`
- `softmax`: `input`, `index_repeat`, `softmax_points`

Let target cardinality be $C = \\max(2, n\_{\\text{categories}})$.

Neighbor method:

$$\\text{label}_i = \\arg\\min_k \\sum_q |X_{iq} - c\_{kq}|^p, \\quad p \\sim \\operatorname{LogUniform}(0.5, 4)$$

Softmax method:

$$L = a \\operatorname{standardize}(X\_{\\text{proj}}) + b, \\quad a \\sim \\operatorname{LogUniform}(0.1, 10), \\quad b_k = \\ln(u_k + 10^{-4})$$
$$\\text{label}\_i \\sim \\operatorname{Categorical}(\\operatorname{softmax}(L_i))$$

Output representation $X'$ is variant-dependent (input copy, repeated index, center lookup, center transformed by random function, or random softmax points). Labels are finally reduced modulo $C$.

## 7. Noise Sampling and Runtime Selection

**Primary Variable:** $\\epsilon$ (active noise draw process for the attempt).<br>
**Dependency Map:** noise family, scale, and family-specific parameters $\\rightarrow \\epsilon$.<br>
**Path to Final Output:** $\\epsilon$ perturbs points, matrices, weights, and explicit noise draws -> latent variability and converter inputs -> final `X`/`y` dispersion and noise metadata.<br>

**Rationale.** Noise is modeled as its own axis so distributional perturbation can be controlled independently from mechanism complexity. Distinguishing primitive samplers from runtime selection clarifies where stochasticity enters.

### 7.1 Primitive family samplers

**Primary Variable:** $\\epsilon\_{\\text{family}}$ (draws from one resolved primitive family).<br>
**Dependency Map:** selected primitive family + shape + scale (+ df or mixture weights when applicable) $\\rightarrow \\epsilon\_{\\text{family}}$.<br>
**Path to Final Output:** $\\epsilon\_{\\text{family}}$ is used by samplers/mechanisms across generation -> affects latent tensors and emitted `X`/`y`.<br>

**Rationale.** Multiple primitive families cover light-tail, double-exponential, and heavy-tail regimes under one API surface. Explicit formulas keep distributional assumptions auditable across backends.

For $\\operatorname{sample_noise}(\\text{shape, family, scale, \\dots})$:

- **Gaussian**: i.i.d. $\\mathcal{N}(0, 1)$
- **Laplace** (inverse CDF):
  $$U \\sim \\operatorname{Uniform}(\\epsilon, 1-\\epsilon), \\quad X = \\begin{cases} \\ln(2U), & U < 0.5 \\ -\\ln(2(1-U)), & U \\ge 0.5 \\end{cases}$$
- **Student-t**:
  $$X = \\frac{Z}{\\sqrt{\\chi^2\_\\nu / \\nu}}, \\quad Z \\sim \\mathcal{N}(0, 1), \\ \\nu > 2$$
- **Mixture**: sample component assignment per element from normalized weights over
  `{gaussian, laplace, student_t}` and draw from that component.

All outputs are scaled by `scale`.

### 7.2 Dataset-level noise-family resolution

**Primary Variable:** `noise_spec` (resolved attempt-level noise sampling specification).<br>
**Dependency Map:** requested family, mixture weights, and deterministic seed stream $\\rightarrow$ runtime selection $\\rightarrow$ `noise_spec`.<br>
**Path to Final Output:** `noise_spec` parameterizes all downstream noise draws in the attempt -> dataset-level distributional character in final `X`/`y` plus noise-distribution metadata.<br>

**Rationale.** In mixture mode, sampling one family per dataset attempt preserves coherent dataset-level identity instead of per-element family switching. Seeded selection guarantees reproducibility and metadata traceability.

Generation runtime resolves one `NoiseRuntimeSelection` per dataset attempt:

- If requested family is not `mixture`: sampled family equals requested family.
- If requested family is `mixture`:
  1. Normalize configured mixture weights.
  1. Use deterministic RNG stream `SeedManager(run_seed).torch_rng("noise_family")`.
  1. Sample exactly one component family for the dataset.

The resulting `noise_spec` uses that sampled family and base scale for all downstream draws in the attempt. Shift noise drift applies through multiplicative scale factor $\\gamma\_\\sigma$.
