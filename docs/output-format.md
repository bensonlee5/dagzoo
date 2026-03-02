# Output Format

Consumer-facing specification for generated data. This is a **contract
document** — downstream users can rely on the guarantees described here.

______________________________________________________________________

## DatasetBundle (in-memory)

Each generated dataset is returned as a `DatasetBundle` with these fields:

| Field           | Type                                | Shape                 |
| --------------- | ----------------------------------- | --------------------- |
| `X_train`       | `torch.Tensor` (float32 or float64) | (n_train, n_features) |
| `y_train`       | `torch.Tensor`                      | (n_train,)            |
| `X_test`        | `torch.Tensor` (float32 or float64) | (n_test, n_features)  |
| `y_test`        | `torch.Tensor`                      | (n_test,)             |
| `feature_types` | `list[str]`                         | length n_features     |
| `metadata`      | `dict[str, Any]`                    | —                     |

**Target dtype**: `int64` for classification, float for regression.

**Feature dtype**: matches the configured torch dtype (float32 or float64).

______________________________________________________________________

## Feature type encoding

Each entry in `feature_types` is one of:

- `"num"` — continuous feature. After postprocessing, values are clipped and
  standardized to approximately zero mean and unit variance.
- `"cat"` — categorical feature. Integer indices in the range
  `0 .. cardinality - 1`.

`feature_types[i]` describes column index `i` in X_train and X_test.

______________________________________________________________________

## On-disk directory structure

```
out_dir/
  shard_00000/
    train.parquet
    test.parquet
    metadata.ndjson
    lineage/
      adjacency.bitpack.bin
      adjacency.index.json
  shard_00001/
    ...
```

**Shard naming**: `shard_{id:05d}` — five-digit zero-padded shard ID.
Default: 128 datasets per shard.

**Shard ID calculation**: `dataset_index // shard_size`.

______________________________________________________________________

## Parquet column schema

Shard-level `train.parquet` and `test.parquet` both use packed row-wise
records:

| Column          | Type                  | Description                                |
| --------------- | --------------------- | ------------------------------------------ |
| `dataset_index` | int64                 | Global dataset index for this row          |
| `row_index`     | int64                 | Row index within the dataset split         |
| `x`             | list[float32/float64] | Full feature vector for this row           |
| `y`             | int64 or float        | Target value for this row (task-dependent) |

**Compression**: zstd (default).

Feature typing metadata remains per-dataset in `metadata.ndjson` records.

______________________________________________________________________

## Metadata NDJSON structure

Each shard writes one `metadata.ndjson` file with one JSON record per dataset.
Each line contains:

| Key             | Type      | Description                                  |
| --------------- | --------- | -------------------------------------------- |
| `dataset_index` | int       | Global dataset index                         |
| `n_train`       | int       | Train row count for the dataset              |
| `n_test`        | int       | Test row count for the dataset               |
| `n_features`    | int       | Feature count for the dataset                |
| `feature_types` | list[str] | Per-feature type annotations (`num`/`cat`)   |
| `metadata`      | object    | The dataset metadata payload described below |

`metadata` contains the same object fields as before.

### Top-level keys

| Key                      | Type        | Description                                                    |
| ------------------------ | ----------- | -------------------------------------------------------------- |
| `backend`                | str         | Always `"torch"`                                               |
| `device`                 | str         | Compute device (e.g., `"cpu"`, `"cuda"`)                       |
| `compute_backend`        | str         | Implementation variant identifier                              |
| `n_features`             | int         | Number of features                                             |
| `n_categorical_features` | int         | Number of categorical features                                 |
| `n_classes`              | int or null | Realized class count in emitted labels (null for regression)   |
| `graph_nodes`            | int         | Number of nodes in the DAG                                     |
| `graph_edges`            | int         | Number of edges in the DAG                                     |
| `graph_depth_nodes`      | int         | Longest path length in the DAG                                 |
| `graph_edge_density`     | float       | Edge count / max possible edges                                |
| `seed`                   | int         | Base seed for this dataset                                     |
| `attempt_used`           | int         | Generation attempt index (0-based)                             |
| `lineage`                | object      | DAG lineage record (see Lineage below)                         |
| `shift`                  | object      | Resolved shift settings and realized observability signals     |
| `noise`                  | object      | Resolved noise-family selection and effective sampling params  |
| `config`                 | object      | Full serialized generator configuration                        |
| `filter`                 | object      | Filter results (see below)                                     |
| `class_structure`        | object      | Present only for classification (see below)                    |
| `missingness`            | object      | Present only when missingness is enabled                       |
| `layout_mode`            | str         | Optional layout mode metadata (`"fixed"` for fixed-layout API) |
| `layout_plan_seed`       | int         | Optional fixed-layout plan seed                                |
| `layout_signature`       | str         | Optional deterministic fixed-layout fingerprint                |

### Shift sub-object

Present for all generated bundles. When shift is disabled, scales are `0.0` and
multipliers are `1.0`.

| Key                         | Type  | Description                                                 |
| --------------------------- | ----- | ----------------------------------------------------------- |
| `enabled`                   | bool  | Whether shift controls were enabled                         |
| `profile`                   | str   | Resolved shift profile (`off`, `graph_drift`, etc.)         |
| `graph_scale`               | float | Resolved graph drift scale                                  |
| `mechanism_scale`           | float | Resolved mechanism drift scale                              |
| `noise_scale`               | float | Resolved noise drift scale                                  |
| `edge_logit_bias_shift`     | float | Additive shift applied to edge logits                       |
| `mechanism_logit_tilt`      | float | Mechanism-family tilt applied at sampling                   |
| `noise_sigma_multiplier`    | float | Sigma multiplier applied to stochastic noise                |
| `edge_odds_multiplier`      | float | Edge-odds multiplier (`exp(edge_logit_bias_shift)`)         |
| `noise_variance_multiplier` | float | Noise-variance multiplier (`noise_sigma_multiplier^2`)      |
| `mechanism_nonlinear_mass`  | float | Probability mass on nonlinear mechanism families (`[0, 1]`) |

### Noise sub-object

Present for all generated bundles.

| Key                 | Type           | Description                                                          |
| ------------------- | -------------- | -------------------------------------------------------------------- |
| `family_requested`  | str            | Configured noise family (`legacy`, `gaussian`, `laplace`, etc.)      |
| `family_sampled`    | str            | Effective family used by the dataset generation runtime              |
| `sampling_strategy` | str            | Runtime selection strategy (`dataset_level`)                         |
| `scale`             | float          | Base noise scale from config                                         |
| `student_t_df`      | float          | Student-t degrees of freedom parameter used by the runtime           |
| `mixture_weights`   | object or null | Effective normalized mixture weights when `family_requested=mixture` |

### Filter sub-object

| Key                   | Type        | Description                                                                                 |
| --------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| `enabled`             | bool        | Whether filtering was active                                                                |
| `wins_ratio`          | float       | Bootstrap wins ratio (when enabled)                                                         |
| `n_valid_oob`         | int         | OOB sample count (when enabled)                                                             |
| `backend`             | str         | Filter implementation (when enabled)                                                        |
| `accepted`            | bool        | Whether the dataset passed (when enabled)                                                   |
| `threshold_requested` | float       | Requested filter threshold before class-aware adjustment                                    |
| `threshold_effective` | float       | Effective threshold used in acceptance decision                                             |
| `threshold_policy`    | str         | Threshold policy identifier (`class_aware_piecewise_v1`)                                    |
| `class_count`         | int or null | Realized class count used by filter (`null` for regression)                                 |
| `class_bucket`        | str         | Class-count bucket for policy lookup (`<=8`, `9-16`, `17-24`, `25-32`, or `not_applicable`) |
| `threshold_delta`     | float       | Difference between requested and effective threshold                                        |

### Class Structure sub-object (classification only)

Present only for classification datasets.

| Key                      | Type        | Description                                        |
| ------------------------ | ----------- | -------------------------------------------------- |
| `n_classes_sampled`      | int         | Layout-sampled class count before postprocessing   |
| `n_classes_realized`     | int         | Unique class count in emitted `y_train` + `y_test` |
| `labels_contiguous`      | bool        | Whether labels form contiguous range `0..K-1`      |
| `train_test_class_match` | bool        | Whether train and test class sets are identical    |
| `min_label`              | int or null | Minimum emitted class label                        |
| `max_label`              | int or null | Maximum emitted class label                        |

### Fixed-layout metadata (optional)

Present only for outputs emitted by fixed-layout batch APIs. These bundles
share one sampled layout and preserve emitted column alignment (feature count,
column order, and lineage feature-to-node mapping).

| Key                | Type | Description                                      |
| ------------------ | ---- | ------------------------------------------------ |
| `layout_mode`      | str  | `"fixed"`                                        |
| `layout_plan_seed` | int  | Seed used to sample the shared fixed-layout plan |
| `layout_signature` | str  | Stable fingerprint for the shared sampled layout |

Fixed-layout APIs validate that the provided `config` remains compatible with
the sampled plan before generation. This prevents plan-driven emitted tensors
from disagreeing with `metadata.config` on layout-driving fields.

### Missingness sub-object (optional)

Present only when missingness is enabled.

| Key                     | Type  | Description                            |
| ----------------------- | ----- | -------------------------------------- |
| `enabled`               | bool  | Always `true` when present             |
| `mechanism`             | str   | `"mcar"`, `"mar"`, or `"mnar"`         |
| `target_rate`           | float | Configured missing rate                |
| `realized_rate_train`   | float | Actual missing fraction in train split |
| `realized_rate_test`    | float | Actual missing fraction in test split  |
| `realized_rate_overall` | float | Actual missing fraction overall        |
| `missing_count_train`   | int   | Number of missing cells in train       |
| `missing_count_test`    | int   | Number of missing cells in test        |
| `missing_count_overall` | int   | Total missing cells                    |

______________________________________________________________________

## Lineage schema

Schema name: `cauchy_generator.dag_lineage`

### Version 1.0.0 (dense, in-memory)

Used in the in-memory metadata `lineage` field during generation. When
lineage is persisted to disk, payloads are rewritten to compact version
`1.1.0`.

```json
{
  "schema_name": "cauchy_generator.dag_lineage",
  "schema_version": "1.0.0",
  "graph": {
    "n_nodes": 8,
    "adjacency": [[0, 1, 0, ...], ...]
  },
  "assignments": {
    "feature_to_node": [2, 3, 5, 7],
    "target_to_node": 7
  }
}
```

- `adjacency` is an n_nodes x n_nodes list of lists. Entries are 0 or 1.
  Upper-triangular only; diagonal is always 0.
- `feature_to_node[i]` is the DAG node index that produces feature `i`.
- `target_to_node` is the DAG node index that produces the target.

### Version 1.1.0 (compact, on-disk)

Used in `metadata.ndjson` dataset records when lineage artifacts are written to disk.
Replaces the dense adjacency matrix with a reference to bitpacked binary
data.

```json
{
  "schema_name": "cauchy_generator.dag_lineage",
  "schema_version": "1.1.0",
  "graph": {
    "n_nodes": 8,
    "edge_count": 12,
    "adjacency_ref": {
      "encoding": "upper_triangle_bitpack_v1",
      "blob_path": "lineage/adjacency.bitpack.bin",
      "index_path": "lineage/adjacency.index.json",
      "dataset_index": 0,
      "bit_offset": 0,
      "bit_length": 28,
      "sha256": "a1b2c3..."
    }
  },
  "assignments": {
    "feature_to_node": [2, 3, 5, 7],
    "target_to_node": 7
  }
}
```

### Adjacency encoding: `upper_triangle_bitpack_v1`

- Packs the `n_nodes * (n_nodes - 1) / 2` upper-triangle bits into bytes.
- Bit order: little-endian.
- `bit_offset` and `bit_length` locate this dataset's bits within the
  shared shard-level blob file.
- `sha256` is a hex-encoded SHA-256 checksum of the packed bytes for this
  dataset's adjacency data.

### Lineage index file

Each shard contains `lineage/adjacency.index.json` with:

- `schema_name` and `schema_version` — echo the lineage schema identifiers.
- `encoding` — always `"upper_triangle_bitpack_v1"`.
- `records` — array of per-dataset offset/length/checksum entries.

______________________________________________________________________

## Contract guarantees

**Determinism** — seed derivation is deterministic. For fixed seed and
configuration, runs are expected to reproduce metadata and numerical outputs
within tolerance. Strict byte-identical tensors/files are not guaranteed
across all backends.

**Feature alignment** — `feature_types[i]` in each metadata record describes
feature index `i` inside packed `x` row vectors and tensor column index `i`
in `X_train` / `X_test`.

**Lineage integrity** — each dataset's bitpacked adjacency data is
protected by a SHA-256 checksum recorded in the metadata.

**Postprocessing invariants**:

- No constant columns — features with zero variance are removed during
  postprocessing.
- Numeric features are clipped and standardized (approximately zero mean,
  unit variance).
- Classification target classes are randomly permuted (label indices carry
  no ordinal meaning).
