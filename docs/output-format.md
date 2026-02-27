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
    dataset_000000/
      train.parquet
      test.parquet
      metadata.json
    dataset_000001/
      ...
    lineage/
      adjacency.bitpack.bin
      adjacency.index.json
  shard_00001/
    ...
```

**Shard naming**: `shard_{id:05d}` — five-digit zero-padded shard ID.
Default: 128 datasets per shard.

**Dataset naming**: `dataset_{index:06d}` — six-digit zero-padded global
dataset index.

**Shard ID calculation**: `dataset_index // shard_size`.

______________________________________________________________________

## Parquet column schema

**Feature columns**: `f_0000`, `f_0001`, ..., `f_{n-1:04d}` — four-digit
zero-padded indices.

**Target column**: `y`.

**Compression**: zstd (default).

Column `f_{i:04d}` corresponds to `feature_types[i]`.

______________________________________________________________________

## Metadata JSON structure

Each dataset's `metadata.json` contains:

### Top-level keys

| Key                      | Type        | Description                                                  |
| ------------------------ | ----------- | ------------------------------------------------------------ |
| `backend`                | str         | Always `"torch"`                                             |
| `device`                 | str         | Compute device (e.g., `"cpu"`, `"cuda"`)                     |
| `compute_backend`        | str         | Implementation variant identifier                            |
| `n_features`             | int         | Number of features                                           |
| `n_categorical_features` | int         | Number of categorical features                               |
| `n_classes`              | int or null | Realized class count in emitted labels (null for regression) |
| `graph_nodes`            | int         | Number of nodes in the DAG                                   |
| `graph_edges`            | int         | Number of edges in the DAG                                   |
| `graph_depth_nodes`      | int         | Longest path length in the DAG                               |
| `graph_edge_density`     | float       | Edge count / max possible edges                              |
| `seed`                   | int         | Base seed for this dataset                                   |
| `attempt_used`           | int         | Generation attempt index (0-based)                           |
| `lineage`                | object      | DAG lineage record (see Lineage below)                       |
| `curriculum`             | object      | Curriculum metadata (see below)                              |
| `config`                 | object      | Full serialized generator configuration                      |
| `filter`                 | object      | Filter results (see below)                                   |
| `class_structure`        | object      | Present only for classification (see below)                  |
| `missingness`            | object      | Present only when missingness is enabled                     |
| `steering`               | object      | Present only when steering is enabled                        |

### Curriculum sub-object

| Key                   | Type        | Description                               |
| --------------------- | ----------- | ----------------------------------------- |
| `mode`                | str         | `"off"`, `"auto"`, or `"fixed"`           |
| `stage`               | int or null | Active stage (1, 2, or 3), or null if off |
| `n_rows_total`        | int         | Total rows (train + test)                 |
| `n_train`             | int         | Training rows                             |
| `n_test`              | int         | Test rows                                 |
| `train_fraction`      | float       | Fraction of rows used for training        |
| `realized_complexity` | object      | Actual structural parameters achieved     |
| `stage_bounds`        | object      | Min/max constraints for this stage        |
| `monotonicity_axes`   | list[str]   | Axes monitored for monotonicity           |

The `realized_complexity` object contains: `n_rows_total`, `n_train`,
`n_test`, `n_features`, `graph_nodes`, `graph_depth_nodes`,
`graph_edge_density`.

The `stage_bounds` object contains nullable min/max pairs:
`n_features_min`, `n_features_max`, `n_nodes_min`, `n_nodes_max`,
`depth_min`, `depth_max`.

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

### Steering sub-object (optional)

Present only when steering is enabled.

| Key                        | Type        | Description                                 |
| -------------------------- | ----------- | ------------------------------------------- |
| `enabled`                  | bool        | Always `true` when present                  |
| `max_attempts`             | int         | Max candidates per slot                     |
| `temperature`              | float       | Softmax temperature                         |
| `candidate_count`          | int         | Candidates actually generated               |
| `candidate_seeds`          | list[int]   | Seed for each candidate                     |
| `scores`                   | list[float] | Weighted distance score per candidate       |
| `probabilities`            | list[float] | Softmax selection probability per candidate |
| `selected_candidate_index` | int         | Index of chosen candidate                   |
| `selected_candidate_seed`  | int         | Seed of chosen candidate                    |
| `selected_score`           | float       | Score of chosen candidate                   |
| `selected_in_band`         | dict        | Per-metric in-band boolean                  |
| `selected_metric_values`   | dict        | Per-metric computed value                   |
| `targets`                  | dict        | Per-metric target spec (min, max, weight)   |

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

Used in `metadata.json` when lineage artifacts are written to disk.
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
      "blob_path": "../lineage/adjacency.bitpack.bin",
      "index_path": "../lineage/adjacency.index.json",
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

**Feature alignment** — `feature_types[i]` describes parquet column
`f_{i:04d}` and tensor column index `i` in `X_train` / `X_test`.

**Lineage integrity** — each dataset's bitpacked adjacency data is
protected by a SHA-256 checksum recorded in the metadata.

**Postprocessing invariants**:

- No constant columns — features with zero variance are removed during
  postprocessing.
- Numeric features are clipped and standardized (approximately zero mean,
  unit variance).
- Classification target classes are randomly permuted (label indices carry
  no ordinal meaning).
