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
- `"cat"` — categorical feature. Observed values are integer indices in the
  range `0 .. cardinality - 1`. When missingness is enabled, missing values are
  encoded as `NaN`.

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

`metadata` contains the dataset-level generation metadata described below.

### Top-level keys

| Key                          | Type        | Description                                                                                         |
| ---------------------------- | ----------- | --------------------------------------------------------------------------------------------------- |
| `backend`                    | str         | Always `"torch"`                                                                                    |
| `device`                     | str         | Compute device (e.g., `"cpu"`, `"cuda"`)                                                            |
| `requested_device`           | str         | Requested runtime device after CLI/config normalization (for example `auto`, `cpu`, `cuda`, `mps`)  |
| `resolved_device`            | str         | Runtime backend selected from the requested device for generation                                   |
| `device_fallback_reason`     | str or null | Reserved field retained for artifact-contract stability; currently always `null`                    |
| `compute_backend`            | str         | Implementation variant identifier                                                                   |
| `n_features`                 | int         | Number of features                                                                                  |
| `n_categorical_features`     | int         | Number of categorical features                                                                      |
| `n_classes`                  | int or null | Realized class count in emitted labels (null for regression)                                        |
| `graph_nodes`                | int         | Number of nodes in the DAG                                                                          |
| `graph_edges`                | int         | Number of edges in the DAG                                                                          |
| `graph_depth_nodes`          | int         | Longest path length in the DAG                                                                      |
| `graph_edge_density`         | float       | Edge count / max possible edges                                                                     |
| `seed`                       | int         | Replay seed recorded by the emitting API. Canonical generation stores the shared run seed here.     |
| `dataset_seed`               | int         | Optional canonical per-dataset child seed derived from `seed`; used for deferred replay/diagnostics |
| `dataset_index`              | int         | Optional canonical dataset position within the run (0-based)                                        |
| `run_num_datasets`           | int         | Optional canonical run length used to replay the saved bundle                                       |
| `attempt_used`               | int         | Generation attempt index (0-based)                                                                  |
| `lineage`                    | object      | DAG lineage record (see Lineage below)                                                              |
| `shift`                      | object      | Resolved shift settings and realized observability signals                                          |
| `noise_distribution`         | object      | Resolved noise-family selection and effective sampling params                                       |
| `config`                     | object      | Full serialized generator configuration                                                             |
| `filter`                     | object      | Filter results (see below)                                                                          |
| `class_structure`            | object      | Present only for classification (see below)                                                         |
| `missingness`                | object      | Present only when missingness is enabled                                                            |
| `layout_mode`                | str         | Optional canonical layout metadata (`"fixed"` for canonical generation outputs)                     |
| `layout_plan_seed`           | int         | Optional internal seed used to sample the shared per-run layout                                     |
| `layout_signature`           | str         | Optional deterministic fingerprint for the shared sampled layout                                    |
| `layout_plan_signature`      | str         | Optional deterministic fingerprint for the internal frozen node execution payload                   |
| `layout_plan_schema_version` | int         | Optional internal metadata version for the canonical shared-layout payload                          |
| `layout_execution_contract`  | str         | Optional internal execution contract identifier for canonical determinism                           |
| `keyed_replay`               | object      | Optional exact keyed subtree replay paths for canonical layout, execution, and dataset roots        |

For canonical generation (`generate_one`, `generate_batch`, `generate_batch_iter`,
and `dagzoo generate`), replay later bundles with the shared `seed`,
`run_num_datasets`, and `dataset_index` by regenerating the canonical batch and
selecting that index. `dataset_seed` preserves the per-bundle child seed for
deferred replay and diagnostics. Exact keyed subtree replay uses `seed`
together with the `keyed_replay` paths.

### `keyed_replay` sub-object

Present for canonical generation outputs. These paths are interpreted relative
to `KeyedRng(metadata["seed"])`:

| Key                        | Type             | Description                                                 |
| -------------------------- | ---------------- | ----------------------------------------------------------- |
| `layout_root_path`         | list[str \| int] | Exact keyed path for replaying the shared per-run layout    |
| `execution_plan_root_path` | list[str \| int] | Exact keyed path for replaying the shared execution subtree |
| `dataset_root_path`        | list[str \| int] | Exact keyed path for replaying one bundle’s dataset subtree |

### Shift sub-object

Present for all generated bundles. When shift is disabled, scales are `0.0` and
multipliers are `1.0`.

| Key                         | Type  | Description                                                 |
| --------------------------- | ----- | ----------------------------------------------------------- |
| `enabled`                   | bool  | Whether shift controls were enabled                         |
| `mode`                      | str   | Resolved shift mode (`off`, `graph_drift`, etc.)            |
| `graph_scale`               | float | Resolved graph drift scale                                  |
| `mechanism_scale`           | float | Resolved mechanism drift scale                              |
| `variance_scale`            | float | Resolved noise drift scale                                  |
| `edge_logit_bias_shift`     | float | Additive shift applied to edge logits                       |
| `mechanism_logit_tilt`      | float | Mechanism-family tilt applied at sampling                   |
| `variance_sigma_multiplier` | float | Sigma multiplier applied to stochastic noise                |
| `edge_odds_multiplier`      | float | Edge-odds multiplier (`exp(edge_logit_bias_shift)`)         |
| `noise_variance_multiplier` | float | Noise-variance multiplier (`variance_sigma_multiplier^2`)   |
| `mechanism_nonlinear_mass`  | float | Probability mass on nonlinear mechanism families (`[0, 1]`) |

### Noise Distribution sub-object

Present for all generated bundles.

| Key                 | Type           | Description                                                             |
| ------------------- | -------------- | ----------------------------------------------------------------------- |
| `family_requested`  | str            | Configured noise family (`gaussian`, `laplace`, `student_t`, `mixture`) |
| `family_sampled`    | str            | Effective family used by the dataset generation runtime                 |
| `sampling_strategy` | str            | Runtime selection strategy (`dataset_level`)                            |
| `base_scale`        | float          | Base noise scale from config                                            |
| `student_t_df`      | float          | Student-t degrees of freedom parameter used by the runtime              |
| `mixture_weights`   | object or null | Effective normalized mixture weights when `family_requested=mixture`    |

### Filter sub-object

| Key                   | Type        | Description                                                                                  |
| --------------------- | ----------- | -------------------------------------------------------------------------------------------- |
| `mode`                | str         | Filter execution mode. Current value is `deferred`.                                          |
| `status`              | str         | `not_run` for freshly generated outputs; `accepted`/`rejected` after `dagzoo filter`.        |
| `enabled`             | bool        | Present after deferred filter replay. Always `true` when replayed.                           |
| `accepted`            | bool        | Present after deferred filter replay.                                                        |
| `wins_ratio`          | float       | Bootstrap wins ratio (present after deferred replay).                                        |
| `n_valid_oob`         | int         | OOB sample count (present after deferred replay).                                            |
| `backend`             | str         | Filter implementation identifier (present after deferred replay).                            |
| `threshold_requested` | float       | Requested filter threshold before class-aware adjustment (present after deferred replay).    |
| `threshold_effective` | float       | Effective threshold used in acceptance decision (present after deferred replay).             |
| `threshold_policy`    | str         | Threshold policy identifier (`class_aware_piecewise_v1`) (present after deferred replay).    |
| `class_count`         | int or null | Realized class count used by filter (`null` for regression) (present after deferred replay). |
| `class_bucket`        | str         | Class-count bucket for policy lookup (present after deferred replay).                        |
| `threshold_delta`     | float       | Difference between requested and effective threshold (present after deferred replay).        |
| `reason`              | str         | Present on rejected outputs when replay emits a specific rejection reason.                   |

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

### Fixed-layout metadata

Present for all canonical generation outputs. These bundles share one sampled
layout per run and preserve emitted column alignment (feature count, column
order, and lineage feature-to-node mapping) within that run.

| Key                          | Type   | Description                                                         |
| ---------------------------- | ------ | ------------------------------------------------------------------- |
| `layout_mode`                | str    | `"fixed"`                                                           |
| `layout_plan_seed`           | int    | Internal seed used to sample the shared per-run layout              |
| `layout_signature`           | str    | Stable fingerprint for the shared sampled layout                    |
| `layout_plan_signature`      | str    | Stable fingerprint for the frozen internal node payload             |
| `layout_plan_schema_version` | int    | Internal canonical layout metadata version                          |
| `layout_execution_contract`  | str    | Internal execution contract (`chunk_batched_v1`)                    |
| `keyed_replay`               | object | Exact keyed subtree replay paths for layout/execution/dataset roots |

Under `chunk_batched_v1`, canonical fixed-layout outputs are deterministic for
the same run seed and realized run shape. Internal plan metadata records the
shared sampled layout and execution-plan fingerprint used for that run, while
`keyed_replay` records the exact keyed subtree roots needed for internal replay.

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

Schema name: `dagzoo.dag_lineage`

### Version 1.0.0 (dense, in-memory)

Used in the in-memory metadata `lineage` field during generation. When
lineage is persisted to disk, payloads are rewritten to compact version
`1.1.0`.

```json
{
  "schema_name": "dagzoo.dag_lineage",
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
  Upper-triangular only; diagonal is always 0. Direction convention is
  `adjacency[src][dst]` (`src -> dst`), so parents of node `j` are found from column `j`.
- `feature_to_node[i]` is the DAG node index that produces feature `i`.
- `target_to_node` is the DAG node index that produces the target.

### Version 1.1.0 (compact, on-disk)

Used in `metadata.ndjson` dataset records when lineage artifacts are written to disk.
Replaces the dense adjacency matrix with a reference to bitpacked binary
data.

```json
{
  "schema_name": "dagzoo.dag_lineage",
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

- Canonical generation (`generate_one`, `generate_batch`, `generate_batch_iter`)
  is fixed-layout-backed and preserves emitted feature schema across the run:
  constant-column removal and feature-column permutation are disabled.
- Numeric features are clipped and standardized (approximately zero mean,
  unit variance).
- Classification target classes are randomly permuted (label indices carry
  no ordinal meaning).
