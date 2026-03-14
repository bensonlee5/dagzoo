# Request File Contract

Public v1 request-file contract for downstream corpus requests.

Use this document when a downstream consumer needs to tell `dagzoo request`
what to generate without depending on the full internal config surface. For the
artifacts that `dagzoo request` produces, reference
[Output Format](output-format.md).

`dagzoo request --request <path>` resolves this public request contract into
canonical `generate -> filter` execution and publishes
`handoff_manifest.json` at the request `output_root`.

## v1 goals

- keep the public field set small
- expose generation intent, not internal engine wiring
- preserve a stable public profile vocabulary instead of repo-local config names
- expose missingness only as a named selector in v1

## v1 schema

Required fields:

- `version`: exact string `v1`
- `task`: `classification` or `regression`
- `dataset_count`: positive integer
- `rows`: total-row spec in one of the three public v1 forms only:
  - fixed integer, for example `1024`
    - quoted numeric strings such as `"1024"` are invalid; fixed rows must be encoded as an integer
  - range string, for example `1024..4096`
  - CSV choice string, for example `1024,2048,4096`
  - YAML list or mapping forms are not part of the request-file contract, even
    though internal `GeneratorConfig` normalization accepts them
- `profile`: stable public request profile; v1 supports `default` and `smoke`
  - `smoke` keeps the request run within the smoke split envelope during execution
    by capping or clearing the active rows spec before run realization so the
    smoke `n_train` / `n_test` split is preserved
- `output_root`: non-empty output root string

Optional fields:

- `missingness_profile`: `none`, `mcar`, `mar`, or `mnar`
  - default: `none`
- `seed`: optional 32-bit integer for reproducible request-driven runs

Round-trip serialization from `RequestFileConfig.to_dict()` emits the same
public wire shape:

- fixed rows serialize as an integer
- range rows serialize as `start..stop`
- choice rows serialize as a CSV string
- unset `seed` is omitted instead of serialized as `null`

## Execution Layout

`dagzoo request --request <path>` treats `output_root` as the request-run root
and writes these artifacts:

- `handoff_manifest.json`: versioned machine-readable handoff manifest for
  downstream consumers
- `generated/`: raw generated shard outputs plus
  `effective_config.yaml` and `effective_config_trace.yaml`
- `filter/`: deferred-filter artifacts
  (`filter_manifest.ndjson` and `filter_summary.json`)
- `curated/`: accepted-only curated shard outputs

## Handoff Manifest

`handoff_manifest.json` is the stable downstream entrypoint for request-driven
corpus handoff. It is a JSON document with:

- `schema_name: dagzoo_request_handoff_manifest`
- `schema_version: 2`
- `identity`: stable request-run and corpus identifiers plus source-family tag
- `request`: request-file echo (`path` plus normalized public `payload`)
- `artifacts`: absolute paths for the request-run root, generated output,
  filtered corpus, filter summary/manifest, and effective-config artifacts
- `artifacts_relative`: manifest-relative paths for portable downstream use
- `checksums`: SHA-256 digests for the effective-config and filter artifacts
- `summary`: generated / accepted / rejected dataset counts plus acceptance rate
- `throughput`: generation-stage and filter-stage elapsed time plus
  datasets-per-minute context
- `hardware`: requested/resolved device context and the applied hardware policy
- `diversity_artifacts`: nullable paths for request-associated diversity reports
- `defaults`: canonical downstream-consumption defaults, including the
  accepted-only curated corpus recommendation

The manifest `throughput` block uses request-run wall-clock stage timing.
`filter/filter_summary.json` remains the underlying deferred-filter artifact and
keeps its own timing semantics.

Downstream consumers should treat `artifacts_relative` as the portable contract
surface and `defaults.recommended_training_corpus=curated` as the canonical
training target for mainline research runs.

`dagzoo request` does not run `dagzoo diversity-audit` implicitly, so the
`diversity_artifacts` paths are currently `null` unless a separate workflow
stores request-associated diversity reports alongside the run.

## Out Of Scope

The request file must not expose raw internal config sections or split-level row
controls. In particular, v1 does not allow:

- `runtime`, `filter`, `graph`, `mechanism`, `noise`, `shift`, `diagnostics`, `benchmark`, `output`, or nested `dataset` sections
- `n_train` or `n_test`
- direct missingness tuning fields such as `missing_rate` or MAR/MNAR logit knobs
- repo-local preset/config file references
- closed-loop feedback or downstream prediction ingestion semantics

## Mapping Intent

The public request contract resolves onto canonical `generate -> filter` runs
plus one machine-readable handoff manifest.

- `profile` maps to a stable internal config choice without exposing config
  file names as the public contract.
- `missingness_profile` maps to canonical missingness behavior without exposing
  full internal missingness tuning in v1.
- `rows` remains the only public row-shape control.

## Valid Examples

Minimal classification request:

```yaml
version: v1
task: classification
dataset_count: 25
rows: 1024
profile: default
output_root: requests/corpus_default
```

Smoke regression request with range rows and explicit missingness:

```yaml
version: v1
task: regression
dataset_count: 3
rows: 1024..4096
profile: smoke
missingness_profile: mcar
output_root: requests/regression_smoke
seed: 42
```

Choice-based row request:

```yaml
version: v1
task: classification
dataset_count: 8
rows: 1024,2048,4096
profile: default
missingness_profile: mar
output_root: requests/profile_matrix
```

## Invalid Examples

Unknown version:

```yaml
version: v2
task: classification
dataset_count: 1
rows: 1024
profile: default
output_root: requests/out
```

Internal config leakage:

```yaml
version: v1
task: classification
dataset_count: 1
rows: 1024
profile: default
output_root: requests/out
runtime:
  device: cpu
```

Split-level row controls instead of `rows`:

```yaml
version: v1
task: classification
dataset_count: 1
profile: default
output_root: requests/out
n_train: 768
n_test: 256
```

Direct missingness tuning instead of the named selector:

```yaml
version: v1
task: classification
dataset_count: 1
rows: 1024
profile: default
output_root: requests/out
missing_rate: 0.25
missing_mechanism: mar
```

Internal-only rows mapping instead of a public v1 encoding:

```yaml
version: v1
task: classification
dataset_count: 2
rows:
  mode: fixed
  value: 1024
profile: default
output_root: requests/out
```
