# Config Resolution

Canonical precedence and effective-config behavior for `dagzoo` runtime commands.

This document is the source of truth for:

- which config source wins when multiple sources set the same field
- where effective config artifacts are written
- how to inspect field-level override traces

______________________________________________________________________

## Generate precedence

`dagzoo generate` resolves config in this order (later steps win):

1. Base YAML (`--config`)
1. CLI device override (`--device`) -> `runtime.device`
1. Hardware policy transforms (`--hardware-policy`)
1. Default CUDA fixed-layout auto-batch floor
   (`runtime.fixed_layout_target_cells`) based on detected GPU memory, applied
   only when the config leaves that field unset
1. CLI rows override (`--rows`) -> `dataset.rows`
1. Missingness CLI overrides:
   - `--missing-rate`
   - `--missing-mechanism`
   - `--missing-mar-observed-fraction`
   - `--missing-mar-logit-scale`
   - `--missing-mnar-logit-scale`
1. Diagnostics CLI switch (`--diagnostics`) -> `diagnostics.enabled=true`

After overrides are applied, staged generation validation runs. Invalid combinations fail fast.

`dataset.rows` accepts fixed total rows (`1024`), ranges (`400..60000`), and CSV choices (`1024,2048,4096`).

______________________________________________________________________

## Request precedence

`dagzoo request` resolves config in this order (later steps win):

1. Base YAML `configs/default.yaml`
1. Request profile overlay:
   - `profile=default`: no extra overlay
   - `profile=smoke`: set `dataset.n_train=128`, `dataset.n_test=32`,
     `dataset.n_features_min=8`, `dataset.n_features_max=12`,
     `graph.n_nodes_min=2`, and `graph.n_nodes_max=12`
1. Request task override -> `dataset.task`
1. Request seed override -> `seed` (when `seed` is set in the request file)
1. Request rows override -> `dataset.rows`
1. Request missingness profile overlay copied from the canonical missingness presets:
   - `none`
   - `mcar`
   - `mar`
   - `mnar`
1. Request output root -> `output.out_dir = <output_root>/generated`
1. CLI device override (`--device`) -> `runtime.device`
1. Hardware policy transforms (`--hardware-policy`)
1. Default CUDA fixed-layout auto-batch floor
   (`runtime.fixed_layout_target_cells`) based on detected GPU memory, applied
   only when the config leaves that field unset
1. Request execution only:
   - `profile=smoke` caps or clears `dataset.rows` against the smoke total-row
     envelope before one-time run realization and records trace source
     `request.smoke_rows_cap`

`dagzoo request` does not add a parallel config surface and does not re-enable
inline filtering. Request runs always execute as canonical generation into
`<output_root>/generated` followed by deferred filtering into
`<output_root>/filter` with accepted-only shards under `<output_root>/curated`.

______________________________________________________________________

## Benchmark precedence

Each preset in `dagzoo benchmark` resolves independently in this order:

1. Base preset config:
   - built-in preset config (`cpu`, `cuda_desktop`, `cuda_h100`) or
   - custom config (`--config` + `--preset custom`)
1. Preset device selection:
   - preset-defined device, or
   - CLI `--device` when a single preset run is selected
1. Hardware policy transforms (`--hardware-policy`)
1. Default CUDA fixed-layout auto-batch floor
   (`runtime.fixed_layout_target_cells`) based on detected GPU memory, applied
   only when the config leaves that field unset
1. Suite caps for `--suite smoke`:
   - `dataset.n_train <= 256`
   - `dataset.n_test <= 128`
   - `dataset.n_features_min/max <= 24`
   - `graph.n_nodes_min/max <= 16`

Runtime count overrides (`--num-datasets`, `--warmup`) are benchmark execution controls and do
not mutate the preset effective config payload.

`dagzoo benchmark` rejects `--device` when multiple presets are selected. Use
per-preset config/device settings for multi-preset runs instead of a shared CLI
override.

`dagzoo benchmark` supports `dataset.rows` in preset configs. Standard/full
suite resolution preserves the rows spec until benchmark orchestration
realizes one run shape. Smoke suites cap the rows spec before that one-time run
realization so smoke benchmarks stay within the intended split-size envelope.

______________________________________________________________________

## Validation stages

`GeneratorConfig` validation is explicit and runs in three stages:

1. Stage 1: field-level normalization and typing per section (`dataset`, `graph`, `mechanism`, `shift`, `noise`, `runtime`, `output`, `diagnostics`, `benchmark`, `filter`)
1. Stage 2: cross-field constraints (for example shift mode compatibility, missingness constraints, and min/max envelopes)
1. Stage 3: post-override revalidation by re-running stage 1 + stage 2 through `GeneratorConfig.validate_generation_constraints()`

`resolve_generate_config()` and `resolve_benchmark_preset_config()` both call stage 3 after applying all runtime overrides/caps.

______________________________________________________________________

## Effective config artifacts

### Generate

Each run writes:

- `<effective_config_root>/effective_config.yaml`
- `<effective_config_root>/effective_config_trace.yaml`

`effective_config_root` is:

1. `--out`, else
1. `output.out_dir` from config, else
1. `--diagnostics-out-dir`, else
1. `diagnostics.out_dir` from config, else
1. `effective_config_artifacts/`

### Request

Each request run writes:

- `<output_root>/generated/effective_config.yaml`
- `<output_root>/generated/effective_config_trace.yaml`

The same request root also reserves:

- `<output_root>/generated/` for raw generated shard outputs
- `<output_root>/filter/` for deferred-filter artifacts
- `<output_root>/curated/` for accepted-only curated shards

### Benchmark

When benchmark artifact output is enabled (`--out-dir` or default timestamped path), each run writes:

- `<artifact_dir>/effective_configs/<preset>.yaml`
- `<artifact_dir>/effective_configs/<preset>_trace.yaml`

If the same preset key appears multiple times in one run, files are suffixed with `_runN`.

Benchmark effective-config artifacts capture the realized per-run config. When
benchmark resolution starts from `dataset.rows`, the stored effective config
records the realized `dataset.n_train` / `dataset.n_test` for that run and
clears `dataset.rows`.

______________________________________________________________________

## Resolution trace schema

Trace artifacts are ordered lists of field-level override events:

```yaml
- path: runtime.device
  source: cli.device
  old_value: auto
  new_value: cpu
- path: dataset.n_train
  source: benchmark.suite_smoke_caps
  old_value: 512
  new_value: 256
- path: runtime.fixed_layout_target_cells
  source: hardware.default_cuda_fixed_layout_target_cells
  old_value: 32000000
  new_value: 160000000
```

Fields:

- `path`: dotted config path that changed
- `source`: override source identifier
- `old_value`: value before this override
- `new_value`: value after this override

Use CLI flags to print payloads to stdout:

- `dagzoo generate --print-effective-config --print-resolution-trace ...`
- `dagzoo benchmark --print-effective-config --print-resolution-trace ...`
