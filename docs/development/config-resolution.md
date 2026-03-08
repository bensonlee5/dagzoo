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
   (`runtime.fixed_layout_target_cells`) based on detected GPU memory
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
   (`runtime.fixed_layout_target_cells`) based on detected GPU memory
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

`dagzoo benchmark` currently rejects configs that set `dataset.rows`; benchmark paths still require explicit split sizing (`dataset.n_train`/`dataset.n_test`).

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

### Benchmark

When benchmark artifact output is enabled (`--out-dir` or default timestamped path), each run writes:

- `<artifact_dir>/effective_configs/<preset>.yaml`
- `<artifact_dir>/effective_configs/<preset>_trace.yaml`

If the same preset key appears multiple times in one run, files are suffixed with `_runN`.

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
