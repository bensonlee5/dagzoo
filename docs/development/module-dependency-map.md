# Module Dependency Map

This file is generated from imports under `src/dagzoo`.
Run `./scripts/dev deps --write-docs` after changing internal module edges.

## Package Dependency DAG

The package graph below collapses strongly connected components so the result stays acyclic.

- `dagzoo` depends on `dagzoo.config`, `dagzoo.hardware`, `dagzoo.hardware_policy`, `dagzoo.types`, `dagzoo.core`, `dagzoo.functions`, `dagzoo.postprocess`, `dagzoo.sampling`
- `dagzoo.__main__` depends on `dagzoo.cli`
- `dagzoo.cli` depends on `dagzoo.config`, `dagzoo.filtering`, `dagzoo.hardware`, `dagzoo.hardware_policy`, `dagzoo.io`, `dagzoo.rng`, `dagzoo.bench`, `dagzoo.diagnostics`, `dagzoo.diagnostics_targets`, `dagzoo.core`, `dagzoo.functions`, `dagzoo.postprocess`, `dagzoo.sampling`
- `dagzoo.converters` depends on `dagzoo.rng`, `dagzoo.core`, `dagzoo.functions`, `dagzoo.postprocess`, `dagzoo.sampling`
- `dagzoo.linalg` depends on `dagzoo.math_utils`, `dagzoo.core`, `dagzoo.functions`, `dagzoo.postprocess`, `dagzoo.sampling`
- `dagzoo.bench`, `dagzoo.diagnostics`, `dagzoo.diagnostics_targets` depends on `dagzoo.config`, `dagzoo.filtering`, `dagzoo.hardware`, `dagzoo.hardware_policy`, `dagzoo.io`, `dagzoo.math_utils`, `dagzoo.rng`, `dagzoo.types`, `dagzoo.core`, `dagzoo.functions`, `dagzoo.postprocess`, `dagzoo.sampling`
- `dagzoo.core`, `dagzoo.functions`, `dagzoo.postprocess`, `dagzoo.sampling` depends on `dagzoo.config`, `dagzoo.filtering`, `dagzoo.graph`, `dagzoo.hardware`, `dagzoo.hardware_policy`, `dagzoo.io`, `dagzoo.math_utils`, `dagzoo.rng`, `dagzoo.types`
- `dagzoo.filtering` depends on `dagzoo.config`, `dagzoo.io`, `dagzoo.math_utils`, `dagzoo.rng`
- `dagzoo.graph` has no internal package dependencies
- `dagzoo.hardware_policy` depends on `dagzoo.config`, `dagzoo.hardware`
- `dagzoo.config` depends on `dagzoo.math_utils`, `dagzoo.rng`
- `dagzoo.hardware` has no internal package dependencies
- `dagzoo.io` depends on `dagzoo.math_utils`, `dagzoo.types`
- `dagzoo.math_utils` has no internal package dependencies
- `dagzoo.rng` has no internal package dependencies
- `dagzoo.types` has no internal package dependencies

## Change-Impact Hotspots

The sections below list direct importers and full transitive downstream modules.
Use them to predict which runtime paths are likely to move when a hot module changes.

### `dagzoo.core.execution_semantics`

- Path: `src/dagzoo/core/execution_semantics.py`
- Imports: `dagzoo.core.execution_sampling_common`, `dagzoo.core.fixed_layout_plan_types`, `dagzoo.core.layout_types`, `dagzoo.core.shift`, `dagzoo.functions.activations`, `dagzoo.math_utils`, `dagzoo.rng`
- Direct downstream modules: `dagzoo.converters.categorical`, `dagzoo.converters.numeric`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.node_pipeline`, `dagzoo.functions.multi`, `dagzoo.functions.random_functions`, `dagzoo.sampling.random_points`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.converters`, `dagzoo.converters.categorical`, `dagzoo.converters.numeric`, `dagzoo.core`, `dagzoo.core.dataset`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.node_pipeline`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`, `dagzoo.functions.multi`, `dagzoo.functions.random_functions`, `dagzoo.sampling.random_points`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.converters`, `dagzoo.diagnostics`, `dagzoo.functions`, `dagzoo.sampling`

### `dagzoo.core.fixed_layout_batched`

- Path: `src/dagzoo/core/fixed_layout_batched.py`
- Imports: `dagzoo.config`, `dagzoo.core.execution_semantics`, `dagzoo.core.fixed_layout_batch_common`, `dagzoo.core.fixed_layout_batch_functions`, `dagzoo.core.fixed_layout_plan_types`, `dagzoo.core.layout`, `dagzoo.core.layout_types`, `dagzoo.rng`, `dagzoo.sampling.noise`
- Direct downstream modules: `dagzoo.converters.categorical`, `dagzoo.converters.numeric`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.node_pipeline`, `dagzoo.functions.multi`, `dagzoo.functions.random_functions`, `dagzoo.sampling.random_points`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.converters`, `dagzoo.converters.categorical`, `dagzoo.converters.numeric`, `dagzoo.core`, `dagzoo.core.dataset`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.node_pipeline`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`, `dagzoo.functions.multi`, `dagzoo.functions.random_functions`, `dagzoo.sampling.random_points`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.converters`, `dagzoo.diagnostics`, `dagzoo.functions`, `dagzoo.sampling`

### `dagzoo.core.fixed_layout_runtime`

- Path: `src/dagzoo/core/fixed_layout_runtime.py`
- Imports: `dagzoo.config`, `dagzoo.core.fixed_layout`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_grouped`, `dagzoo.core.fixed_layout_plan_types`, `dagzoo.core.fixed_layout_prepare`, `dagzoo.core.generation_context`, `dagzoo.core.generation_runtime`, `dagzoo.core.layout`, `dagzoo.core.layout_types`, `dagzoo.core.noise_runtime`, `dagzoo.core.shift`, `dagzoo.core.validation`, `dagzoo.rng`, `dagzoo.types`
- Direct downstream modules: `dagzoo.bench.corpus_probe`, `dagzoo.bench.suite`, `dagzoo.cli.commands.generate`, `dagzoo.core.dataset`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.core`, `dagzoo.core.dataset`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.diagnostics`

### `dagzoo.core.layout`

- Path: `src/dagzoo/core/layout.py`
- Imports: `dagzoo.config`, `dagzoo.core.fixed_layout_plan_types`, `dagzoo.core.layout_types`, `dagzoo.core.shift`, `dagzoo.functions._rng_helpers`, `dagzoo.graph`, `dagzoo.rng`, `dagzoo.sampling`
- Direct downstream modules: `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_runtime`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.converters`, `dagzoo.converters.categorical`, `dagzoo.converters.numeric`, `dagzoo.core`, `dagzoo.core.dataset`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.node_pipeline`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`, `dagzoo.functions.multi`, `dagzoo.functions.random_functions`, `dagzoo.sampling.random_points`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.converters`, `dagzoo.diagnostics`, `dagzoo.functions`, `dagzoo.sampling`

### `dagzoo.core.node_pipeline`

- Path: `src/dagzoo/core/node_pipeline.py`
- Imports: `dagzoo.core.execution_semantics`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_plan_types`, `dagzoo.core.layout_types`, `dagzoo.rng`, `dagzoo.sampling.noise`
- Direct downstream modules: `dagzoo.bench.micro`
- Transitive downstream modules: `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.micro`, `dagzoo.bench.suite`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`

### `dagzoo.core.dataset`

- Path: `src/dagzoo/core/dataset.py`
- Imports: `dagzoo.config`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.rng`, `dagzoo.types`
- Direct downstream modules: `dagzoo`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.core`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.core`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.diagnostics`

### `dagzoo.core.config_resolution`

- Path: `src/dagzoo/core/config_resolution.py`
- Imports: `dagzoo.config`, `dagzoo.hardware`, `dagzoo.hardware_policy`
- Direct downstream modules: `dagzoo.bench.preset_specs`, `dagzoo.bench.suite`, `dagzoo.cli.commands.generate`
- Transitive downstream modules: `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.preset_specs`, `dagzoo.bench.suite`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`

### `dagzoo.core.generation_runtime`

- Path: `src/dagzoo/core/generation_runtime.py`
- Imports: `dagzoo.config`, `dagzoo.core.layout_types`, `dagzoo.core.metadata`, `dagzoo.core.noise_runtime`, `dagzoo.core.shift`, `dagzoo.core.validation`, `dagzoo.postprocess.postprocess`, `dagzoo.rng`, `dagzoo.types`
- Direct downstream modules: `dagzoo.core.fixed_layout_runtime`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.core`, `dagzoo.core.dataset`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.diagnostics`

### `dagzoo.config`

- Path: `src/dagzoo/config/__init__.py`
- Imports: `dagzoo.config.constants`, `dagzoo.config.io`, `dagzoo.config.models`, `dagzoo.config.normalization`, `dagzoo.config.rows`
- Direct downstream modules: `dagzoo`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.preset_specs`, `dagzoo.bench.runtime_support`, `dagzoo.bench.stage_metrics`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.common`, `dagzoo.cli.effective_config`, `dagzoo.cli.parsing`, `dagzoo.core.config_resolution`, `dagzoo.core.dataset`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_grouped`, `dagzoo.core.fixed_layout_prepare`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.generation_context`, `dagzoo.core.generation_runtime`, `dagzoo.core.layout`, `dagzoo.core.noise_runtime`, `dagzoo.core.shift`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`, `dagzoo.diagnostics_targets`, `dagzoo.filtering.deferred_filter`, `dagzoo.filtering.deferred_filter_replay`, `dagzoo.hardware_policy`, `dagzoo.postprocess.postprocess`, `dagzoo.sampling.missingness`, `dagzoo.sampling.noise`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.preset_specs`, `dagzoo.bench.runtime_support`, `dagzoo.bench.stage_metrics`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.filter`, `dagzoo.cli.commands.generate`, `dagzoo.cli.commands.hardware`, `dagzoo.cli.common`, `dagzoo.cli.effective_config`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.cli.parsing`, `dagzoo.converters`, `dagzoo.converters.categorical`, `dagzoo.converters.numeric`, `dagzoo.core`, `dagzoo.core.config_resolution`, `dagzoo.core.dataset`, `dagzoo.core.execution_semantics`, `dagzoo.core.fixed_layout_batch_common`, `dagzoo.core.fixed_layout_batch_functions`, `dagzoo.core.fixed_layout_batched`, `dagzoo.core.fixed_layout_grouped`, `dagzoo.core.fixed_layout_prepare`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.generation_context`, `dagzoo.core.generation_runtime`, `dagzoo.core.layout`, `dagzoo.core.metadata`, `dagzoo.core.metrics_torch`, `dagzoo.core.node_pipeline`, `dagzoo.core.noise_runtime`, `dagzoo.core.shift`, `dagzoo.diagnostics`, `dagzoo.diagnostics.coverage`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`, `dagzoo.diagnostics.metrics`, `dagzoo.diagnostics_targets`, `dagzoo.filtering`, `dagzoo.filtering.deferred_filter`, `dagzoo.filtering.deferred_filter_replay`, `dagzoo.functions.multi`, `dagzoo.functions.random_functions`, `dagzoo.hardware_policy`, `dagzoo.linalg`, `dagzoo.linalg.random_matrices`, `dagzoo.postprocess`, `dagzoo.postprocess.postprocess`, `dagzoo.sampling`, `dagzoo.sampling.missingness`, `dagzoo.sampling.noise`, `dagzoo.sampling.random_points`, `dagzoo.sampling.random_weights`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.converters`, `dagzoo.core`, `dagzoo.diagnostics`, `dagzoo.diagnostics_targets`, `dagzoo.filtering`, `dagzoo.functions`, `dagzoo.hardware_policy`, `dagzoo.linalg`, `dagzoo.postprocess`, `dagzoo.sampling`

### `dagzoo.io.lineage_schema`

- Path: `src/dagzoo/io/lineage_schema.py`
- Imports: none
- Direct downstream modules: `dagzoo.core.metadata`, `dagzoo.io`, `dagzoo.io.parquet_writer`
- Transitive downstream modules: `dagzoo`, `dagzoo.__main__`, `dagzoo.bench`, `dagzoo.bench.collectors`, `dagzoo.bench.corpus_probe`, `dagzoo.bench.guardrails`, `dagzoo.bench.micro`, `dagzoo.bench.runtime_support`, `dagzoo.bench.stage_metrics`, `dagzoo.bench.suite`, `dagzoo.bench.throughput`, `dagzoo.cli`, `dagzoo.cli.commands.benchmark`, `dagzoo.cli.commands.diagnostics`, `dagzoo.cli.commands.generate`, `dagzoo.cli.entrypoint`, `dagzoo.cli.parser`, `dagzoo.core`, `dagzoo.core.dataset`, `dagzoo.core.fixed_layout_runtime`, `dagzoo.core.generation_runtime`, `dagzoo.core.metadata`, `dagzoo.core.metrics_torch`, `dagzoo.diagnostics`, `dagzoo.diagnostics.coverage`, `dagzoo.diagnostics.effective_diversity`, `dagzoo.diagnostics.effective_diversity.artifacts`, `dagzoo.diagnostics.effective_diversity.calibration`, `dagzoo.diagnostics.effective_diversity.runner`, `dagzoo.diagnostics.metrics`, `dagzoo.diagnostics_targets`, `dagzoo.filtering`, `dagzoo.filtering.deferred_filter`, `dagzoo.filtering.deferred_filter_artifacts`, `dagzoo.io`, `dagzoo.io.parquet_writer`
- Downstream package areas: `dagzoo.bench`, `dagzoo.cli`, `dagzoo.core`, `dagzoo.diagnostics`, `dagzoo.diagnostics_targets`, `dagzoo.filtering`
