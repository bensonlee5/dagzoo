# Architecture Diagrams

Visual documentation of `cauchy-generator` control flow and data flow.

## Reading Guide

- Core generation is torch-native end-to-end (CPU/CUDA/MPS execution paths).
- Curriculum and stage resolution are handled in `core/curriculum.py`.
- Dataset layout and graph sampling are handled in `core/layout.py`.
- Steering candidate scoring uses torch-native metrics (`core/steering_metrics.py`).
- Diagnostics coverage extraction uses `diagnostics/metrics.py`, which normalizes bundles to CPU and delegates all mathematical metric computation to the unified torch-native steering metrics.

## 1. High-Level System Overview

CLI has three commands. `generate` and `benchmark` both apply hardware-aware tuning; `hardware` only inspects hardware.

```mermaid
flowchart TB
    CLI["cauchy-gen CLI<br/><code>cli.main()</code>"]
    CLI --> Generate["generate"]
    CLI --> Benchmark["benchmark"]
    CLI --> Hardware["hardware"]

    subgraph GenFlow["Generate Command"]
        direction TB
        GenCfg["GeneratorConfig.from_yaml()"]
        GenHW["detect_hardware() + apply_hardware_profile()"]
        GenIter["generate_batch_iter()"]
        MissingInject["inject_missingness() + metadata<br/>(when configured)"]
        DiagWrap{"diagnostics enabled?"}
        DiagAgg["CoverageAggregator.update_bundle()"]
        WriteCheck{"--no-write?"}
        Parquet["write_parquet_shards_stream()"]
        InMem["In-memory generation only"]
        DiagArtifacts["coverage_summary.json / coverage_summary.md"]

        GenCfg --> GenHW --> GenIter --> MissingInject --> DiagWrap
        DiagWrap -- yes --> DiagAgg --> WriteCheck
        DiagWrap -- no --> WriteCheck
        WriteCheck -- no --> Parquet
        WriteCheck -- yes --> InMem
        DiagAgg --> DiagArtifacts
    end

    subgraph BenchFlow["Benchmark Command"]
        direction TB
        BenchCfg["default config or --config"]
        Specs["resolve_profile_run_specs()"]
        Suite["run_benchmark_suite()<br/>per-profile detect_hardware() + apply_hardware_profile()"]
        Baseline["load_baseline() / build_baseline_payload()"]
        MissingGuard["missingness_guardrails<br/>(acceptance + runtime control check)"]
        Report["write_suite_json() / write_suite_markdown()"]

        BenchCfg --> Specs --> Suite --> MissingGuard --> Report
        Baseline --> Suite
    end

    subgraph HwFlow["Hardware Command"]
        direction TB
        Detect["detect_hardware()"]
        Print["Print backend/device/profile"]
        Detect --> Print
    end

    Generate --> GenFlow
    Benchmark --> BenchFlow
    Hardware --> HwFlow
```

### Context

- `generate` streams bundles; writing and diagnostics are stream consumers, not separate generation passes.
- Missingness injection is applied inside generation (postprocess boundary) when configured, and emitted in bundle metadata.
- `--no-write` keeps generation in memory while still allowing diagnostics artifact output when enabled.
- `benchmark` runs generation repeatedly through profile specs, can emit diagnostics per profile, and records missingness guardrail outcomes when missingness is enabled.
- `hardware` does not load `GeneratorConfig`.

## 2. Generation Pipeline Control Flow

Core runtime flow from `generate_batch_iter()` through seed derivation, curriculum/layout sampling, graph generation, postprocessing, and steering selection.

```mermaid
flowchart TB
    Entry["generate_batch_iter()<br/><code>core/dataset.py</code>"]
    Entry --> Resolve["Resolve requested/resolved device,<br/>steering settings, SeedManager(run_seed)"]
    Resolve --> Loop["For dataset i in range(num_datasets)"]
    Loop --> Seed["dataset_seed = manager.child('dataset', i)"]

    Seed --> SteerCheck{"steering enabled?"}
    SteerCheck -- no --> OneSeeded
    SteerCheck -- yes --> Steered

    subgraph Steered["_generate_one_steered()"]
        direction TB
        CandLoop["candidate_idx in 0..max_attempts"]
        CandSeed["candidate_seed (deterministic child seed)"]
        CandGen["_generate_one_seeded()"]
        CandMetrics["extract_steering_metrics()<br/>(torch-native metric subset)"]
        CandScore["_score_candidate_against_targets()"]
        CandSelect["_select_softmax_candidate()<br/>(torch softmax + torch RNG)"]
        CandLoop --> CandSeed --> CandGen --> CandMetrics --> CandScore
        CandScore --> CandLoop
        CandScore --> CandSelect
    end

    subgraph OneSeeded["_generate_one_seeded()"]
        direction TB
        SM["SeedManager(seed)"]
        Curric["_sample_curriculum()<br/><code>core/curriculum.py</code>"]
        Layout["_sample_layout()<br/><code>core/layout.py</code>"]
        TorchGen["_generate_torch()"]
        SM --> Curric --> Layout --> TorchGen
    end

    subgraph TorchRetry["_generate_torch() retry loop"]
        direction TB
        GraphGen["_generate_graph_dataset_torch()<br/>DAG traversal + node pipeline"]
        Filter{"_apply_filter_torch()<br/>RF learnability filter"}
        Retry["Retry with seed + attempt"]
        Shuffle["Shuffle rows via randperm"]
        Split["Train/test split"]
        Post["postprocess_dataset()<br/>torch-native clipping/standardization/permutation"]
        Missing["inject_missingness()<br/>MCAR/MAR/MNAR (optional)"]
        Validate{"classification split valid?"}
        Bundle["DatasetBundle<br/>(X_train, y_train, X_test, y_test, feature_types, metadata)"]

        GraphGen --> Filter
        Filter -- rejected --> Retry --> GraphGen
        Filter -- accepted --> Shuffle --> Split --> Post --> Missing --> Validate
        Validate -- invalid --> Retry
        Validate -- valid --> Bundle
    end

    OneSeeded --> TorchRetry --> Yield["yield bundle"]
    Steered --> Selected["selected DatasetBundle"]
    Selected --> Yield
    Yield --> Loop
```

### Context

- Steering runs a bounded candidate loop per dataset, then selects one candidate probabilistically using deterministic seeded RNG.
- The main generation path is torch-native. Diagnostics extraction is separate from steering decisions.
- Retry behavior is inside `_generate_torch()` and applies to both steered and non-steered generation.
- Missingness injection happens after postprocess and before final bundle emission, with deterministic seed lineage.

## 3. DAG Node Data Flow

Data flow inside `_generate_graph_dataset_torch()`: root nodes sample base random points; child nodes transform parent outputs; node outputs are converted/extracted into final `X` and `y`.

```mermaid
flowchart TB
    subgraph GraphSetup["Graph Setup"]
        DAG["sample_cauchy_dag()<br/><code>graph/cauchy_graph.py</code>"]
        Assign["_sample_assignments()<br/><code>core/layout.py</code>"]
        Specs["_build_node_specs()<br/><code>core/layout.py</code>"]
    end

    DAG --> Walk
    Assign --> Specs
    Walk["For node_idx in 0..n_nodes"] --> ParentCheck{"Has parent outputs?"}

    ParentCheck -- "no (root)" --> SamplePts["sample_random_points()<br/>(n_rows x total_dim)"]
    ParentCheck -- "yes (child)" --> MultiFunc["apply_multi_function()<br/>compose parent tensors"]

    SamplePts --> Pipeline
    MultiFunc --> Pipeline

    subgraph Pipeline["apply_node_pipeline()"]
        direction TB
        Clean["nan_to_num + clamp"]
        Std["standardize()"]
        Weight["sample_random_weights() x broadcast"]
        Norm["Normalize by mean L2 norm"]
        Convert["Per ConverterSpec:<br/>apply_categorical_converter() or apply_numeric_converter()"]
        Scale["Scale by log_uniform(0.1, 10)"]
        Clean --> Std --> Weight --> Norm --> Convert --> Scale
    end

    Pipeline --> NodeOut["node_outputs[node_idx] = x_node"]
    Convert --> Extract["extracted values:<br/>feature_i and/or target"]
    NodeOut --> Walk
    Extract --> Assemble

    subgraph Assemble["Assemble Dataset Tensors"]
        direction TB
        FeatureFill["Populate feature columns from extracted feature_values"]
        TargetFill["Populate y from extracted target_values"]
        Fallback["If missing values:<br/>sample fallback torch random values"]
        FeatureFill --> Fallback
        TargetFill --> Fallback
    end

    Assemble --> Out["Return (X, y, filter_details)"]
```

### Context

- Node execution order follows DAG topological index order (`0..n_nodes-1` with upper-triangular adjacency).
- Parent-to-child flow is tensor-based and device-aware.
- Converters both transform node-local representation and emit extracted feature/target values used to assemble final dataset tensors.

## 4. Steering and Diagnostics Feedback Loops

Steering and diagnostics are related but distinct loops: steering influences candidate selection; diagnostics aggregates reporting metrics across emitted bundles.

```mermaid
flowchart TB
    subgraph Steering["Steering Loop (_generate_one_steered)"]
        direction TB
        Init["Init steering_state per metric:<br/>selected=0, in_band=0"]
        Cand["Generate candidate bundle"]
        SteerMetrics["extract_steering_metrics()<br/>core/steering_metrics.py"]
        Score["Distance-to-band scoring +<br/>under-coverage reweighting"]
        Select["Softmax select by temperature<br/>(torch RNG, deterministic by seed)"]
        Update["Update steering_state counts"]
        Winner["Selected bundle + steering metadata"]

        Init --> Cand --> SteerMetrics --> Score --> Select --> Update --> Winner
    end

    subgraph Diagnostics["Diagnostics CoverageAggregator"]
        direction TB
        Stream["Wrap generated bundle stream"]
        UpdateBundle["aggregator.update_bundle(bundle)"]
        FullMetrics["extract_dataset_metrics()<br/>diagnostics/metrics.py (delegates to unified torch metrics)"]
        Accum["_MetricAccumulator.update()"]
        Summary["aggregator.build_summary()"]
        OutJSON["write_coverage_summary_json()"]
        OutMD["write_coverage_summary_markdown()"]

        Stream --> UpdateBundle --> FullMetrics --> Accum --> Summary
        Summary --> OutJSON
        Summary --> OutMD
    end

    subgraph MissingnessGuardrails["Missingness Guardrails (benchmark mode)"]
        direction TB
        MGEnabled{"missingness enabled?"}
        MGCollect["Collect bundle.metadata['missingness']<br/>coverage + realized-rate checks"]
        MGBaseline["Run missingness-off control throughput"]
        MGStatus["Emit missingness_guardrails<br/>status/issues in profile summary"]
        MGDisabled["Emit missingness_guardrails<br/>enabled=false"]

        MGEnabled -- yes --> MGCollect --> MGBaseline --> MGStatus
        MGEnabled -- no --> MGDisabled
    end

    Winner --> Stream
    Stream --> MGEnabled

    subgraph StateFlow["Run-Level State"]
        direction LR
        SS["steering_state persists across datasets in generate_batch_iter()"]
        CA["CoverageAggregator accumulates across yielded bundles"]
    end
```

### Context

- Steering metrics are a targeted subset optimized for selection-time performance.
- Diagnostics metrics are broader and reporting-focused; extraction runs on CPU-normalized bundles via torch-native metric computation.
- Steering and diagnostics can be enabled independently, though they are often used together.
- Missingness guardrails are benchmark-only and activate when missingness is enabled in the resolved profile config.
