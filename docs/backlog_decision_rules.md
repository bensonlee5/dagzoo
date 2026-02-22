# Backlog Decision Rules

This document defines how improvement ideas are ranked and when they are ready for implementation PRs.

## Objective

Primary objective:

- Maximize expected downstream model quality gain.

Secondary objectives:

- Keep implementation risk manageable.
- Preserve reproducibility and throughput guardrails.

## Scoring Rubric

Each candidate idea is scored as:

`final_score = quality_gain + evidence_strength + coverage_gap + architecture_fit - effort_penalty - risk_penalty`

Component definitions:

- `quality_gain` (0-5):
  expected impact on downstream model quality.
- `evidence_strength` (0-3):
  confidence from primary literature and consistency across sources.
- `coverage_gap` (0-3):
  degree to which the idea closes known synthetic-vs-real data blind spots.
- `architecture_fit` (0-2):
  compatibility with current module boundaries (`config`, `core`, `functions`, `postprocess`, `filtering`).
- `effort_penalty` (0-3):
  engineering complexity and cross-module coordination cost.
- `risk_penalty` (0-3):
  regression risk, maintainability risk, or unclear operational behavior.

Priority bands:

- `P0`: score >= 7
- `P1`: score 5-6
- `P2`: score \<= 4

Tie-breakers (in order):

1. Higher `quality_gain`.
1. Higher `evidence_strength`.
1. Lower `effort_penalty`.

## Go/No-Go Gates For Implementation PRs

A backlog item is `go` only if all gates pass:

1. Problem statement:
   explicit quality target and failure mode addressed.
1. Interface spec:
   config/API additions are documented with defaults and backward compatibility.
1. Test plan:
   unit + integration + reproducibility scenarios are defined.
1. Benchmark guardrail:
   expected runtime/memory impact and pass/fail thresholds are stated.
1. Rollback plan:
   feature flag or config disable path exists.

Automatic `no-go` triggers:

- Missing deterministic behavior expectations for fixed seeds.
- Missing acceptance criteria for both classification and regression paths when applicable.
- New config knobs without safe defaults preserving current behavior.

## Default Assumptions For Future Work

- New behavior is opt-in unless explicitly replacing broken behavior.
- Existing config files remain valid without modification.
- Benchmarks use current threshold conventions from `GeneratorConfig.benchmark`.
- No changes to `reference/` documents; those remain source artifacts.

## Required Artifacts Per Implemented Item

Every future implementation PR should include:

- Updated design note in `docs/`.
- Config schema updates and examples in `configs/`.
- Tests covering invariants, reproducibility, and integration paths.
- Benchmark delta summary for affected profiles.
