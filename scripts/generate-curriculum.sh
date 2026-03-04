#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/generate-curriculum.sh \
    --base-config <path> \
    --out-root <dir> \
    --datasets-per-stage <int> \
    --n-test <int> \
    (--train-start <int> --train-stop <int> --train-step <int> | --train-values <csv>) \
    [--chunk-size <int>] \
    [--device <auto|cpu|cuda|mps>] \
    [--hardware-policy <none|cuda_tiered_v1>] \
    [--seed <int>] \
    [--n-features <int> | --stage-columns <csv>] \
    [--no-dataset-write] \
    [--dry-run]

Notes:
- Stage rows are required via range or CSV values.
- Columns are optional:
  - --n-features applies one fixed column count to every stage.
  - --stage-columns applies per-stage column counts and must match stage row count.
- chunk-size controls sequential datasets per generate call (not parallel workers).
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 1
}

PYTHON_RUNNER=()
GEN_RUNNER=()
if [[ -n "${CURRICULUM_PYTHON_BIN:-}" ]]; then
  PYTHON_RUNNER=("${CURRICULUM_PYTHON_BIN}")
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_RUNNER=("${REPO_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=("uv" "run" "python")
else
  die "Python runner not found; set CURRICULUM_PYTHON_BIN, create .venv, or install uv"
fi

if [[ -n "${CURRICULUM_DAGSYNTH_BIN:-}" ]]; then
  GEN_RUNNER=("${CURRICULUM_DAGSYNTH_BIN}")
elif [[ -x "${REPO_ROOT}/.venv/bin/dagzoo" ]]; then
  GEN_RUNNER=("${REPO_ROOT}/.venv/bin/dagzoo")
elif command -v uv >/dev/null 2>&1; then
  GEN_RUNNER=("uv" "run" "dagzoo")
else
  die "dagzoo runner not found; set CURRICULUM_DAGSYNTH_BIN, create .venv, or install uv"
fi

is_int() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

is_pos_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

BASE_CONFIG=""
OUT_ROOT=""
DATASETS_PER_STAGE=""
N_TEST=""
TRAIN_START=""
TRAIN_STOP=""
TRAIN_STEP=""
TRAIN_VALUES=""
CHUNK_SIZE="64"
DEVICE="auto"
SEED_OVERRIDE=""
N_FEATURES=""
STAGE_COLUMNS=""
HARDWARE_POLICY="none"
NO_WRITE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config)
      [[ $# -ge 2 ]] || die "--base-config requires a value"
      BASE_CONFIG="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || die "--out-root requires a value"
      OUT_ROOT="$2"
      shift 2
      ;;
    --datasets-per-stage)
      [[ $# -ge 2 ]] || die "--datasets-per-stage requires a value"
      DATASETS_PER_STAGE="$2"
      shift 2
      ;;
    --n-test)
      [[ $# -ge 2 ]] || die "--n-test requires a value"
      N_TEST="$2"
      shift 2
      ;;
    --train-start)
      [[ $# -ge 2 ]] || die "--train-start requires a value"
      TRAIN_START="$2"
      shift 2
      ;;
    --train-stop)
      [[ $# -ge 2 ]] || die "--train-stop requires a value"
      TRAIN_STOP="$2"
      shift 2
      ;;
    --train-step)
      [[ $# -ge 2 ]] || die "--train-step requires a value"
      TRAIN_STEP="$2"
      shift 2
      ;;
    --train-values)
      [[ $# -ge 2 ]] || die "--train-values requires a value"
      TRAIN_VALUES="$2"
      shift 2
      ;;
    --chunk-size)
      [[ $# -ge 2 ]] || die "--chunk-size requires a value"
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --hardware-policy)
      [[ $# -ge 2 ]] || die "--hardware-policy requires a value"
      HARDWARE_POLICY="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a value"
      SEED_OVERRIDE="$2"
      shift 2
      ;;
    --n-features)
      [[ $# -ge 2 ]] || die "--n-features requires a value"
      N_FEATURES="$2"
      shift 2
      ;;
    --stage-columns)
      [[ $# -ge 2 ]] || die "--stage-columns requires a value"
      STAGE_COLUMNS="$2"
      shift 2
      ;;
    --no-dataset-write)
      NO_WRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$BASE_CONFIG" ]] || die "--base-config is required"
[[ -n "$OUT_ROOT" ]] || die "--out-root is required"
[[ -n "$DATASETS_PER_STAGE" ]] || die "--datasets-per-stage is required"
[[ -n "$N_TEST" ]] || die "--n-test is required"
[[ -f "$BASE_CONFIG" ]] || die "config file not found: $BASE_CONFIG"

is_pos_int "$DATASETS_PER_STAGE" || die "--datasets-per-stage must be a positive integer"
is_pos_int "$N_TEST" || die "--n-test must be a positive integer"
is_pos_int "$CHUNK_SIZE" || die "--chunk-size must be a positive integer"

case "$DEVICE" in
  auto|cpu|cuda|mps) ;;
  *) die "--device must be one of: auto, cpu, cuda, mps" ;;
esac

[[ -n "$HARDWARE_POLICY" ]] || die "--hardware-policy must be non-empty"

if [[ -n "$N_FEATURES" && -n "$STAGE_COLUMNS" ]]; then
  die "--n-features and --stage-columns are mutually exclusive"
fi
if [[ -n "$N_FEATURES" ]]; then
  is_pos_int "$N_FEATURES" || die "--n-features must be a positive integer"
fi

have_range=0
if [[ -n "$TRAIN_START" || -n "$TRAIN_STOP" || -n "$TRAIN_STEP" ]]; then
  have_range=1
fi
have_values=0
if [[ -n "$TRAIN_VALUES" ]]; then
  have_values=1
fi
if (( have_range == have_values )); then
  die "Specify exactly one of range (--train-start/--train-stop/--train-step) or --train-values"
fi

TRAIN_ROWS=()
if (( have_values == 1 )); then
  IFS=',' read -r -a raw_rows <<< "$TRAIN_VALUES"
  [[ ${#raw_rows[@]} -gt 0 ]] || die "--train-values must include at least one row count"
  for raw in "${raw_rows[@]}"; do
    row="$(echo "$raw" | xargs)"
    is_pos_int "$row" || die "Invalid train row value in --train-values: $raw"
    TRAIN_ROWS+=("$row")
  done
else
  is_int "$TRAIN_START" || die "--train-start must be an integer"
  is_int "$TRAIN_STOP" || die "--train-stop must be an integer"
  is_int "$TRAIN_STEP" || die "--train-step must be an integer"
  (( TRAIN_STEP != 0 )) || die "--train-step must not be 0"
  (( TRAIN_START > 0 )) || die "--train-start must be > 0"
  (( TRAIN_STOP > 0 )) || die "--train-stop must be > 0"

  if (( TRAIN_STEP > 0 && TRAIN_START > TRAIN_STOP )); then
    die "Range is empty: start > stop with positive step"
  fi
  if (( TRAIN_STEP < 0 && TRAIN_START < TRAIN_STOP )); then
    die "Range is empty: start < stop with negative step"
  fi

  val=$TRAIN_START
  while true; do
    TRAIN_ROWS+=("$val")
    next=$((val + TRAIN_STEP))
    if (( TRAIN_STEP > 0 )); then
      (( next <= TRAIN_STOP )) || break
    else
      (( next >= TRAIN_STOP )) || break
    fi
    val=$next
  done
fi

[[ ${#TRAIN_ROWS[@]} -gt 0 ]] || die "No stage rows resolved"

STAGE_COLS=()
if [[ -n "$STAGE_COLUMNS" ]]; then
  IFS=',' read -r -a raw_cols <<< "$STAGE_COLUMNS"
  [[ ${#raw_cols[@]} -gt 0 ]] || die "--stage-columns must include at least one value"
  for raw in "${raw_cols[@]}"; do
    col="$(echo "$raw" | xargs)"
    is_pos_int "$col" || die "Invalid stage column value in --stage-columns: $raw"
    STAGE_COLS+=("$col")
  done
  if (( ${#STAGE_COLS[@]} != ${#TRAIN_ROWS[@]} )); then
    die "--stage-columns count (${#STAGE_COLS[@]}) must match stage rows count (${#TRAIN_ROWS[@]})"
  fi
fi

if [[ -n "$SEED_OVERRIDE" ]]; then
  is_int "$SEED_OVERRIDE" || die "--seed must be an integer"
  RUN_SEED="$SEED_OVERRIDE"
else
  RUN_SEED="$("${PYTHON_RUNNER[@]}" - "$BASE_CONFIG" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = yaml.safe_load(f) or {}
seed = payload.get("seed", 1)
print(int(seed))
PY
)"
fi

mkdir -p "$OUT_ROOT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

RECORDS_TSV="$TMP_DIR/chunk_records.tsv"
: > "$RECORDS_TSV"

RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STATUS="success"
ERROR_MESSAGE=""
TOTAL_GENERATED=0

create_stage_config() {
  local stage_config="$1"
  local n_train="$2"
  local n_test="$3"
  local n_features="$4"

  "${PYTHON_RUNNER[@]}" - "$BASE_CONFIG" "$stage_config" "$n_train" "$n_test" "$n_features" <<'PY'
import sys
import yaml

base_path, out_path, n_train, n_test, n_features = sys.argv[1:]
with open(base_path, "r", encoding="utf-8") as f:
    payload = yaml.safe_load(f) or {}
if not isinstance(payload, dict):
    raise ValueError("Base config must be a mapping")

dataset = payload.setdefault("dataset", {})
if not isinstance(dataset, dict):
    raise ValueError("dataset section must be a mapping")

dataset["n_train"] = int(n_train)
dataset["n_test"] = int(n_test)
if n_features:
    n_feat = int(n_features)
    dataset["n_features_min"] = n_feat
    dataset["n_features_max"] = n_feat

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(payload, f, sort_keys=False)
PY
}

for idx in "${!TRAIN_ROWS[@]}"; do
  n_train="${TRAIN_ROWS[$idx]}"
  stage_features=""
  if [[ -n "$N_FEATURES" ]]; then
    stage_features="$N_FEATURES"
  elif (( ${#STAGE_COLS[@]} > 0 )); then
    stage_features="${STAGE_COLS[$idx]}"
  fi

  stage_seed=$((RUN_SEED + idx * 1000003))
  stage_id="stage_$(printf '%03d' "$idx")_train_${n_train}"
  stage_dir="$OUT_ROOT/$stage_id"
  stage_cfg="$TMP_DIR/${stage_id}.yaml"
  create_stage_config "$stage_cfg" "$n_train" "$N_TEST" "$stage_features"

  remaining="$DATASETS_PER_STAGE"
  chunk_idx=0

  while (( remaining > 0 )); do
    if (( remaining < CHUNK_SIZE )); then
      chunk_n="$remaining"
    else
      chunk_n="$CHUNK_SIZE"
    fi
    chunk_seed=$((stage_seed + chunk_idx))

    chunk_out=""
    if (( NO_WRITE == 0 )); then
      chunk_out="$stage_dir/chunk_$(printf '%03d' "$chunk_idx")"
      mkdir -p "$chunk_out"
    fi

    cmd=(
      "${GEN_RUNNER[@]}" generate
      --config "$stage_cfg"
      --num-datasets "$chunk_n"
      --device "$DEVICE"
      --hardware-policy "$HARDWARE_POLICY"
      --seed "$chunk_seed"
    )
    if (( NO_WRITE == 1 )); then
      cmd+=(--no-dataset-write)
    else
      cmd+=(--out "$chunk_out")
    fi

    status="ok"
    exit_code=0

    if (( DRY_RUN == 1 )); then
      echo "DRY RUN: ${cmd[*]}"
      status="dry_run"
    else
      echo "Running stage=${stage_id} chunk=${chunk_idx} rows(train/test)=${n_train}/${N_TEST} datasets=${chunk_n} seed=${chunk_seed}"
      set +e
      "${cmd[@]}"
      exit_code=$?
      set -e
      if (( exit_code != 0 )); then
        status="failed"
        RUN_STATUS="failed"
        ERROR_MESSAGE="Stage ${stage_id}, chunk ${chunk_idx} failed with exit code ${exit_code}"
      fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$idx" "$stage_id" "$n_train" "$N_TEST" "$stage_features" "$chunk_idx" "$chunk_seed" "$chunk_n" "$chunk_out" "$status" "$exit_code" \
      >> "$RECORDS_TSV"

    if [[ "$status" == "failed" ]]; then
      break 2
    fi

    if [[ "$status" == "ok" ]]; then
      TOTAL_GENERATED=$((TOTAL_GENERATED + chunk_n))
    fi
    remaining=$((remaining - chunk_n))
    chunk_idx=$((chunk_idx + 1))
  done
done

RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MANIFEST_PATH="$OUT_ROOT/curriculum_manifest.json"

TRAIN_ROWS_CSV="$(IFS=,; echo "${TRAIN_ROWS[*]}")"
STAGE_COLS_CSV=""
if (( ${#STAGE_COLS[@]} > 0 )); then
  STAGE_COLS_CSV="$(IFS=,; echo "${STAGE_COLS[*]}")"
fi

"${PYTHON_RUNNER[@]}" - "$MANIFEST_PATH" "$RECORDS_TSV" "$BASE_CONFIG" "$OUT_ROOT" "$RUN_STARTED_AT" "$RUN_COMPLETED_AT" "$RUN_STATUS" "$ERROR_MESSAGE" "$RUN_SEED" "$DEVICE" "$HARDWARE_POLICY" "$NO_WRITE" "$CHUNK_SIZE" "$DATASETS_PER_STAGE" "$N_TEST" "$TRAIN_ROWS_CSV" "$STAGE_COLS_CSV" "$N_FEATURES" "$TOTAL_GENERATED" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

(
    manifest_path,
    records_path,
    base_config,
    out_root,
    run_started_at,
    run_completed_at,
    run_status,
    error_message,
    run_seed,
    device,
    hardware_policy,
    no_write,
    chunk_size,
    datasets_per_stage,
    n_test,
    train_rows_csv,
    stage_cols_csv,
    n_features,
    total_generated,
) = sys.argv[1:]

records_by_stage: dict[int, list[dict[str, object]]] = defaultdict(list)
with open(records_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        (
            stage_index,
            stage_id,
            n_train,
            stage_n_test,
            stage_features,
            chunk_index,
            chunk_seed,
            chunk_datasets,
            chunk_out,
            status,
            exit_code,
        ) = line.split("\t")
        stage_idx_i = int(stage_index)
        records_by_stage[stage_idx_i].append(
            {
                "chunk_index": int(chunk_index),
                "chunk_seed": int(chunk_seed),
                "num_datasets": int(chunk_datasets),
                "out_dir": chunk_out or None,
                "status": status,
                "exit_code": int(exit_code),
            }
        )

train_rows = [int(v) for v in train_rows_csv.split(",") if v]
stage_cols = [int(v) for v in stage_cols_csv.split(",") if v] if stage_cols_csv else None
stages_payload = []
for idx, n_train in enumerate(train_rows):
    chunks = sorted(records_by_stage.get(idx, []), key=lambda x: int(x["chunk_index"]))
    if not chunks:
        stage_status = "skipped"
    elif any(str(c["status"]) == "failed" for c in chunks):
        stage_status = "failed"
    elif all(str(c["status"]) == "dry_run" for c in chunks):
        stage_status = "dry_run"
    else:
        stage_status = "success"

    stage_features = None
    if stage_cols is not None and idx < len(stage_cols):
        stage_features = int(stage_cols[idx])
    elif n_features:
        stage_features = int(n_features)

    stage_seed = int(run_seed) + idx * 1000003
    stages_payload.append(
        {
            "stage_index": idx,
            "stage_id": f"stage_{idx:03d}_train_{n_train}",
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_features": stage_features,
            "stage_seed": int(stage_seed),
            "generated_datasets": int(sum(int(c["num_datasets"]) for c in chunks if str(c["status"]) == "ok")),
            "status": stage_status,
            "chunks": chunks,
        }
    )

payload = {
    "schema_version": 1,
    "base_config": str(Path(base_config).resolve()),
    "out_root": str(Path(out_root).resolve()),
    "run_started_at": run_started_at,
    "run_completed_at": run_completed_at,
    "status": run_status,
    "error": error_message or None,
    "run_seed": int(run_seed),
    "device": device,
    "hardware_policy": hardware_policy,
    "no_write": bool(int(no_write)),
    "chunk_size": int(chunk_size),
    "datasets_per_stage": int(datasets_per_stage),
    "n_test": int(n_test),
    "train_rows": train_rows,
    "stage_columns": stage_cols,
    "fixed_n_features": int(n_features) if n_features else None,
    "total_generated_datasets": int(total_generated),
    "stages": stages_payload,
}

manifest_file = Path(manifest_path)
manifest_file.parent.mkdir(parents=True, exist_ok=True)
with manifest_file.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
PY

echo "Wrote curriculum manifest: $MANIFEST_PATH"
if [[ "$RUN_STATUS" == "failed" ]]; then
  echo "$ERROR_MESSAGE" >&2
  exit 1
fi

echo "Completed curriculum run. stages=${#TRAIN_ROWS[@]} total_generated=${TOTAL_GENERATED}"
