#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

MECHANISM="${1:-mcar}"
MISSING_RATE="${2:-0.2}"
NUM_DATASETS="${3:-10}"
DEVICE="${4:-auto}"
OUT_DIR="${5:-data/run_missing_${MECHANISM}_$(date +%Y%m%d_%H%M%S)}"
SEED_ARG="${6:-}"

case "$MECHANISM" in
  mcar|mar|mnar) ;;
  *)
    echo "Error: mechanism must be one of: mcar, mar, mnar" >&2
    exit 1
    ;;
esac

CMD=(
  uv run cauchy-gen generate
  --config "configs/default.yaml"
  --num-datasets "$NUM_DATASETS"
  --device "$DEVICE"
  --out "$OUT_DIR"
  --missing-rate "$MISSING_RATE"
  --missing-mechanism "$MECHANISM"
)

if [[ -n "$SEED_ARG" ]]; then
  CMD+=(--seed "$SEED_ARG")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
