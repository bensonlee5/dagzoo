#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

NUM_DATASETS="${1:-10}"
DEVICE="${2:-auto}"
OUT_DIR="${3:-data/run_curriculum_$(date +%Y%m%d_%H%M%S)}"
SEED_ARG="${4:-}"
CURRICULUM_MODE="${5:-auto}"

CMD=(
  uv run cauchy-gen generate
  --config "configs/curriculum_tabiclv2.yaml"
  --num-datasets "$NUM_DATASETS"
  --device "$DEVICE"
  --out "$OUT_DIR"
  --curriculum "$CURRICULUM_MODE"
)

if [[ -n "$SEED_ARG" ]]; then
  CMD+=(--seed "$SEED_ARG")
fi

echo "Running curriculum generation: ${CMD[*]}"
"${CMD[@]}"
