#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

CONFIG_PATH="${1:-configs/default.yaml}"
NUM_DATASETS="${2:-2}"
DEVICE="${3:-cpu}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found: $CONFIG_PATH" >&2
  exit 1
fi

CMD=(
  uv run dagzoo generate
  --config "$CONFIG_PATH"
  --num-datasets "$NUM_DATASETS"
  --device "$DEVICE"
  --no-dataset-write
)

echo "Running smoke generation: ${CMD[*]}"
"${CMD[@]}"
