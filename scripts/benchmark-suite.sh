#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-standard}"
PRESET="${2:-all}"
OUT_DIR="${3:-}"
DIAGNOSTICS="${4:-off}"
DIAGNOSTICS_OUT_DIR="${5:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found in PATH" >&2
  exit 1
fi

args=(benchmark --suite "$SUITE" --preset "$PRESET")
if [[ -n "$OUT_DIR" ]]; then
  args+=(--out-dir "$OUT_DIR")
fi
if [[ "$DIAGNOSTICS" == "on" || "$DIAGNOSTICS" == "true" || "$DIAGNOSTICS" == "1" ]]; then
  args+=(--diagnostics)
fi
if [[ -n "$DIAGNOSTICS_OUT_DIR" ]]; then
  args+=(--diagnostics-out-dir "$DIAGNOSTICS_OUT_DIR")
fi

uv run dagzoo "${args[@]}"
