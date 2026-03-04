#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-cpu}"
DIAGNOSTICS="${2:-off}"
DIAGNOSTICS_OUT_DIR="${3:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found in PATH" >&2
  exit 1
fi

args=(benchmark --suite smoke --preset "$PRESET")
if [[ "$DIAGNOSTICS" == "on" || "$DIAGNOSTICS" == "true" || "$DIAGNOSTICS" == "1" ]]; then
  args+=(--diagnostics)
fi
if [[ -n "$DIAGNOSTICS_OUT_DIR" ]]; then
  args+=(--diagnostics-out-dir "$DIAGNOSTICS_OUT_DIR")
fi

uv run dagzoo "${args[@]}"
