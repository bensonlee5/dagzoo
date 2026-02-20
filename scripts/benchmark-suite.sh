#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-standard}"
PROFILE="${2:-all}"
OUT_DIR="${3:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found in PATH" >&2
  exit 1
fi

args=(benchmark --suite "$SUITE" --profile "$PROFILE")
if [[ -n "$OUT_DIR" ]]; then
  args+=(--out-dir "$OUT_DIR")
fi

uv run cauchy-gen "${args[@]}"
