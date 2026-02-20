#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-cpu}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found in PATH" >&2
  exit 1
fi

uv run cauchy-gen benchmark --suite smoke --profile "$PROFILE"
