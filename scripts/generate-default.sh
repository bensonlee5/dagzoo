#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/generate-from-config.sh" \
  "configs/default.yaml" \
  "${1:-10}" \
  "${2:-auto}" \
  "${3:-data/run_default_$(date +%Y%m%d_%H%M%S)}" \
  "${4:-}"
