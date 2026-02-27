#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/generate-from-config.sh" \
  "configs/preset_many_class_generate_smoke.yaml" \
  "${1:-10}" \
  "${2:-cpu}" \
  "${3:-data/run_many_class_$(date +%Y%m%d_%H%M%S)}" \
  "${4:-}"
