#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/generate-from-config.sh" \
  "configs/preset_cuda_h100.yaml" \
  "${1:-100}" \
  "${2:-cuda}" \
  "${3:-data/run_h100_$(date +%Y%m%d_%H%M%S)}" \
  "${4:-}"
