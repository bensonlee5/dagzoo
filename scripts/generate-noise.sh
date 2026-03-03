#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

FAMILY="${1:-mixture}"
NUM_DATASETS="${2:-10}"
DEVICE="${3:-cpu}"
OUT_DIR="${4:-data/run_noise_${FAMILY}_$(date +%Y%m%d_%H%M%S)}"
SEED_ARG="${5:-}"

case "$FAMILY" in
  legacy)
    CONFIG_PATH="configs/default.yaml"
    ;;
  gaussian)
    CONFIG_PATH="configs/preset_noise_gaussian_generate_smoke.yaml"
    ;;
  laplace)
    CONFIG_PATH="configs/preset_noise_laplace_generate_smoke.yaml"
    ;;
  student_t)
    CONFIG_PATH="configs/preset_noise_student_t_generate_smoke.yaml"
    ;;
  mixture)
    CONFIG_PATH="configs/preset_noise_mixture_generate_smoke.yaml"
    ;;
  *)
    echo "Error: family must be one of: legacy, gaussian, laplace, student_t, mixture" >&2
    exit 1
    ;;
esac

"${SCRIPT_DIR}/generate-from-config.sh" \
  "$CONFIG_PATH" \
  "$NUM_DATASETS" \
  "$DEVICE" \
  "$OUT_DIR" \
  "$SEED_ARG"
