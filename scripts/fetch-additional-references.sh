#!/usr/bin/env bash
set -euo pipefail

mkdir -p reference

# id filename
papers=(
  "2311.10609 Scaling_TabPFN_2311.10609.pdf"
  "2402.11137 TuneTables_2402.11137.pdf"
  "2406.05207 LoCalPFN_Retrieval_FineTuning_2406.05207.pdf"
  "2406.05216 TabPFGen_2406.05216.pdf"
  "2410.18164 TabDPT_2410.18164.pdf"
  "2411.10634 Drift_Resilient_TabPFN_2411.10634.pdf"
  "2502.02527 TabPFN_Unleashed_2502.02527.pdf"
  "2502.05564 TabICL_2502.05564.pdf"
  "2502.06684 EquiTabPFN_2502.06684.pdf"
  "2506.10914 Foundation_Models_for_Causal_Inference_2506.10914.pdf"
)

for entry in "${papers[@]}"; do
  id="${entry%% *}"
  file="${entry#* }"
  out="reference/${file}"
  if [[ -f "$out" ]]; then
    echo "skip: $out"
    continue
  fi
  echo "download: $id -> $out"
  curl -fL "https://arxiv.org/pdf/${id}.pdf" -o "$out"
done

echo "done"
