#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 DATA_DIRECTORY LOCAL_MODEL_DIRECTORY [GPU_IDS=0] [OUTPUT_ROOT=outputs/q_flowcover_direct_residual_layers]" >&2
  exit 2
fi

dataset_path="$1"
model_path="$2"
gpu_ids="${3:-0}"
output_root="${4:-outputs/q_flowcover_direct_residual_layers}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The existing layer-18 experiments are retained; this continuation runs 15–17.
# An explicitly empty QFLOW_SKIP_LAYERS includes all four layers again.
read -r -a skip_layers <<< "${QFLOW_SKIP_LAYERS-18}"

# One independent pruning decision after each human-numbered visual block.
for decision_layer in 15 16 17 18; do
  skip=false
  for layer in "${skip_layers[@]}"; do
    if [[ "$layer" == "$decision_layer" ]]; then skip=true; fi
  done
  if [[ "$skip" == true ]]; then
    echo "[SKIP LAYER] layer=$decision_layer (QFLOW_SKIP_LAYERS; existing results retained)"
    continue
  fi
  QFLOW_SUMMARY_ROOT="$output_root" QFLOW_SUMMARY_LAYERS="15 16 17 18" \
    bash "$script_dir/run_pruning_sweep.sh" \
    "$dataset_path" \
    "$model_path" \
    "$gpu_ids" \
    "$output_root/layer_${decision_layer}" \
    "$decision_layer"
done
