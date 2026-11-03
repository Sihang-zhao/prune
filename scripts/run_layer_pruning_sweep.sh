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

# One independent pruning decision after each human-numbered visual block.
for decision_layer in 15 16 17 18; do
  bash "$script_dir/run_pruning_sweep.sh" \
    "$dataset_path" \
    "$model_path" \
    "$gpu_ids" \
    "$output_root/layer_${decision_layer}" \
    "$decision_layer"
done
