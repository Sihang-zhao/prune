#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 DATA_DIRECTORY LOCAL_MODEL_DIRECTORY [GPU_IDS=0] [OUTPUT_ROOT=outputs/q_flowcover_direct_residual]" >&2
  exit 2
fi

dataset_path="$1"
model_path="$2"
gpu_ids="${3:-0}"
output_root="${4:-outputs/q_flowcover_direct_residual}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${QFLOW_PYTHON:-python}"

for prune_percent in 50 60 70 80 85 90 95; do
  "$python_bin" "$project_root/testn.py" \
    --data "$dataset_path" \
    --model "$model_path" \
    --offline \
    --gpus "$gpu_ids" \
    --dtype bf16 \
    --prune-method q_flowcover_direct_residual \
    --exec-mode hard18 \
    --visual-prune-percent "$prune_percent" \
    --local-radius 1 \
    --affinity-mode raw \
    --qflow-query-chunk 256 \
    --max-image-pixels 4194304 \
    --max-image-patches 1 \
    --max-new-tokens 32 \
    --worker-cases 128 \
    --text-source baseline \
    --output-dir "$output_root/prune_${prune_percent}"
done
