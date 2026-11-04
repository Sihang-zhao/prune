#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 DATA MODEL [GPU_IDS=0] [OUTPUT_ROOT=outputs/perturbation] [LAYER=18]" >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data="$1"
model="$2"
gpu_ids="${3:-0}"
output_root="${4:-outputs/perturbation}"
layer="${5:-18}"
for percent in 50 60 70 80 85 90 95 98; do
  "$python_bin" "$project_root/evaluate.py" \
    --data "$data" --model "$model" --offline --gpus "$gpu_ids" --dtype bf16 \
    --method perturbation --implementation optimized --layer "$layer" \
    --prune-percent "$percent" --resume \
    --output-dir "$output_root/L${layer}/prune_${percent}"
  "$python_bin" "$project_root/scripts/summarize.py" \
    --input-root "$output_root" --output-prefix "$output_root/summary"
done
