#!/usr/bin/env bash
set -euo pipefail

: "${DATA:?Set DATA to the XLRS dataset path}"
: "${MODEL:?Set MODEL to the Qwen3-VL checkpoint path}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
SCORE_LAYER="${SCORE_LAYER:-18}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/q_flowcover_perturbation_optimized}"
if (($# == 0)); then
    set -- 50 60 70 80 85 90
fi

for percent in "$@"; do
    "$PYTHON_BIN" -u test_perturbation.py \
        --data "$DATA" --model "$MODEL" --split train --gpus "$GPU_IDS" \
        --prune_method q_flowcover_perturbation --selector_impl optimized \
        --greedy_backend reference --max_attention_mb 512 \
        --exec_mode full19_gather --decision_layer_human "$SCORE_LAYER" \
        --visual-prune-percent "$percent" --text_source baseline \
        --query_chunk_size 256 --group_chunk_size 32 --source_chunk_size 256 \
        --max-image-patches 1 --max-new-tokens 32 --warmup-cases 1 \
        --verify-candidates 0 --no-save-diagnostics \
        --output-dir "$EXPERIMENT_ROOT/L${SCORE_LAYER}/prune_${percent}"
done
