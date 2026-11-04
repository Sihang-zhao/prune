#!/usr/bin/env bash
set -euo pipefail

: "${DATA:?Set DATA to the XLRS dataset path}"
: "${MODEL:?Set MODEL to the Qwen3-VL checkpoint path}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/perturbation_ablation}"
SCORE_LAYER="${SCORE_LAYER:-18}"
GPU_IDS="${GPU_IDS:-0}"
common=(--data "$DATA" --model "$MODEL" --split train --gpus "$GPU_IDS"
        --exec-mode full19_gather --decision-layer-human "$SCORE_LAYER"
        --text-source baseline --max-image-patches 1 --max-new-tokens 32
        --max-image-pixels "${MAX_IMAGE_PIXELS:-4194304}"
        --query-chunk-size 256 --group-chunk-size 32 --source-chunk-size 256
        --local-radius 1 --seed 0 --warmup-cases 1)

# One case with explicit native deletion verification and spatial diagnostics.
"$PYTHON_BIN" -u test_perturbation.py "${common[@]}" \
    --prune-method q_flowcover_perturbation --visual-prune-percent 70 \
    --dry-run --output-dir "$EXPERIMENT_ROOT/L${SCORE_LAYER}/dry_run_70"

# Timed comparisons exclude explicit verification and diagnostic-only flow
# selection/rendering. All four use the same full-(l+1) gather boundary.
for method in text_cos q_flowcover_direct_residual q_flowcover_perturbation q_flowcover_perturbation_nocover; do
    for percent in 50 70 85 90; do
        "$PYTHON_BIN" -u test_perturbation.py "${common[@]}" \
            --prune-method "$method" --visual-prune-percent "$percent" \
            --verify-candidates 0 --no-save-diagnostics \
            --output-dir "$EXPERIMENT_ROOT/L${SCORE_LAYER}/${method}/prune_${percent}"
    done
done
