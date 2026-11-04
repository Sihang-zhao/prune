#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 DATA_DIRECTORY LOCAL_MODEL_DIRECTORY [GPU_IDS=0] [OUTPUT_ROOT=outputs/q_flowcover_direct_residual] [DECISION_LAYER=18]" >&2
  exit 2
fi

dataset_path="$1"
model_path="$2"
gpu_ids="${3:-0}"
output_root="${4:-outputs/q_flowcover_direct_residual}"
decision_layer="${5:-18}"
# Layers are human-numbered (1-based): prune after the selected visual block.
case "$decision_layer" in
  15|16|17|18) ;;
  *)
    echo "DECISION_LAYER must be 15, 16, 17, or 18 (1-based)." >&2
    exit 2
    ;;
esac
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${QFLOW_PYTHON:-python}"
summary_root="${QFLOW_SUMMARY_ROOT:-$output_root}"
read -r -a summary_layers <<< "${QFLOW_SUMMARY_LAYERS:-$decision_layer}"

update_summary() {
  if ! "$python_bin" "$project_root/scripts/summarize_results.py" \
    --input-root "$summary_root" \
    --output-prefix "$summary_root/summary" \
    --layers "${summary_layers[@]}" \
    --prune-percents 50 60 70 80 85 90 95 98 \
    --method q_flowcover_direct_residual \
    --data "$dataset_path" \
    --model "$model_path" \
    --exec-mode hard18 \
    --dtype bf16 \
    --preserve-existing; then
    echo "Summary update failed; run scripts/summarize_results.py again after checking its message. Benchmark results remain saved." >&2
  fi
}

# Create pending rows up front and refresh after every completed experiment.
update_summary

for prune_percent in 50 60 70 80 85 90 95 98; do
  test_arguments=(
    --data "$dataset_path" \
    --model "$model_path" \
    --offline \
    --gpus "$gpu_ids" \
    --dtype bf16 \
    --prune-method q_flowcover_direct_residual \
    --exec-mode hard18 \
    --decision-layer-human "$decision_layer" \
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
  )
  plan="$("$python_bin" "$project_root/scripts/sweep_state.py" "${test_arguments[@]}")"
  IFS=$'\t' read -r action source <<< "$plan"
  case "$action" in
    skip)
      echo "[SKIP] layer=$decision_layer prune=$prune_percent% already completed: $source"
      continue
      ;;
    resume)
      echo "[RESUME] layer=$decision_layer prune=$prune_percent% checkpoint: $source"
      test_arguments+=(--resume-from "$source")
      ;;
    run)
      echo "[RUN] layer=$decision_layer prune=$prune_percent%"
      ;;
    *)
      echo "Unexpected sweep state: $plan" >&2
      exit 1
      ;;
  esac
  "$python_bin" "$project_root/testn.py" "${test_arguments[@]}"
  update_summary
done
