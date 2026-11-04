# Equivalent perturbation-statistics optimization

The original `q_flowcover_perturbation` implementation remains the default
`--selector_impl reference`. The explicit `--selector_impl optimized` path
keeps the same original image, native Q/K/V, all K Query groups, per-view
source boundaries, score layer, raw deletion-error objective and sequential
greedy. It does not cap Queries, lower image resolution, remove heads, or
replace the objective with static score Top-K.

## Actual bottleneck

The existing 50% run (`outputs/q_flowcover_perturbation/L18/prune_50/1104_1136`)
contained 23 completed cases. Mean model inference was 2.750 s: selector
2.306 s, including perturbation 2.007 s and greedy 0.279 s. Serialization was
only about 1 ms. Most inputs had N=4096 groups, K=2048, 16384 source raw
tokens and 8192 raw Queries, with 16 heads of dimension 72.

With Query/source/group chunks 256/256/32, the old statistics traversed 2048
source chunks followed by 4096 group chunks. These required roughly 12,288
small QK/AV calls plus many elementwise kernels, and constructed wide local
group readout/error tensors. Saving less JSON does not remove this cost.

## Same deletion error, a different contraction

For one head and raw Query, with source group attention vector `a_g`, group
Value matrix `V_g`, total group mass `m` and Full readout `o`:

```text
b = a_g V_g
G_g = V_g V_g^T
||b - m o||² = a_g G_g a_g^T
               - 2m sum_{v in g} a_v (o dot V_v)
               + m² ||o||²
D = ||b - m o||² / (1-m)²
```

The group Gram matrix has native-unit by native-unit size (4x4 for this
checkpoint). The new path computes one rectangular QK and one complete-source
softmax per bounded Query chunk; AV and oV are matrix operations. All groups
are reduced together. It retains no `[heads,all_queries,all_groups,head_dim]`
tensor and no full N-by-N attention/similarity matrix.

Statistics stay FP32 and native forward stays bf16. A common per-head Value
translation improves conditioning without changing `b-m*o`. Cancellation
flags route suspicious groups to the direct vector formula. Near-unit mass
uses stable explicit source-masked recomputation. No arbitrary epsilon or
score renormalization changes the objective. Triggered recovery is timed and
included in the operation estimate.

`--max_attention_mb 512` bounds one FP32 rectangular attention allocation,
not the sum of every temporary. It adjusts Query chunk size only. The
actual effective chunk, shape and recovery IDs are recorded. Group/source
chunk options remain relevant to numerical recovery paths.

## GPU greedy choice

CPU tests found local refresh cheaper arithmetically. On the actual H200,
however, its additional ATen launches took about 0.42 s versus the original
greedy's 0.28 s. Therefore the optimized statistics path defaults to the
**original exact greedy**, selected explicitly as `--greedy_backend reference`.
This is a shared greedy primitive, not fallback to another selector or to
the old statistics. `local_refresh` and `scripted_local_refresh` remain
explicit experimental alternatives. No unvalidated Triton/CUDA-graph path is
silently enabled.

## Executed real-checkpoint paired validation

GPU: H200 NVL on gpu2. Runtime: zsh Python 3.11 / torch 2.14.1+cu126,
Transformers 4.57.6. Native model dtype bf16; selector statistics FP32;
TF32 was not enabled. Existing other GPU processes were left running.

Loaded Qwen3-VL-8B once. After one warmup per implementation, 13 real XLRS
cases were sampled uniformly across the ordered 260-case dataset, without
attention/perturbation preselection. Each case ran reference and optimized
with identical preprocessing, prompt, 50% budget, human score layer 18,
full layer 19 before gathering, maximum 32 new tokens, and greedy primitive.
Image saving and explicit deletion debug verification were disabled.

| Paired 13-case mean | Reference | Optimized |
| --- | ---: | ---: |
| Model inference, including selector/gather/LLM | 2.5132 s | **0.9200 s** |
| Perturbation statistics | 1.8292 s | **0.2715 s** |
| Greedy | 0.2643 s | 0.2656 s |
| Maximum allocated GPU memory | 17.22 GiB | 17.92 GiB |

All 13 raw keep-index hashes and final generated answers matched exactly.
The maximum D difference was 1.04e-6 from FP32 contraction/reduction order;
the original objective and budget were unchanged. This is measured subset
equivalence, not a bitwise claim for every possible floating-point input.
These are model inference times (the evaluator's existing scope), excluding
image I/O and processor preparation. Whole-test wall time is separate.
Individual cases can exceed 1.2 s; the target concerns the average.

Report: `outputs/perturbation_optimization/real_e2e_13/report.json`.
The initial three-case local-refresh comparison is preserved separately under
`real_e2e_local/`; the native-size synthetic benchmark is under
`gpu_synthetic/` and is not presented as an end-to-end result.

The complete CPU test suite passed 155 tests after integration. The new
statistics tests include GQA/MQA, masks, half input, arbitrary native group
sizes, explicit deletion, numerical recovery, memory bounds and final greedy
equivalence. Existing baseline and unrelated lite selectors remain intact.

The 13-case optimized maximum was 1.618 s; its mean was 0.920 s. This is an
average-latency target, not a per-case deadline.

## Completed full 260-case, 50% validation

The optimized entry subsequently completed every case in the same
XLRS-Bench-lite-subdata-20 dataset on the real H200/Qwen3-VL-8B, using the
command below and an independent output directory:

| Full-dataset measurement | Result |
| --- | ---: |
| Completed / runtime errors | **260 / 0** |
| Average model inference, same original `testtime` scope | **0.9944 s** |
| Median model inference | 0.9060 s |
| Average visual encoding and pruning (`pruning_latency`) | **0.8524 s** |
| Average selector | 0.5858 s |
| Average perturbation statistics | 0.3133 s |
| Average greedy | 0.2594 s |
| P95 model inference | 1.7840 s |
| Maximum allocated GPU memory | 17.93 GiB |
| Correct / total (no claim of improvement over full reference) | 133 / 260 |
| Entire test wall time, including loading/preparation/I/O | 440.58 s |

The average model inference meets the 1.2-second target. It must not be
confused with total wall time divided by case count or a per-case deadline.
GPU sharing and differing questions/input sizes contribute to variability.

Every record confirmed `query_group_count=K` and
`source_query_count=K*merge_unit`: the speedup did not come from Query
sampling. All 23 existing successful reference-run cases were also joined
by their original dataset index and compared against the new full run;
their selected indices and raw generated answers all matched. Together with
the separately paired 13-case experiment this provides direct equivalence
evidence; it is not a claim that every one of the 260 cases was rerun with
both implementations.

Result directory:
`outputs/q_flowcover_perturbation_optimized/L18/prune_50/1104_1223/`
contains `config.json`, `results.jsonl`, `results.md`, `summary.json`, and
`test.log`. Full scalar validation and old-run comparisons are saved in
`outputs/perturbation_optimization/full_260_validation.json`.

The original pinned torch 2.6 CPU environment also passed all 22 focused
optimized statistics/greedy/integration tests; deployment GPU measurements
above used the existing zsh torch 2.14 CUDA environment.

## Run the optimized method

```bash
cd /public/home/20250075/sihang/prune-github
export DATA=/public/home/20250075/data/XLRS-Bench-lite-subdata-20
export MODEL=/public/home/20250075/sihang/prune/Qwen3-VL-8B

for p in 50 60 70 80 85 90; do
  python -u test_perturbation.py \
    --data "$DATA" --model "$MODEL" --split train --gpus 0 \
    --prune_method q_flowcover_perturbation --selector_impl optimized \
    --greedy_backend reference --max_attention_mb 512 \
    --exec_mode full19_gather --decision_layer_human 18 \
    --visual-prune-percent "$p" --text_source baseline \
    --query_chunk_size 256 --group_chunk_size 32 --source_chunk_size 256 \
    --max-image-patches 1 --max-new-tokens 32 --warmup-cases 1 \
    --verify-candidates 0 --no-save-diagnostics \
    --output-dir "outputs/q_flowcover_perturbation_optimized/L18/prune_${p}" \
    || break
done
```

The same loop is saved in `scripts/run_perturbation_optimized_50_90.sh`.
After exporting DATA/MODEL, run it without arguments for all six budgets,
or pass selected percentages, e.g. `bash scripts/run_perturbation_optimized_50_90.sh 60 70 80 85 90`.

Without `--selector_impl optimized`, the original reference statistics still
run. Different implementations/backend/memory settings cannot silently share
a resumed result file. The full-source partition and K are preserved when
chunks change.

Reproduce the paired validation:

```bash
python scripts/benchmark_perturbation_e2e.py \
  --data "$DATA" --model "$MODEL" --device cuda:0 \
  --n 13 --case-selection uniform --prune-percent 50 \
  --score-layer 18 --greedy-backend reference \
  --output-dir outputs/perturbation_optimization/real_e2e_repeat
```

## Source changes

New independent code:

- `compression/perturbation_statistics_optimized.py`
- `compression/perturbation_greedy_optimized.py`
- `compression/q_flowcover_perturbation_optimized.py`
- `scripts/profile_perturbation_selector.py`
- `scripts/benchmark_perturbation_e2e.py`
- `tests/test_perturbation_statistics_optimized.py`
- `tests/test_perturbation_greedy_optimized.py`
- `tests/test_perturbation_optimized.py`

The independent perturbation entry/worker/adapter add explicit implementation
settings and dispatch; evaluator resume checks include them. The original
`compression/q_flowcover_perturbation.py` is retained. Unrelated pre-existing
working-tree changes are not part of this optimization.
