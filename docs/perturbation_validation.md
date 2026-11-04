# Implementation and executed validation

This page records the initial CPU implementation validation. Subsequent real
Qwen3-VL-8B/H200 validation and the equivalent statistics optimization are
documented in [perturbation_optimization.md](perturbation_optimization.md).

This is an independent experiment in `prune-github`. Original archived
results have not been reused as new measurements.

## Files and explicit execution paths

New algorithm and runtime:

- `test_perturbation.py`: independent A/B/C/D CLI, configurable human score
  layer, explicit `full19_gather` boundary, rejects unknown modes, and exits
  unsuccessfully if any evaluated case fails.
- `compression/q_flowcover_perturbation.py`: streamed native-Q/K/V deletion
  statistics, sparse local affinity, exact empty-set greedy coverage.
- `compression/qwen3_vl_perturbation.py`: read-only next-block QKV capture,
  real full `l+1` block, one gather, compact suffix, existing native
  merger/DeepStack/LLM visual-cache integration.
- `compression/perturbation_worker.py`: persistent model worker, warmup,
  actual evaluation protocol, first-sample numerical check, plotting after
  inference, common executed-shape visual FLOPs convention for all methods.
- `compression/perturbation_diagnostics.py`: original-ID spatial maps, exact
  selection masks, comparisons, NPZ/CSV/JSON, and recovery/correlation stats.
- `scripts/run_perturbation_ablation.sh`: one checked dry-run followed by
  four methods at 50/70/85/90 percent removal.
- `scripts/smoke_perturbation.py`: reproducible CPU native tiny-checkpoint
  integration test with actual local processor and one real XLRS sample.
- `scripts/benchmark_perturbation.py`: same-input, warmed-up CPU inference
  comparison with the actual shipped chunk defaults.
- `scripts/summarize_perturbation.py`: read-only multi-layer ranking analysis,
  strict input/group identity checks, no imputation for missing candidates.
- `tests/test_perturbation*.py`, `tests/test_summarize_perturbation.py`:
  mathematical, native runtime, CLI, serialization, and summary checks.
- `docs/perturbation.md`: complete mathematical and execution definitions.

Existing files changed are `evaluator.py` (explicit new-worker dispatch and
entry/settings-isolated resume), and `README.md` (a documentation link).
There is no runtime replacement of an old selector. Existing `testn.py`,
`compression/q_flowcover.py`, `compression/q_flowcover_variants.py`,
`compression/qwen3_vl_modes.py`, `compression/worker.py`, and both original
model files were verified byte-for-byte equal to repository HEAD.

```text
old testn.py
  -> evaluator -> original compression.worker
  -> original Qwen3VLCompressionRunner -> original text-cos/direct-residual

new test_perturbation.py
  -> evaluator (evaluation_variant=perturbation)
  -> perturbation_worker
     A/B -> original Qwen3VLCompressionRunner -> original selector
     C/D -> Qwen3VLPerturbationRunner
         -> completed H_l / original aligned text-cos / original K rounding
         -> real native full block l+1, scoped read-only qkv hooks
         -> independent perturbation selector
         -> one complete-group raw gather
         -> native compact suffix, merger/DeepStack and shared LLM generation
```

The name `full19_gather` is historical. For score layer `l=15` the statistics
and gather boundary are layer 16; for `l=18` they are layer 19. No layer 19
boundary is hard-coded. Queries `P` are not forced into final selection.

## Executed correctness checks

Environment: Python 3.10.22, torch 2.6.0+cu126, Transformers 4.57.6; CUDA was
unavailable. The complete test suite passed **115 tests** in about 40 seconds:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

This includes mathematical dense references, every-round greedy gain checks,
raw-query/head aggregation, GQA/masks, near-unit mass, unchanged source arrays,
complete group indices, R=0, lambda=0, full budget, multiple image views,
different score layers, real shorter next-block inputs, repeatability, native
context reconstruction, and exception-safe hook removal.

The actual native attention projection **input** was compared with context
reconstructed from captured Q/K/V, native RoPE, native scale, and per-image
boundaries. This checks the native tensor convention independently of the
deletion formula.

Synthetic mathematical reference (12 groups, 4 raw/group, GQA Q-heads=4,
KV-heads=2, head dimension=8):

- Maximum raw readout difference between closed form and independent explicit
  source deletion: **3.5763e-7**.
- Maximum aggregated D difference across four chunk configurations:
  **2.9802e-8**.
- Support difference at most **1.4902e-8**.

The native tiny-model smoke additionally checked two candidates using **all
production Queries and pi**, physically removing their K/V rows and independently
softmax-normalizing the survivors. Native readout error was at most
**6.0660e-8** and projected-difference error at most **3.3179e-9**. The two
production/explicit D values differed by at most **2.7285e-12**.

Raw report: `outputs/perturbation_validation/math_report.json`.

## Executed end-to-end test

Ran one actual XLRS sample for each A/B/C/D method, using the actual local
8B-checkpoint tokenizer/image-processor files and a **tiny randomly initialized
native Qwen3-VL checkpoint**, on CPU. This did **not** load real 8B weights.
The evaluator loaded data, built the canonical question/options prompt,
computed selections, compacted LLM slots/position IDs, generated two tokens,
parsed the answer, and saved JSONL/Markdown. There were zero runtime errors.
Random-model answers are not benchmark accuracy evidence.

All four methods retained exactly 19/64 groups at requested 70% removal
(`round(64*0.3)=19`), with one group equal to four raw tokens. The actual
20-block visual trace was:

```text
blocks 1–19: 256 raw tokens each
block 20:    76 raw tokens
final LLM visual slots: 19
```

The new branch captured the real next-block QKV projection once, projected
it zero additional times, and gathered once. Its diagnostic overview,
independent panels and raw scalar arrays were saved successfully.

Artifacts: `outputs/perturbation_validation/cli_smoke/report.json` and the
four timestamped method subdirectories beneath `cli_smoke/`. The initial
smoke requested a 4096-pixel cap, but the existing public processor enforces
its inherited 65,536-pixel minimum; the observed grid was `[1,16,16]` with
patch size 16. The reproducible smoke script now explicitly requests 65,536
to describe that actual input honestly. No original processor behavior was
changed.

## Executed timings and their limits

Selector-only CPU microbenchmark, synthetic N=256/K=64, unit=4, H=4, d=16,
one CPU thread, two warmups and five measurements:

| Method | Median selector time |
| --- | ---: |
| Text-cos | 0.0698 ms |
| Original direct-residual flow | 7.752 ms |
| Perturbation coverage | 44.752 ms |

Separately, the same real input/tiny checkpoint was measured with one warmup
and five repetitions, no plotting/comparison/explicit verification, using
stress chunks Query=4/group=2/source=16:

| Method | Median inference | Median selector |
| --- | ---: | ---: |
| A text-cos | 33.15 ms | 0.242 ms |
| B original direct-residual | 37.49 ms | 2.433 ms |
| C perturbation + coverage | 124.07 ms | 88.42 ms |
| D perturbation without local coverage | 121.92 ms | 87.14 ms |

These deliberately small chunks exercise the streaming paths and increase
Python/kernel scheduling overhead. They are not the shipping defaults or a
representative GPU benchmark. Raw samples and exact scope are saved in
`outputs/perturbation_validation/cpu_inference_comparison.json`; they must not
be presented as an H200 speed comparison or as a speedup.

The required efficiency follow-up used the **shipping default chunks
256/32/256**, with the same CPU, weights, sample, warmup and repetition counts:

| Method | Median inference | Median selector |
| --- | ---: | ---: |
| A text-cos | 35.20 ms | 0.270 ms |
| B original direct-residual | 36.95 ms | 1.914 ms |
| C perturbation + coverage | 39.44 ms | 3.600 ms |
| D perturbation without local coverage | 39.40 ms | 3.464 ms |

This confirms the small-chunk stress overhead must not be mistaken for the
default implementation's cost. It remains a tiny **CPU** comparison, with
no implication for real 8B GPU performance. Each method's five repeated
selections and token traces were deterministic. Reproduce with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_perturbation.py \
  --model outputs/perturbation_validation/cli_smoke/tiny_random_checkpoint \
  --data outputs/perturbation_validation/cli_smoke/real_sample.jsonl \
  --warmup 1 --repeats 5 --output-dir outputs/perturbation_validation
```

Raw report: `outputs/perturbation_validation/cpu_default_chunk_comparison.json`.

GPU inference totals include selection, required synchronization, actual
gather and generation. Native QKV projection overlaps the real visual block;
extra RoPE decoding overlaps selector timing. Shared flow/perturbation scan
is one stage, not two independently summed stages. Diagnostic flow's extra
greedy selection and in-selector CPU serialization are measured if enabled.
External panel/NPZ/CSV saving is reported separately. Production comparisons
disable diagnostic comparison and explicit verification.

FLOPs are **partial analytical executed-shape estimates**, not a hardware
profiler measurement. The new entry aligns the visual shape convention for
A/B/C/D. Additional QK/AV, statistics, affinity and greedy work are included;
patch convolution, normalization/RoPE/activation, some reduction/indexing,
memory traffic and launches remain incompletely counted. The old entry's
archived estimates are untouched. Native prefill-hook timing is labelled as
a host estimate; synchronized full generation wall time is separately saved.

## Unmeasured results and research risks

- At the initial CPU-validation stage, real Qwen3-VL-8B GPU accuracy,
  throughput and peak-memory measurements had **not** run. The subsequent
  real H200 paired comparison and complete 260-case optimized 50% run are
  documented in [perturbation_optimization.md](perturbation_optimization.md).
  The complete 16-combination A/B/C/D budget sweep remains unmeasured.
- Raw D has a different numeric scale from probability mass p. Lambda=1 is
  explicit, and D is not secretly normalized into the selection objective.
  Its V-scale/sample dependence can make coverage weak or dominant. Tiny
  random-weight outputs do not establish better question answering.
- Error is fixed-Query attention readout error before W_O; optional W_O
  checks validate the difference tensor, but production D is not the full
  block/MLP/final-answer error. This is a verified group extension of an
  idea, not a reproduction of original single-token CAOTE.
- A/B keep their original layer-l flow tensors; C/D use layer-l+1 tensors.
  The optional same-l+1 flow mask controls this difference but does not
  generate an old-B VQA answer. Report this source-layer distinction.
- Single-token/group errors are not additive simultaneous-deletion losses.
  Greedy local coverage is a surrogate; no global objective optimum is
  claimed. Exact tied baseline top-K boundaries may select different IDs
  than smallest-ID greedy ties; do not promise cross-device bitwise ties.
- Single-layer inference cannot measure persistence. The read-only cross-layer
  summary was tested on existing single-layer smoke outputs and correctly
  reported it **unmeasured**. No new multi-layer model sweep was run.
- Runtime adapter targets still-image native Qwen3-VL vision MHA. It rejects
  unknown QKV layouts and videos; the generic statistics function separately
  tests GQA/MQA mapping. It never invents approximate native tensors.
