# Perturbation-guided local coverage

For the explicitly selected, equivalent high-throughput implementation and
real 8B/H200 timing measurements, see [optimization](perturbation_optimization.md).
Use `--selector_impl optimized`; the reference default remains available.

This independent experiment adds `q_flowcover_perturbation`; it does not alter
the archived `q_flowcover_direct_residual` implementation or its default
parameters. The existing source provenance is documented in
[SOURCE_MAP.md](../SOURCE_MAP.md).

## What the score measures

Let the text-scoring layer be human layer `l`. Its completed Full visual state
provides the existing, native-merger-aligned text-cos scores `r`. Each image view
has `N` native groups and the baseline budget `K`, including its original
rounding rule. Set

```text
p = softmax(r / tau)
P = original text-cos TopK(r, K)
pi[P] = softmax(r[P] / tau), pi[outside P] = 0
```

The default temperature is one, matching the original all-group text softmax.
`P` discovers sources; its members are not mandatory members of the final
selected set. The online algorithm does not use candidate-specific real model
forwards or an extra image/text encoder.

For each raw Query `a` in each Query group `i`, native layer `l+1` provides
post-RoPE Query/Key and Value tensors. Attention is normalized over the entire
legal source set of that same image view. For a source group `g`, define, per
head:

```text
o[a,h]   = sum_v A[a,h,v] V[v,h]
m[a,h,g] = sum_{v in g} A[a,h,v]
b[a,h,g] = sum_{v in g} A[a,h,v] V[v,h]
e[a,h,g] = (b[a,h,g] - m[a,h,g] o[a,h]) / (1 - m[a,h,g])
```

This is the exact difference between the Full and Full-minus-group attention
readouts, for fixed Queries and the same surviving Keys/Values. When the
remaining mass is numerically too small, use an explicit source-masked stable
recomputation; adding an arbitrary epsilon would change the score.

**The raw Queries must remain separate through this nonlinear division and
squared-norm operation.** Averaging four Query rows into a group first, then
applying `(b-m*o)/(1-m)`, generally produces a different answer. Group and Query
weights are applied only after each raw Query's deletion error is computed.
Likewise, heads are not averaged before evaluating their errors. GQA uses the
native Query-to-KV-head correspondence.

The default scalar `D[g]` is the `pi`-weighted Query-group average of the raw
Query squared error and head error, before the output projection:

```text
D[g] = sum_{i in P} pi[i] * mean_{a in raw(i), h} ||e[a,h,g]||_2^2
```

Consequently the squared-error denominator is `(1-m)^2`. This experiment's
squared, pre-output-projection readout score differs from CAOTE's original
unsquared single-token eviction cost. The optional explicit-deletion check
compares both the raw multi-head readout and its native output projection;
using the projection in that check does not imply it is part of the default
production `D` score.

It measures a **fixed-Query attention readout
error**, not a final VQA answer error or the full Transformer/MLP perturbation.
The corrected raw-Query group extension is inspired by CAOTE's eviction idea;
it is not a claim that the original CAOTE algorithm has been reproduced.
The [CAOTE paper, equations 8–19](https://arxiv.org/html/2504.14051v3)
uses a single-token L2 readout error. This experiment uses group deletion and
**squared** L2 error: the squared denominator is `(1-m)^2`, not `(1-m)`.
It does not introduce an approximate group-averaged attention denominator.

## Selection objective

Only sources outside the initial text Query set supply residual evidence:

```text
omega[g] = D[g] if g is outside P, otherwise 0
```

Group descriptors use the original native group-hidden ordering, followed by
L2 normalization. Sparse local affinity uses positive cosine only, with the
existing fixed local radius and same-image constraint:

```text
kappa[g,j] = max(0, cosine(z[g], z[j])) within local neighborhood
coverage[g] = max_{j in S} kappa[g,j]
gain[j] = p[j] + lambda * sum_g omega[g] * relu(kappa[g,j] - coverage[g])
```

Greedy selection starts with the empty set and updates coverage after every
choice until exactly `K` groups have been selected. No high-error protection
quota, query-mass threshold, centered affinity, adaptive radius, seeded swap,
token merge, or LLM recall is introduced. The no-local-coverage ablation uses
only self-affinity, i.e. `TopK(p + lambda * omega, K)`.

## Execution boundary and comparisons

The new online score reads native `l+1` Q/K/V. To avoid repeated QKV projections
and repeated visual prefixes, the Full `l+1` block runs once, then one actual
group gather shortens the subsequent suffix. The historical execution-mode
name `full19_gather` denotes this relative `l+1` boundary when the scoring layer
is changed; it is not an instruction to hard-code layers 18/19.

Text-cos and old direct-residual comparison modes retain their original
selector implementations. In particular, the old direct-residual method
obtains flow from layer `l`, while the new perturbation method uses `l+1`.
Report that source-layer difference when interpreting the A/B/C/D ablation.
The optional diagnostic flow comparison computed from the *new method's*
`l+1` Q/K is a separate, source-controlled counterfactual. Its mask is not an
archived baseline result and does not imply that a second VQA answer was
generated. The diagnostic field `flow_comparison_attention_layer` records the
actual source layer.

Native group expansion, original raw ordering/positions, per-view boundaries,
DeepStack, the merger, and LLM visual slots remain the adapter's responsibility.
The selector never renumbers groups according to importance.

## Optional diagnostic artifacts

`compression/perturbation_diagnostics.py` is a post-inference CPU renderer.
It consumes the exact prepared input image views and validated per-view grids;
it never guesses a square shape from the number of groups. Its PNG/NPZ/CSV/JSON
artifact rendering and saving follow the timed model call and are reported
separately as `diagnostics_wall_seconds`.

The optional source-controlled flow comparator runs a second greedy selection
inside the selector; that work remains included in `output.seconds` and its
own timing/FLOPs fields. Likewise, the selector's conversion of its Tensor
diagnostics to CPU lists is included in the measured model call and reported
as diagnostic serialization work. These costs must not be described as
excluded merely because the subsequent Pillow drawing is post-inference.
For production timing comparisons, use `--no-save-diagnostics` to disable the
extra comparator; the actual perturbation selector still runs in full.
`flow_attention_seconds` and `perturbation_seconds` label the same shared
attention scan, and QKV-acquisition time overlaps the native next block. Do not
add these overlapping entries as if they were independent phases.

Each sample contains one subdirectory per actual model input view:

```text
sample_<id>_<identity_hash>/
  summary.json
  view_000/
    input.png
    overview.png
    summary.json
    groups.csv
    arrays.npz
    subfig/
      original.png
      question_answer.png
      text_cos.png
      query_set.png
      deletion_perturbation.png
      flow_support.png
      text_keep.png
      flow_keep.png
      final_keep.png
      text_comparison.png
      flow_comparison.png
      analysis.png
```

Every panel is rendered and saved independently before composing the overview.
The exact model input aspect ratio is preserved. Rainbow heatmaps include their
numeric ranges; selection masks use a translucent white removed region.
Comparison maps use blue for baseline-only, orange for new-only, and teal for
shared selections. All array rows retain the original view-local group ID;
`group_row` and `group_col` explicitly store spatial mapping.

Diagnostics retain raw `D`, normalized `D / sum(D)` (zero when all errors are
zero), incoming support, text scores, Query membership, final masks, and
selection intersections/differences. D/support Spearman uses average ranks for
ties and is JSON `null` when a vector is constant. A zero/near-constant D map is
explicitly marked uninformative rather than inventing sensitive tokens by ID.
When `K=N`, no deletion statistics are needed and no native statistics are
captured. Those unmeasured values are exported as per-ID `NaN` in NPZ / `NA` in
CSV, with explicit unavailable panels and JSON `statistics_computed=false`;
they are never presented as measured zero errors.

The descriptive "high perturbation / low text" count uses each view's lower
60% text quantile and upper 10% perturbation quantile, outside Query set `P`.
Boundary ties are included. These thresholds are **analysis-only definitions**,
not selector quotas or extra mandatory keeps. The summary records actual
threshold values, retention/recovery counts, and the matching original IDs.

Single-layer inference cannot establish cross-layer persistence. Each summary
therefore records `cross_layer_persistence.measured = false`; a separate
multi-layer experiment is required to measure that property.

## Validation boundary

The implementation must compare closed-form readout errors with explicitly
source-masked deletion on small native attention tensors, including full-source
normalization, unequal head counts when applicable, Query chunk sizes, masks,
and the near-unit-mass path. Dense tiny references are tests only; the online
path stores bounded Query/group chunks, not a full `N x N` attention or
similarity matrix.

The seven CPU renderer tests in `tests/test_perturbation_diagnostics.py` verify
original grid IDs, exact K, invalid inputs, tie-aware Spearman, uninformative
scores, missing-flow honesty, and independently saved panels/unchanged arrays.
Passing these rendering tests does not constitute a real 8B-checkpoint VQA,
CUDA speed, or peak-memory benchmark. Actual run results belong in the final
validation report and must distinguish native tiny-checkpoint tests from
server-scale experiments.

## Server commands

Set `MODEL` to an existing local checkpoint directory before running. These
commands are reproducible entry points, not a claim that this environment has
completed an 8B GPU benchmark.

```bash
cd /public/home/20250075/sihang/prune-github
export DATA=/public/home/20250075/data/XLRS-Bench-lite-subdata-20
export MODEL=/public/home/20250075/sihang/prune/Qwen3-VL-8B

# One case, actual generation, native two-candidate consistency check,
# plus all CPU diagnostic panels and arrays.
python -u test_perturbation.py \
  --data "$DATA" --model "$MODEL" --split train --gpus 0 \
  --prune_method q_flowcover_perturbation \
  --exec_mode full19_gather --decision_layer_human 18 \
  --visual-prune-percent 70 --query_chunk_size 256 \
  --group_chunk_size 32 --source_chunk_size 256 \
  --max-image-patches 1 --max-new-tokens 32 --dry_run \
  --output-dir outputs/q_flowcover_perturbation/dry70
```

For matched-budget A/B/C/D throughput runs, disable the *extra diagnostic flow
selection and image serialization* explicitly. The new method's own deletion
statistics, greedy coverage, real group gather, and generation remain in the
measured inference path.

```bash
for method in text_cos q_flowcover_direct_residual \
              q_flowcover_perturbation q_flowcover_perturbation_nocover; do
  for removed in 50 70 85 90; do
    python -u test_perturbation.py \
      --data "$DATA" --model "$MODEL" --split train --gpus 0 \
      --prune_method "$method" --exec_mode full19_gather \
      --decision_layer_human 18 --visual-prune-percent "$removed" \
      --query_chunk_size 256 --group_chunk_size 32 --source_chunk_size 256 \
      --local_radius 1 --max-image-patches 1 --max-new-tokens 32 \
      --no-save-diagnostics --verify-candidates 0 \
      --output-dir "outputs/perturbation_ablation/${method}/prune_${removed}" || break 2
  done
done
```

`removed=70` means 70% removed, approximately 30% retained under the original
per-view rounding rule. Compare actual reported `N`, `K`, and `K/N`, not only
the directory label. A/B keep their original scoring and source layers even
when the suffix gather boundary is matched. C/D use raw deletion errors with
`lambda=1` by default; raw-error and text-probability scales may differ, so the
recorded lambda is part of the method configuration rather than an implied
unit-free or calibrated uncertainty weight.

## Summarize already measured cross-layer runs

Do not infer persistence from the single default layer. After multiple score
layers have actually been measured with matching input/settings, the optional
read-only summary command is:

```bash
python scripts/summarize_perturbation.py \
  --input-roots outputs/perturbation_layer12 outputs/perturbation_layer15 \
                outputs/perturbation_layer18 outputs/perturbation_layer20 \
  --scope outside_query --top-fraction 0.1 \
  --output-dir outputs/perturbation_cross_layer_summary
```

It joins the actual `Case.id` (not directory indices), view index, matching
recorded model/data/budget profile, exact saved input-image SHA256, native
group grid and coordinates. Missing/full-budget statistics, mismatched inputs,
and conflicting repeats are excluded with reasons. The default scope is each
layer's own sources outside its independently computed Query set `P`; it is
not a cross-layer fixed candidate pool. Spearman uses only original IDs
eligible in both compared layers, with tie-averaged ranks. Top-fraction sets
are selected independently within each layer's eligible set, including ties.
The overlap coefficient is intersection size divided by the smaller Top-set
size; Jaccard is reported separately.

Outputs are `summary.json`, `layer_pairs.csv`, and `rankings.csv`. Single-layer,
insufficient-overlap, or uninformative-score inputs retain
`cross_layer_persistence_measured=false`. Noneligible positions are omitted
from ranking rows rather than imputed as zero error. These descriptive
statistics do not establish final-answer causal importance or a significance
test. The five summary tests exercise data identity, true original IDs,
single-layer honesty, inverted ranks, and tied/conflicting input handling.
