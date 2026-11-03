# Source mapping and archived experiment provenance

This extraction follows the experiment stored under `prune/outputs/q_flowcover_direct_residual`. All original paths below are relative to the parent workspace's `prune/` directory; line numbers refer to the source inspected on 2026-11-03.

## Original execution path

| Original file | Role |
|---|---|
| `testn.py:115–148` | Parse the direct-variant method/settings, set the layer-18 mode, install the compression worker, and run the canonical evaluator |
| `testx.py:24–35` | Dynamically load `testp.py` for shared evaluation infrastructure |
| `testp.py:599–608` | Spawn inference workers and shard cases |
| `compression/testx_worker.py:197–226` | Build `Qwen3VLCompressionRunner` with the recorded selector settings |
| `compression/qwen3_vl_modes.py:531–550` | Dispatch `q_flowcover_direct_*` to the variant selector |
| `compression/q_flowcover_variants.py:324` | Public per-view variant-selector entry point |
| `compression/q_flowcover_variants.py:215` | One-view `q_flowcover_direct_residual` execution |
| `compression/q_flowcover_variants.py:154–200` | Sparse local greedy direct-text plus residual-flow coverage |
| `compression/q_flowcover.py` | Shared native attention support, cosine scoring, and ordered Top-K helpers |

## Exact method identity

The target method is **`q_flowcover_direct_residual`**, selected by the original `testn.py`. It is distinct from the later `direct_residual` entry point in `testww.py` and `pruning_direct_residual/`; those files are not the source of these archived runs.

For the target branch, all-group text scores are softmax-normalized to `text_mass`. The query set is the text Top-K set. Native layer-18 RoPE Q/K provide query-to-all attention support, with the support inside the query set set to zero. Greedy selection adds the candidate's direct text mass to its residual-flow-weighted local coverage gain. It retains exactly K groups in original order. This is not the ordinary Q-FlowCover equal-weight blend objective.

The archived active settings are `hard18`, layer 18, query chunk 256, local radius 1, `raw` affinity, and `baseline` scoring text. The recorded `query_mass=0.9` does not select a cumulative-mass query set for this branch; query groups are Top-K. Recorded candidate chunk 64 and adaptive-radius maximum 3 likewise do not change this branch.

## Standalone release layout

```text
testn.py
  → evaluator.py
  → compression/worker.py
  → compression/qwen3_vl_modes.py
  → compression/q_flowcover_variants.py
      ↳ shared helpers in compression/q_flowcover.py
```

The cleaned entry point and worker are imported directly, without dynamically loading the original `testp.py` or replacing its worker. The model/data/scoring support is packaged locally. The two algorithm files remain byte-for-byte copies of the inspected original sources; unrelated experiment entry points and large outputs are omitted.

| Unchanged algorithm file | SHA256 |
|---|---|
| `compression/q_flowcover_variants.py` | `23b715c5ee755430615ceb9c6b699e6768f367c43f4d8db11cd7cb86dbe6be77` |
| `compression/q_flowcover.py` | `a6e4c02d0dfc6c8a62f646a2375220364256d56bc8f85929f0c0f38803ae6d2d` |

## Archived results and reproducibility boundary

All seven runs completed on the **260-case `XLRS-Bench-lite-subdata-20` subset**, with 13 tasks and 20 cases per task, not the full XLRS-Bench-lite dataset. All summaries report zero runtime errors. The archived model was Qwen3-VL-8B, actual dtype bf16, one CUDA worker device, one image view, a 4,194,304-pixel budget, and 32 maximum generated tokens.

- [reference_metrics.csv](results/reference_metrics.csv): compact metrics for 50/60/70/80/85/90/95% pruning.
- [reference_runs.json](results/reference_runs.json): portable parameters, method notes, run paths, and selected metrics.
- [reference_summaries/](results/reference_summaries/): original summary JSON files copied byte-for-byte; these contain no machine-specific absolute paths.
- [source_manifest.json](source_manifest.json): current source hashes and original configuration/summary hashes, with relative source paths.

The historical records store `commit="unknown"` and include no historical source snapshot. Configurations, logs, and result method metadata identify the method and settings, while the source hashes identify the **current inspected implementation used for this extraction**. They do not establish a historical commit-level source archive. Archived latency/FLOPs measurements are supplied as recorded; FLOPs are estimates and the GPU hardware model was not recorded.

No per-sample result JSONL, images, datasets, model weights, or runtime logs are included in this release.
