#!/usr/bin/env python3
"""Matched native-sized synthetic selector benchmark; never a VQA quality result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--groups", type=int, default=4096)
    p.add_argument("--keep", type=int, default=2048)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=72)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output-dir", default="outputs/perturbation_optimization")
    a = p.parse_args()
    import torch
    from compression.q_flowcover_perturbation import select_q_flowcover_perturbation_views
    from compression.q_flowcover_perturbation_optimized import select_q_flowcover_perturbation_optimized_views

    if a.groups % a.width or not 0 < a.keep <= a.groups:
        p.error("grid and K are invalid")
    torch.manual_seed(41)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    n, h, d = a.groups, a.heads, a.head_dim
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    q, k, v = [torch.randn(4 * n, h, d, device=device, dtype=dtype) for _ in range(3)]
    hidden = torch.randn(4 * n, h * d, device=device, dtype=dtype)
    projected = torch.randn(n, 128, device=device, dtype=dtype)
    text = torch.randn(16, 128, device=device, dtype=dtype)
    inputs = ([projected], text, [q], [k], [v], [hidden], [a.keep], [4 * n], 4, "q_flowcover_perturbation", [(n // a.width, a.width)])
    options = dict(scale=d ** -.5, query_chunk_size=256, group_chunk_size=32, source_chunk_size=256, compare_flow=False)
    rows, values = [], {}
    for impl in ("reference", "optimized_reference", "optimized_local_refresh", "optimized_scripted_local_refresh"):
        fn = select_q_flowcover_perturbation_views if impl == "reference" else select_q_flowcover_perturbation_optimized_views
        kwargs = dict(options)
        if impl != "reference":
            kwargs.update(greedy_backend=impl.removeprefix("optimized_"))
        for _ in range(a.warmup):
            fn(*inputs, **kwargs)
        samples = []
        for _ in range(a.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            selections, records = fn(*inputs, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append({"wall_seconds": time.perf_counter() - start, "stage_seconds": {key: val for key, val in records[0].items() if key.endswith("_seconds")},
                            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        record = records[0]
        row = {"implementation": impl, "median_wall_seconds": statistics.median(r["wall_seconds"] for r in samples),
               "median_perturbation_seconds": statistics.median(r["stage_seconds"]["perturbation_seconds"] for r in samples),
               "median_greedy_seconds": statistics.median(r["stage_seconds"]["greedy_seconds"] for r in samples),
               "selected_group_hash": record["selected_group_hash"], "samples": samples,
               "stable_recomputed_group_count": len(record.get("stable_recomputed_group_ids", [])),
               "cancellation_recomputed_group_count": len(record.get("cancellation_recomputed_group_ids", []))}
        values[impl] = record
        if impl != "reference":
            ref = values["reference"]
            row["same_keep_idx_as_reference"] = record["raw_keep_idx_hash"] == ref["raw_keep_idx_hash"]
            row["different_group_count"] = len(set(record["final_group_index"]) ^ set(ref["final_group_index"]))
            for key in ("deletion_scores", "support"):
                row[key + "_max_abs_error"] = max(abs(x - y) for x, y in zip(record[key], ref[key]))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    directory = Path(a.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"scope": "synthetic native-sized selector only; not an 8B end-to-end or VQA accuracy benchmark", "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": torch.__version__, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "settings": vars(a), "records": rows}
    (directory / "selector_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
