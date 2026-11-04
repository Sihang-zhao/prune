#!/usr/bin/env python3
"""Reproduce a small four-method, single-case CPU inference comparison.

Use the saved tiny random validation checkpoint to check delivery defaults;
this is not an 8B/GPU throughput or VQA accuracy benchmark.  Each method loads
the same checkpoint once, warms up, then repeats the same prepared image and
prompt.  Loading, image I/O and plotting are excluded from model testtime.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

# Make direct ``python scripts/benchmark_perturbation.py`` invocation portable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/perturbation_benchmark")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")

    import numpy as np
    import torch
    from compression.perturbation_worker import _load, align_comparison_flops, build_runner
    from test_perturbation import METHODS, get_args
    from utils.data import load_cases

    torch.set_num_threads(1)
    model = str(Path(args.model).expanduser().resolve())
    data = str(Path(args.data).expanduser().resolve())
    cases = load_cases(data, limit=1, split="train")
    if not cases:
        raise ValueError("benchmark dataset contains no case")
    case = cases[0]
    records = []
    wall_started = time.perf_counter()
    for method in METHODS:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        # Query/group/source defaults are intentionally inherited from the
        # delivered entry rather than copied or overridden by this script.
        settings = get_args([
            "--model", model, "--data", data, "--offline", "--dtype", "fp32",
            "--prune-method", method, "--decision-layer-human", "18",
            "--visual-prune-percent", "70", "--max-image-patches", "1",
            "--max-image-pixels", "65536", "--max-new-tokens", "2",
            "--verify-candidates", "0", "--no-save-diagnostics",
            "--warmup-cases", str(args.warmup), "--seed", "0",
        ])
        cfg = vars(settings).copy()
        cfg["visual_keep_ratio"] = 1.0 - float(cfg["visual_prune_percent"]) / 100.0
        runner = build_runner("cpu", cfg)
        images = _load(case, cfg)
        for _ in range(args.warmup):
            runner.answer(images, case.eval_prompt(), scoring_text=case.question)
        samples = []
        traces = []
        for _ in range(args.repeats):
            output = runner.answer(images, case.eval_prompt(), scoring_text=case.question)
            align_comparison_flops(output, runner, cfg)
            stages = dict(getattr(output, "stage_timings", {}) or {})
            sample = {
                "inference_seconds": float(output.seconds),
                "selector_seconds": stages.get("selector"),
                "estimated_flops": int(output.flops),
                "stages": stages,
                "answer": output.answer,
                "selected_indices": output.selected_indices,
                "qkv_capture_count": getattr(output, "qkv_capture_count", None),
                "qkv_projection_replay_count": getattr(output, "qkv_projection_replay_count", None),
                "actual_gather_count": getattr(output, "actual_gather_count", None),
            }
            samples.append(sample)
            traces.append(list(output.vision_layer_token_counts))
        if any(trace != traces[0] for trace in traces):
            raise RuntimeError(f"{method}: repeated native token traces differ")
        if any(sample["selected_indices"] != samples[0]["selected_indices"] for sample in samples):
            raise RuntimeError(f"{method}: repeated selected original group IDs differ")
        row = {
            "method": method,
            "N": int(output.original_image_tokens),
            "K": int(output.retained_tokens),
            "raw_token_trace": traces[0],
            "image_grid_thw": getattr(output, "image_grid_thw", None),
            "input_image_sizes": [list(image.size) for image in images],
            "repeated_answers_identical": all(sample["answer"] == samples[0]["answer"] for sample in samples),
            "median_inference_seconds": statistics.median(sample["inference_seconds"] for sample in samples),
            "median_selector_seconds": statistics.median(sample["selector_seconds"] for sample in samples if sample["selector_seconds"] is not None),
            "query_chunk": cfg["query_chunk_size"],
            "group_chunk": cfg["group_chunk_size"],
            "source_chunk": cfg["source_chunk_size"],
            "records": samples,
        }
        records.append(row)
        print(f"{method}: inference={row['median_inference_seconds']:.6f}s; "
              f"selector={row['median_selector_seconds']:.6f}s; N={row['N']}; K={row['K']}", flush=True)
        del runner, output, images

    if len({(row["N"], row["K"]) for row in records}) != 1:
        raise RuntimeError("methods did not execute the same per-view group budget")

    report = {
        "scope": "CPU native checkpoint + real processor, one fixed case, delivery-default chunks; diagnostics/verification disabled. The saved validation checkpoint is tiny random; these timings do NOT measure 8B/GPU performance or VQA accuracy.",
        "model": model, "data": data, "sample_id": case.id,
        "cpu_threads": 1, "dtype": "float32", "seed": 0,
        "score_layer": 18, "perturbation_layer": 19,
        "gather_after_human_layer": 19, "exec_mode": "full19_gather",
        "removed_percent": 70, "temperature": 1.0, "perturbation_lambda": 1.0,
        "local_radius": cfg["local_radius"], "query_chunk": cfg["query_chunk_size"],
        "group_chunk": cfg["group_chunk_size"], "source_chunk": cfg["source_chunk_size"],
        "max_image_pixels": 65536, "max_image_patches": 1, "max_new_tokens": 2,
        "warmup_iterations_per_method": args.warmup, "measured_iterations_per_method": args.repeats,
        "diagnostics_enabled": False, "verification_enabled": False,
        "inference_timing_scope": "visual encoding through answer text decoding, including selector/indexing/LLM; processor and image I/O excluded, as in the evaluator",
        "peak_gpu_memory_bytes": None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_used": False},
        "records": records,
        "entire_benchmark_wall_seconds": time.perf_counter() - wall_started,
    }
    directory = Path(args.output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "cpu_default_chunk_comparison.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"saved: {destination}", flush=True)
    return report


if __name__ == "__main__":
    main()
