#!/usr/bin/env python3
"""Paired real-checkpoint reference/optimized inference; one model load.

Original prepared inputs, model dtype, prompt, decoder and group budget are
identical. Raw scores, actual selected IDs, answers, CUDA wall times and
memory are compared rather than extrapolating a synthetic selector benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-layer", type=int, default=18)
    parser.add_argument("--prune-percent", type=float, default=50.)
    parser.add_argument("--greedy-backend", choices=("reference", "local_refresh", "scripted_local_refresh"), default="reference")
    parser.add_argument("--case-selection", choices=("head", "uniform"), default="head",
                        help="Uniform samples the entire ordered dataset deterministically; no score preselection")
    parser.add_argument("--output-dir", default="outputs/perturbation_optimization/real_e2e")
    args = parser.parse_args()
    import torch
    from test_perturbation import get_args
    from compression.perturbation_worker import build_runner, _load
    from utils.data import load_cases

    if args.n < 1:
        parser.error("n must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    settings = get_args([
        "--data", args.data, "--model", args.model, "--offline",
        "--prune-method", "q_flowcover_perturbation", "--selector-impl", "optimized",
        "--decision-layer-human", str(args.score_layer), "--visual-prune-percent", str(args.prune_percent),
        "--greedy-backend", args.greedy_backend, "--no-save-diagnostics", "--verify-candidates", "0",
        "--max-image-patches", "1", "--max-new-tokens", "32", "--max-image-pixels", "4194304",
    ])
    cfg = vars(settings).copy()
    cfg["visual_keep_ratio"] = 1. - args.prune_percent / 100.
    cfg["image_root"] = str(Path(args.data).resolve() if Path(args.data).is_dir() else Path(args.data).resolve().parent)
    if args.case_selection == "head":
        cases = load_cases(args.data, limit=args.n, split="train")
    else:
        all_cases = load_cases(args.data, split="train")
        count = min(args.n, len(all_cases))
        indices = [0] if count == 1 else [round(i * (len(all_cases) - 1) / (count - 1)) for i in range(count)]
        cases = [all_cases[i] for i in indices]
    runner = build_runner(str(device), cfg)
    first_images = _load(cases[0], cfg)
    for impl in ("reference", "optimized"):
        print(f"warmup {impl}", flush=True)
        runner.perturbation_selector_impl = impl
        runner.answer(first_images, cases[0].eval_prompt(), scoring_text=cases[0].question)
    del first_images
    rows = []
    for case in cases:
        images = _load(case, cfg)
        results = {}
        for impl in ("reference", "optimized"):
            runner.perturbation_selector_impl = impl
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            output = runner.answer(images, case.eval_prompt(), scoring_text=case.question)
            records = output.q_flowcover_debug
            result = {
                "inference_seconds": output.seconds,
                "pruning_latency_seconds": output.pruning_latency,
                "stage_timings": output.stage_timings,
                "peak_memory_bytes": output.peak_memory,
                "answer": output.answer,
                "selected_indices": output.selected_indices,
                "raw_keep_idx_hash": output.raw_keep_idx_hash,
                "N": output.original_image_tokens, "K": output.retained_tokens,
                "vision_layer_token_counts": output.vision_layer_token_counts,
                "qkv_capture_count": output.qkv_capture_count,
                "qkv_projection_replay_count": output.qkv_projection_replay_count,
                "views": [{key: record.get(key) for key in (
                    "implementation_file", "statistics_implementation", "max_attention_chunk_shape",
                    "query_group_count", "source_query_count", "stable_recomputed_group_ids",
                    "cancellation_recomputed_group_ids", "greedy_implementation", "deletion_scores", "support",
                )} for record in records],
            }
            results[impl] = result
            print(f"sample {case.index} {impl}: {output.seconds:.4f}s; "
                  f"perturb={output.stage_timings['perturbation']:.4f}s; "
                  f"greedy={output.stage_timings['greedy']:.4f}s", flush=True)
        ref, opt = results["reference"], results["optimized"]
        same = ref["raw_keep_idx_hash"] == opt["raw_keep_idx_hash"]
        errors = {key: max(abs(x - y) for rv, ov in zip(ref["views"], opt["views"])
                           for x, y in zip(rv[key], ov[key])) for key in ("deletion_scores", "support")}
        row = {"sample_id": case.id, "sample_index": case.index,
               "same_keep_idx": same, "same_answer": ref["answer"] == opt["answer"],
               "different_group_count": sum(len(set(r) ^ set(o)) for r, o in zip(ref["selected_indices"], opt["selected_indices"])),
               "max_abs_errors": errors, "implementations": results}
        rows.append(row)
        print(f"equivalence: keep_idx={same}; answer={row['same_answer']}; errors={errors}", flush=True)
    summary = {
        "scope": "real-checkpoint paired inference on same ordered XLRS cases; GPU may be shared with other processes",
        "settings": vars(args), "model_loaded_once": True, "warmup_per_implementation": 1,
        "model_dtype": str(next(runner.model.parameters()).dtype),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "case_count": len(rows), "all_keep_idx_equal": all(r["same_keep_idx"] for r in rows),
        "all_answers_equal": all(r["same_answer"] for r in rows), "cases": rows,
    }
    for impl in ("reference", "optimized"):
        summary[impl] = {
            "mean_inference_seconds": statistics.mean(r["implementations"][impl]["inference_seconds"] for r in rows),
            "median_inference_seconds": statistics.median(r["implementations"][impl]["inference_seconds"] for r in rows),
            "mean_perturbation_seconds": statistics.mean(r["implementations"][impl]["stage_timings"]["perturbation"] for r in rows),
            "mean_greedy_seconds": statistics.mean(r["implementations"][impl]["stage_timings"]["greedy"] for r in rows),
        }
    summary["target_1_2_seconds_met_on_these_cases"] = summary["optimized"]["mean_inference_seconds"] <= 1.2
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
