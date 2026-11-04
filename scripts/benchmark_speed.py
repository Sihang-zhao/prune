#!/usr/bin/env python3
"""Compare current and accelerated inference on identical cases and one model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--prune-percent", type=float, default=70)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-image-pixels", type=int, default=4194304)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="outputs/speed_validation/paired_inference.json")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    import torch
    import triton
    from evaluate import get_args
    from evaluation.data import load_cases
    from evaluation.worker import build_runner, _load

    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        parser.error("This comparison requires the CUDA/Triton environment")
    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    try:
        gpu_processes = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    args.device.split(":")[1],
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.TimeoutExpired):
        gpu_processes = None
    settings = get_args(
        [
            "--data",
            args.data,
            "--model",
            args.model,
            "--offline",
            "--dtype",
            "bf16",
            "--layer",
            str(args.layer),
            "--prune-percent",
            str(args.prune_percent),
            "--max-image-pixels",
            str(args.max_image_pixels),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
    )
    cfg = vars(settings).copy()
    cfg["model"] = str(Path(args.model).expanduser().resolve())
    cfg["data"] = str(Path(args.data).expanduser().resolve())
    data = Path(cfg["data"])
    cfg["image_root"] = str(data if data.is_dir() else data.parent)
    cfg["vision_depth"] = json.loads((Path(cfg["model"]) / "config.json").read_text())[
        "vision_config"
    ]["depth"]
    all_cases = load_cases(cfg["data"], None, cfg["split"])
    count = min(args.cases, len(all_cases))
    if not count:
        raise ValueError("No cases to benchmark")
    indices = (
        [0] if count == 1 else [round(i * (len(all_cases) - 1) / (count - 1)) for i in range(count)]
    )
    cases = [all_cases[index] for index in indices]
    model = build_runner(args.device, cfg)
    modes = {"current": ("reference", "deferred"), "accelerated": ("cuda_fused", "inline")}

    def configure(mode):
        model.perturbation_greedy_backend, model.perturbation_cancellation_recovery = modes[mode]

    records = []
    for position, case in enumerate(cases):
        images = _load(case, cfg)
        prompt = case.eval_prompt()
        # Warm both modes on this input shape, including Triton compilation.
        for mode in modes:
            configure(mode)
            model.answer(images, prompt, scoring_text=case.question)
        row = {"index": case.index, "id": case.id}
        order = list(modes) if position % 2 == 0 else list(reversed(modes))
        for mode in order:
            configure(mode)
            torch.cuda.synchronize(args.device)
            output = model.answer(images, prompt, scoring_text=case.question)
            row[mode] = {
                "seconds": float(output.seconds),
                "stages": output.stage_timings,
                "answer": output.answer,
                "selected_indices": output.selected_indices,
                "trace": output.vision_layer_token_counts,
                "query_counts": [view["query_group_count"] for view in output.q_flowcover_debug],
                "query_hashes": [view["S0_hash"] for view in output.q_flowcover_debug],
                "N": output.original_image_tokens,
                "K": output.retained_tokens,
            }
            del output
        row["same_indices"] = (
            row["current"]["selected_indices"] == row["accelerated"]["selected_indices"]
        )
        row["same_answer"] = row["current"]["answer"] == row["accelerated"]["answer"]
        row["same_trace"] = row["current"]["trace"] == row["accelerated"]["trace"]
        row["same_queries"] = (
            row["current"]["query_counts"] == row["accelerated"]["query_counts"]
            and row["current"]["query_hashes"] == row["accelerated"]["query_hashes"]
        )
        records.append(row)
        print(
            f"case {case.index}: {row['current']['seconds']:.4f}s -> {row['accelerated']['seconds']:.4f}s; indices_equal={row['same_indices']}",
            flush=True,
        )
    report = {
        "scope": "Paired native inference, identical prepared views/prompts/budget, one model load; warmup and compilation excluded.",
        "data": cfg["data"],
        "model": cfg["model"],
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "config": cfg,
        "mode_overrides": {
            name: {"greedy_backend": backend, "cancellation_recovery": recovery}
            for name, (backend, recovery) in modes.items()
        },
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "measured_calls_per_case_per_mode": 1,
        "gpu_processes_before_model_load": gpu_processes,
        "layer": args.layer,
        "prune_percent": args.prune_percent,
        "cases": len(records),
        "records": records,
    }
    for mode in modes:
        report[mode] = {
            "mean_seconds": statistics.mean(row[mode]["seconds"] for row in records),
            "mean_selector_seconds": statistics.mean(
                row[mode]["stages"]["selector"] for row in records
            ),
            "mean_statistics_seconds": statistics.mean(
                row[mode]["stages"]["perturbation"] for row in records
            ),
            "mean_greedy_seconds": statistics.mean(
                row[mode]["stages"]["greedy"] for row in records
            ),
        }
    report["all_indices_equal"] = all(row["same_indices"] for row in records)
    report["all_answers_equal"] = all(row["same_answer"] for row in records)
    report["all_queries_equal"] = all(row["same_queries"] for row in records)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"mean: {report['current']['mean_seconds']:.4f}s -> {report['accelerated']['mean_seconds']:.4f}s; report={path}"
    )
    if not all(
        row["same_indices"] and row["same_answer"] and row["same_trace"] and row["same_queries"]
        for row in records
    ):
        raise RuntimeError(
            "Paired selection/answer equivalence failed; inspect the report before using this backend"
        )


if __name__ == "__main__":
    main()
