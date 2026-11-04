#!/usr/bin/env python3
"""Evaluate perturbation-guided visual pruning or the native Qwen3-VL base."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--method", choices=("perturbation", "nocover", "base"), default="perturbation"
    )
    parser.add_argument("--implementation", choices=("reference", "optimized"), default="optimized")
    parser.add_argument(
        "--layer", type=int, default=18, help="Score at L; gather after native full block L+1"
    )
    parser.add_argument("--prune-percent", type=float, default=None, help="Percentage removed")
    parser.add_argument(
        "--keep-tokens", type=int, default=None, help="Optional fixed merged-group budget per view"
    )
    parser.add_argument("--output-dir", default="outputs/perturbation")
    parser.add_argument(
        "--gpus", default=None, help="Comma-separated visible CUDA IDs; otherwise GPU 0 or CPU"
    )
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-image-pixels", type=int, default=4194304)
    parser.add_argument("--max-image-patches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--include-global-view", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tile-single-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-selection", choices=("uniform", "head", "tail"), default="uniform")
    parser.add_argument("--patch-size", type=int, default=448)
    parser.add_argument("--patch-stride", type=int, default=336)
    parser.add_argument("--global-view-size", type=int, default=448)
    parser.add_argument("--decode-hf-images", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--perturbation-lambda", type=float, default=1.0)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--group-chunk-size", type=int, default=32)
    parser.add_argument("--source-chunk-size", type=int, default=256)
    parser.add_argument("--max-attention-mb", type=float, default=512.0)
    parser.add_argument(
        "--cancellation-recovery",
        choices=("deferred", "inline"),
        default="deferred",
        help="Repair numerical cancellation using the current attention chunk",
    )
    parser.add_argument(
        "--greedy-backend",
        choices=("reference", "local_refresh", "scripted_local_refresh", "cuda_fused"),
        default="reference",
    )
    parser.add_argument("--diagnostics", choices=("summary", "full"), default="summary")
    parser.add_argument("--verify-candidates", type=int, default=0)
    parser.add_argument("--warmup-cases", type=int, default=1)
    parser.add_argument(
        "--worker-cases",
        type=int,
        default=0,
        help="0 keeps one model per device; positive values restart workers after each batch",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args(argv)
    if (args.implementation != "optimized" or args.method == "base") and (
        args.cancellation_recovery != "deferred" or args.greedy_backend == "cuda_fused"
    ):
        parser.error("inline recovery and cuda_fused require optimized perturbation/nocover")
    if args.layer < 2:
        parser.error("--layer must be at least 2")
    if args.method == "base":
        if args.prune_percent not in (None, 0) or args.keep_tokens is not None:
            parser.error("base retains all visual groups; do not set a pruning budget")
        args.prune_percent = 0.0
    else:
        if args.keep_tokens is not None and args.prune_percent is not None:
            parser.error("--keep-tokens and --prune-percent are mutually exclusive")
        args.prune_percent = 70.0 if args.prune_percent is None else args.prune_percent
    if not math.isfinite(args.prune_percent) or not 0 <= args.prune_percent < 100:
        parser.error("--prune-percent must be in [0,100)")
    positive = (
        "max_image_pixels",
        "max_image_patches",
        "max_new_tokens",
        "patch_size",
        "patch_stride",
        "global_view_size",
        "query_chunk_size",
        "group_chunk_size",
        "source_chunk_size",
    )
    if any(getattr(args, name) <= 0 for name in positive) or (
        args.limit is not None and args.limit <= 0
    ):
        parser.error("image, token, chunk and sample counts must be positive")
    if args.keep_tokens is not None and args.keep_tokens <= 0:
        parser.error("--keep-tokens must be positive")
    if min(args.local_radius, args.warmup_cases, args.verify_candidates, args.worker_cases) < 0:
        parser.error("radius, warmup, verification and worker counts must be nonnegative")
    if args.greedy_backend == "cuda_fused" and args.local_radius > 2:
        parser.error("cuda_fused supports --local-radius 0, 1 or 2")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be positive and finite")
    if not math.isfinite(args.perturbation_lambda) or args.perturbation_lambda < 0:
        parser.error("--perturbation-lambda must be nonnegative and finite")
    if not math.isfinite(args.max_attention_mb) or args.max_attention_mb <= 0:
        parser.error("--max-attention-mb must be positive and finite")
    model_config = Path(args.model).expanduser() / "config.json"
    if model_config.is_file() and args.method != "base":
        vision = json.loads(model_config.read_text())["vision_config"]
        if args.layer >= int(vision["depth"]):
            parser.error("--layer needs a following native visual block")
    return args


def main(argv=None):
    from evaluation.runner import run

    summary = run(get_args(argv))
    if summary["errors"]:
        raise RuntimeError(
            f"{summary['errors']} error cases were saved; use --resume to retry them"
        )
    return summary


if __name__ == "__main__":
    main()
