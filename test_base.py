#!/usr/bin/env python3
"""Evaluate native, unpruned Qwen3-VL with the current shared FLOPs estimate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluator


def _parser() -> argparse.ArgumentParser:
    parser = evaluator.build_parser()
    parser.description = __doc__
    parser.add_argument("--warmup-cases", type=int, default=1,
                        help="Warm up on the first n cases before their measured inference; warmup is not scored")
    parser.set_defaults(
        output_dir="outputs/qwen3_base_current", worker_cases=0, dtype="bf16",
        visual_pruning=False, visual_keep_ratio=1.0, visual_prune_percent=0.0,
        visual_keep_tokens=None, prune_method="qwen3_base", compression_mode="off",
        exec_mode="off", pruning_mode="none", pruning_layer=18,
        decision_layer_human=None, decision_block_index=None,
        vision_layers="6,18,27", pruning_strategy="none", text_source="baseline",
        evaluation_variant="base", att_stats="none", save_att_stats=False,
        save_per_sample=False, max_debug_samples=0,
        q_flowcover_radius=1, q_flowcover_query_chunk=256,
        q_flowcover_candidate_chunk=64, q_flowcover_query_mass=0.9,
        q_flowcover_local_radius=1, q_flowcover_adaptive_radius_max=3,
        q_flowcover_affinity_mode="raw",
        flops_accounting="qwen3_vl_raw_tokens_native_mergers_v1",
    )
    return parser


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.warmup_cases < 0:
        parser.error("--warmup-cases must be nonnegative")
    if args.visual_prune_percent != 0 or args.visual_keep_ratio != 1.0 or args.visual_keep_tokens is not None:
        parser.error("The native baseline retains all visual groups; pruning options must remain 0% / keep-ratio 1")
    path = Path(args.model).expanduser() / "config.json"
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
        vision = config.get("vision_config", {})
        depth = int(vision.get("depth", vision.get("num_hidden_layers", 27)))
        args.vision_layers = f"6,18,{depth}"
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    summary = evaluator.test(get_args(argv))
    if int(summary.get("errors", 0)):
        raise RuntimeError(f"Native baseline completed with {summary['errors']} error cases; inspect saved results and resume to retry them")
    return summary


if __name__ == "__main__":
    main()
