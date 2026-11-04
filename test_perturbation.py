#!/usr/bin/env python3
"""Independent perturbation-guided VQA experiment; existing testn is unchanged."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import evaluator


METHODS = (
    "text_cos", "q_flowcover_direct_residual",
    "q_flowcover_perturbation", "q_flowcover_perturbation_nocover",
)


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = evaluator.build_parser()
    parser.description = "Question-conditioned group deletion error + local coverage (independent experiment)"
    parser.set_defaults(output_dir="outputs/q_flowcover_perturbation", worker_cases=0)
    parser.add_argument("--prune_method", "--prune-method", choices=METHODS,
                        default="q_flowcover_perturbation")
    parser.add_argument("--exec_mode", "--exec-mode", choices=("full19_gather",), default="full19_gather",
                        help="Historical name: full execution through l+1, then exactly one gather; l is configurable")
    parser.add_argument("--decision_layer_human", "--decision-layer-human", type=int, default=18,
                        help="Score layer l (1-based); native attention is from l+1, gather follows l+1")
    parser.add_argument("--prune_fraction", "--prune-fraction", type=float, default=None,
                        help="REMOVED fraction in [0,1); same per-view budget rounding as baseline")
    parser.add_argument("--text_source", "--text-source", choices=("baseline",), default="baseline")
    parser.add_argument("--temperature", type=float, default=1., help="p=softmax(r/tau), pi=softmax(r[P]/tau)")
    parser.add_argument("--perturbation_lambda", "--perturbation-lambda", type=float, default=1.,
                        help="Raw deletion-error coverage multiplier; no implicit error renormalization")
    parser.add_argument("--local_radius", "--local-radius", type=int, default=1)
    parser.add_argument("--query_chunk_size", "--query-chunk-size", type=int, default=256,
                        help="Maximum raw Queries in an extra statistics chunk")
    parser.add_argument("--group_chunk_size", "--group-chunk-size", type=int, default=32)
    parser.add_argument("--source_chunk_size", "--source-chunk-size", type=int, default=256)
    parser.add_argument("--selector_impl", "--selector-impl", choices=("reference", "optimized"), default="reference",
                        help="Explicit implementation; optimized keeps all text Top-K Queries and the original objective")
    parser.add_argument("--max_attention_mb", "--max-attention-mb", type=float, default=512.,
                        help="Optimized rectangular attention allocation budget; only adjusts chunk size")
    parser.add_argument("--greedy_backend", "--greedy-backend", choices=("reference", "local_refresh", "scripted_local_refresh"), default="reference",
                        help="Optimized exact local greedy implementation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup_cases", "--warmup-cases", type=int, default=1)
    parser.add_argument("--verify_candidates", "--verify-candidates", type=int, default=2,
                        help="One-time explicit deletion check on the first measured sample; 0 disables")
    parser.add_argument("--save_diagnostics", "--save-diagnostics", action=argparse.BooleanOptionalAction,
                        default=True, help="Save panels after timing; optional same-layer flow comparison is additional measured selector work")
    parser.add_argument("--save_per_sample", "--save-per-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="One real case, with index and formula checks")
    args = parser.parse_args(argv)
    if args.decision_layer_human < 2:
        parser.error("--decision-layer-human must be >=2 (the shared three-layer indexing contract)")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and >0")
    if not math.isfinite(args.perturbation_lambda) or args.perturbation_lambda < 0:
        parser.error("--perturbation-lambda must be finite and >=0")
    if args.local_radius < 0 or min(args.query_chunk_size, args.group_chunk_size, args.source_chunk_size) < 1:
        parser.error("radius must be non-negative and all chunk sizes positive")
    if args.warmup_cases < 0 or args.verify_candidates < 0:
        parser.error("warmup/verification counts must be non-negative")
    if not math.isfinite(args.max_attention_mb) or args.max_attention_mb <= 0:
        parser.error("--max-attention-mb must be finite and positive")
    if args.prune_method in METHODS[:2] and args.selector_impl != "reference":
        parser.error("--selector-impl optimized applies to perturbation C/D only; A/B keep their original implementations")
    if args.prune_fraction is not None:
        if not math.isfinite(args.prune_fraction) or not 0 <= args.prune_fraction < 1:
            parser.error("--prune-fraction is the removed fraction in [0,1)")
        if args.visual_prune_percent is not None:
            parser.error("--prune-fraction and --visual-prune-percent are mutually exclusive")
        args.visual_prune_percent = args.prune_fraction * 100
    if args.prune_method in METHODS[:2] and (args.temperature != 1. or args.perturbation_lambda != 1.):
        parser.error("A/B controls retain their original defaults; temperature/lambda only apply to C/D")
    config_path = Path(args.model).expanduser() / "config.json"
    depth = 24
    args.config_verified = config_path.is_file()
    if config_path.is_file():
        vision = json.loads(config_path.read_text(encoding="utf-8")).get("vision_config", {})
        depth = int(vision.get("depth", 0))
        if not depth:
            parser.error("checkpoint config has no valid vision_config.depth")
    if args.decision_layer_human >= depth:
        parser.error(f"score layer l={args.decision_layer_human} needs native l+1; vision depth={depth}")
    layer = args.decision_layer_human
    args.evaluation_variant = "perturbation"
    args.compression_mode = args.exec_mode
    args.pruning_mode = "middle_text_early"
    args.pruning_layer = layer
    args.pruning_strategy = "topk"
    args.vision_layers = f"{min(6, layer - 1)},{layer},{depth}"
    args.visual_pruning = True
    args.score_layer = layer
    args.perturbation_layer = layer + 1
    args.actual_prune_after_layer = layer + 1
    args.decision_block_index = layer - 1
    args.bridge_block_index = layer
    # Existing A/B use their original selector unchanged. These fields also
    # preserve the common evaluator's resume and output contracts.
    args.q_flowcover_radius = args.local_radius
    args.q_flowcover_local_radius = args.local_radius
    args.q_flowcover_query_chunk = args.query_chunk_size
    args.q_flowcover_candidate_chunk = args.group_chunk_size
    args.q_flowcover_query_mass = .9
    args.q_flowcover_adaptive_radius_max = 3
    args.q_flowcover_affinity_mode = "raw"
    args.att_stats = "none"
    args.save_att_stats = False
    args.max_debug_samples = 0
    if args.dry_run:
        args.limit = 1
        args.verify_candidates = max(2, args.verify_candidates)
    return args


def main(argv: list[str] | None = None):
    summary = evaluator.test(get_args(argv))
    if int(summary.get("errors", 0)):
        raise RuntimeError("experiment contains failed cases; inspect saved results.jsonl before continuing the sweep")
    return summary


if __name__ == "__main__":
    main()
