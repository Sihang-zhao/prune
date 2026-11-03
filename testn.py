#!/usr/bin/env python3
"""Portable evaluator for the recorded Q-FlowCover direct-residual experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluator


def _parser() -> argparse.ArgumentParser:
    parser = evaluator.build_parser()
    parser.add_argument(
        "--prune_method", "--prune-method",
        choices=("text_cos", "q_flowcover_direct_residual"),
        default="q_flowcover_direct_residual", help="Recorded direct-residual selector or text-cosine baseline",
    )
    parser.add_argument(
        "--exec_mode", "--exec-mode",
        choices=("hard18", "full19_gather", "att_bridge19_fullkv"),
        default="hard18", help="Execution of visual blocks after the selected pruning layer",
    )
    parser.add_argument("--prune_fraction", "--prune-fraction", type=float, default=None,
                        help="Fraction removed, in [0,1); alternative to --visual-prune-percent")
    parser.add_argument("--qflow_radius", "--qflow-radius", type=int, default=1,
                        help="Legacy recorded field; direct-residual uses --local-radius")
    parser.add_argument("--qflow_query_chunk", "--qflow-query-chunk", type=int, default=256)
    parser.add_argument("--qflow_candidate_chunk", "--qflow-candidate-chunk", type=int, default=64,
                        help="Legacy recorded field; not used by direct-residual")
    parser.add_argument("--local_radius", "--local-radius", type=int, default=1,
                        help="Sparse local coverage radius; recorded experiments used 1")
    parser.add_argument("--affinity_mode", "--affinity-mode", choices=("raw", "local_centered"), default="raw",
                        help="Local affinity; recorded experiments used raw")
    parser.add_argument("--query_mass", "--query-mass", type=float, default=0.9,
                        help="Legacy field; this method always queries text Top-K groups")
    parser.add_argument("--adaptive_radius_max", "--adaptive-radius-max", type=int, default=3,
                        help="Legacy field; this method uses the fixed --local-radius")
    parser.add_argument("--text_source", "--text-source", choices=("baseline",), default="baseline",
                        help="Scoring text is the question alone, as in the recorded runs")
    parser.add_argument("--save_per_sample", "--save-per-sample",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Include additional per-sample index diagnostics")
    parser.add_argument("--att_stats", "--att-stats", choices=("none", "summary", "debug"), default="none")
    parser.add_argument("--save_att_stats", "--save-att-stats", action="store_true")
    parser.add_argument("--max_debug_samples", "--max-debug-samples", type=int, default=0)
    parser.add_argument("--decision_layer_human", "--decision-layer-human", type=int, choices=(15, 16, 17, 18), default=18,
                        help="Prune once after this visual block (1-based); the historical default is 18")
    return parser


def resolve_decision_block(model: str, decision_layer_human: int = 18) -> dict[str, Any]:
    """Validate the selected layer boundary for a local checkpoint."""
    layer = int(decision_layer_human)
    model_path = Path(model).expanduser()
    config_path = model_path / "config.json" if model_path.is_dir() else None
    details: dict[str, Any] = {
        "decision_layer_human": layer,
        "decision_block_index": layer - 1,
        "config_path": str(config_path) if config_path else None,
        "config_verified": False,
    }
    if config_path is not None and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vision = config.get("vision_config", config)
        depth = int(vision.get("depth", vision.get("num_hidden_layers", 0)) or 0)
        if layer <= 0 or depth <= layer:
            raise ValueError(f"layer {layer} must precede the final visual block; checkpoint depth={depth}")
        details.update(
            config_verified=True,
            vision_depth=depth,
            deepstack_visual_indexes=list(vision.get("deepstack_visual_indexes", [])),
            spatial_merge_size=int(vision.get("spatial_merge_size", 1)),
        )
    return details


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.qflow_radius < 0 or args.local_radius < 0:
        parser.error("--qflow-radius and --local-radius must be non-negative")
    if args.qflow_query_chunk <= 0 or args.qflow_candidate_chunk <= 0:
        parser.error("Q-FlowCover chunk sizes must be positive")
    if not 0.0 < args.query_mass <= 1.0:
        parser.error("--query-mass must be in (0,1]")
    if args.adaptive_radius_max < 1:
        parser.error("--adaptive-radius-max must be at least 1")
    if args.prune_fraction is not None:
        if not 0.0 <= args.prune_fraction < 1.0:
            parser.error("--prune-fraction must be in [0,1)")
        if args.visual_prune_percent is not None:
            parser.error("--prune-fraction cannot be combined with --visual-prune-percent")
        args.visual_prune_percent = args.prune_fraction * 100.0
    if args.att_stats != "none" and args.exec_mode != "att_bridge19_fullkv":
        parser.error("--att-stats requires --exec-mode att_bridge19_fullkv")
    if args.save_att_stats and args.att_stats == "none":
        parser.error("--save-att-stats requires --att-stats summary or debug")
    if args.max_debug_samples < 0:
        parser.error("--max-debug-samples must be non-negative")
    args.compression_mode = args.exec_mode
    args.pruning_mode = "middle_text_early"
    args.pruning_layer = args.decision_layer_human
    args.pruning_strategy = "topk"
    args.vision_layers = f"6,{args.decision_layer_human},24"
    args.visual_pruning = True
    args.q_flowcover_radius = args.qflow_radius
    args.q_flowcover_query_chunk = args.qflow_query_chunk
    args.q_flowcover_candidate_chunk = args.qflow_candidate_chunk
    args.q_flowcover_query_mass = args.query_mass
    args.q_flowcover_local_radius = args.local_radius
    args.q_flowcover_adaptive_radius_max = args.adaptive_radius_max
    args.q_flowcover_affinity_mode = args.affinity_mode
    for key, value in resolve_decision_block(args.model, args.decision_layer_human).items():
        setattr(args, key, value)
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return evaluator.test(get_args(argv))


if __name__ == "__main__":
    main()
