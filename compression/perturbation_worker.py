"""Spawn-safe worker used exclusively by test_perturbation.py.

The canonical dataset, prompt, answer parser and reporting contracts are
shared. Model loading is persistent; plotting and serialization follow the
model's timed answer call. A/B explicitly instantiate the old runner.
"""
from __future__ import annotations

import random
import time
import traceback
from pathlib import Path
from typing import Any

from utils.data import Case, load_image
from utils.metrics import score_prediction
from .worker import _extra_record, _path_value, _worker_dtype


def build_runner(device: str, cfg: dict[str, Any]):
    from .qwen3_vl_modes import Qwen3VLCompressionRunner
    from .qwen3_vl_perturbation import PERTURBATION_METHODS, Qwen3VLPerturbationRunner

    method = cfg["prune_method"]
    if method not in {*PERTURBATION_METHODS, "text_cos", "q_flowcover_direct_residual"}:
        raise RuntimeError(f"unregistered perturbation experiment method: {method}")
    kwargs = dict(
        name=cfg["model"], device=device, dtype=_worker_dtype(cfg["dtype"], device),
        max_new_tokens=cfg["max_new_tokens"], trust_remote_code=cfg["trust_remote_code"],
        local_files_only=cfg["offline"], max_image_pixels=cfg["max_image_pixels"],
        enable_thinking=cfg["enable_thinking"], visual_pruning=True,
        visual_keep_ratio=cfg["visual_keep_ratio"], visual_keep_tokens=cfg.get("visual_keep_tokens"),
        vision_layers=tuple(map(int, cfg["vision_layers"].split(","))),
        pruning_mode="middle_text_early", pruning_layer=cfg["decision_layer_human"],
        pruning_strategy="topk", compression_mode=cfg["compression_mode"],
        prune_method=method, decision_layer_human=cfg["decision_layer_human"],
    )
    if method in PERTURBATION_METHODS:
        return Qwen3VLPerturbationRunner(
            **kwargs, perturbation_lambda=cfg["perturbation_lambda"],
            perturbation_temperature=cfg["temperature"],
            perturbation_query_chunk=cfg["query_chunk_size"],
            perturbation_group_chunk=cfg["group_chunk_size"],
            perturbation_key_chunk=cfg["source_chunk_size"],
            perturbation_local_radius=cfg["local_radius"],
            perturbation_verify_candidates=0,
            perturbation_compare_flow=cfg["save_diagnostics"],
            perturbation_selector_impl=cfg.get("selector_impl", "reference"),
            perturbation_max_attention_mb=cfg.get("max_attention_mb", 512.),
            perturbation_greedy_backend=cfg.get("greedy_backend", "reference"),
        )
    return Qwen3VLCompressionRunner(
        **kwargs, q_flowcover_query_chunk=cfg["q_flowcover_query_chunk"],
        q_flowcover_candidate_chunk=cfg["q_flowcover_candidate_chunk"],
        q_flowcover_local_radius=cfg["q_flowcover_local_radius"],
        q_flowcover_radius=cfg["q_flowcover_radius"],
    )


def _load(case: Case, cfg: dict[str, Any]):
    return load_image(
        case, cfg.get("image_root"), max_patches=cfg["max_image_patches"],
        include_global=cfg["include_global_view"], selection=cfg["patch_selection"],
        tile_single=cfg["tile_single_images"], patch_size=cfg["patch_size"],
        patch_stride=cfg["patch_stride"], global_size=cfg["global_view_size"],
        max_source_pixels=cfg["max_image_pixels"], decode_hf_images=cfg.get("decode_hf_images", False),
    )


def make_record(output: Any, case: Case, cfg: dict[str, Any], images: list[Any], device: str):
    answer, correct, details = score_prediction(case.gt, output.answer, case.options, case.question, case.category)
    extra = _extra_record(output, case, cfg, details)
    metadata = getattr(output, "__dict__", {})
    extra.update({key: metadata.get(key) for key in (
        "algorithm_id", "selector_file", "selector_function", "score_layer", "perturbation_layer",
        "actual_prune_after_layer", "qkv_capture_count", "qkv_projection_replay_count",
        "actual_gather_count", "native_verification", "flops_note",
    )})
    extra["experiment_role"] = {
        "text_cos": "A", "q_flowcover_direct_residual": "B",
        "q_flowcover_perturbation": "C", "q_flowcover_perturbation_nocover": "D",
    }[cfg["prune_method"]]
    extra["score_layer"] = cfg["score_layer"]
    extra["actual_prune_after_layer"] = cfg["actual_prune_after_layer"]
    extra["source_statistics_layer"] = cfg["score_layer"] + int(extra["experiment_role"] in "CD")
    extra["diagnostic_flow_comparator"] = "same_l_plus_1_attention; independent original B uses l attention"
    extra["seed"] = cfg["seed"]
    extra["selector_impl"] = cfg.get("selector_impl", "reference")
    extra["greedy_backend"] = cfg.get("greedy_backend") if extra["selector_impl"] == "optimized" else None
    extra["max_attention_mb"] = cfg.get("max_attention_mb") if extra["selector_impl"] == "optimized" else None
    return case.record(
        answer=answer, seconds=output.seconds, flops=output.flops,
        raw_model_answer=output.answer, result=answer,
        path=_path_value(case.image, cfg.get("image_root")), patch_count=len(images),
        status=details["status"], correct=correct,
        input_tokens=output.input_tokens, output_tokens=output.output_tokens,
        image_tokens=output.image_tokens, original_image_tokens=output.original_image_tokens,
        pruned_image_tokens=output.pruned_image_tokens, retained_tokens=output.retained_tokens,
        retention_ratio=output.retention_ratio, pruning_latency=output.pruning_latency,
        prefill_latency=output.prefill_latency, pruning_seconds=output.pruning_seconds,
        generation_seconds=output.generation_seconds, decode_seconds=output.decode_seconds,
        timing_scope="visual_encoding_to_answer_decode_including_selector_and_indexing",
        selection_budgets=output.selection_budgets, selected_indices=output.selected_indices,
        selection_groups=output.selection_groups, selection_mode=output.selection_mode,
        selection_strategy=output.selection_strategy, selection_layer=output.selection_layer,
        vision_layer_token_counts=output.vision_layer_token_counts,
        vision_merge_unit=output.vision_merge_unit, device=device, stages=list(cfg.get("stages", [])),
        **extra, **{key: value for key, value in details.items() if key != "status"},
    )


def align_comparison_flops(output: Any, runner: Any, cfg: dict[str, Any]) -> None:
    """Use one analytical shape convention for A/B/C/D in this entry only.

    The historical testn output and selector computations are left untouched.
    The shared original text-tower estimate remains partial, explicitly noted.
    """
    from .qwen3_vl_perturbation import Qwen3VLPerturbationRunner

    if cfg["prune_method"] in ("text_cos", "q_flowcover_direct_residual"):
        vision = runner.config.vision_config
        hidden, intermediate, depth = int(vision.hidden_size), int(vision.intermediate_size), int(vision.depth)
        unit = int(vision.spatial_merge_size) ** 2
        raw = [int(t) * int(h) * int(w) for t, h, w in output.image_grid_thw]
        compact = [len(ids) for ids in output.selected_indices]
        before = cfg["score_layer"] * output.original_image_tokens
        after = (depth - cfg["score_layer"]) * output.image_tokens
        legacy_visual = (before + after) * 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
        actual_visual = Qwen3VLPerturbationRunner._visual_shape_flops(
            runner, raw, compact, unit, tuple(runner.vit.deepstack_visual_indexes),
        )
        output.flops = int(output.flops) - int(legacy_visual) + actual_visual
    output.flops_note = (
        "Partial analytical estimate, not measured hardware FLOPs. All four methods in this entry "
        "use actual raw-token full-(l+1)/compact-suffix visual matrix shapes and merger shapes, "
        "plus the shared text-tower estimate and the actually executed selector estimate. "
        "Patch Conv3d, normalization, native softmax/activation, RoPE, some reductions/indexing, "
        "sampling, memory traffic and kernel launches are not fully covered. Diagnostic flow "
        "comparison is included if enabled. Explicit debug verification is separately timed and "
        "has a partial matrix-operation estimate; disable it for production FLOPs comparisons."
    )


def worker(device: str, rows: list[dict[str, Any]], cfg: dict[str, Any], out: Any) -> None:
    try:
        import numpy as np
        import torch

        cfg = {**cfg, "device": device}
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.cuda.manual_seed_all(cfg["seed"])
        print(f"[{device}] loading {cfg['prune_method']}: score L{cfg['score_layer']}; "
              f"full L{cfg['score_layer'] + 1} then one gather", flush=True)
        runner = build_runner(device, cfg)
        cfg["stages"] = runner.stages()
        for index in range(min(int(cfg["warmup_cases"]), len(rows))):
            print(f"[{device}] warmup {index + 1}/{min(cfg['warmup_cases'], len(rows))}", flush=True)
            warm_case = Case(**rows[index])
            warm_images = _load(warm_case, cfg)
            runner.answer(warm_images, warm_case.eval_prompt(), scoring_text=warm_case.question)
            del warm_images
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        for row in rows:
            case = Case(**row)
            try:
                images = _load(case, cfg)
                # Verification is a separately labelled, one-time debug cost.
                # With it disabled, measured work is exactly the selected method.
                if hasattr(runner, "perturbation_verify_candidates"):
                    runner.perturbation_verify_candidates = cfg["verify_candidates"] if case.index == 0 else 0
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                output = runner.answer(images, case.eval_prompt(), scoring_text=case.question)
                align_comparison_flops(output, runner, cfg)
                item = make_record(output, case, cfg, images, device)
                item["first_sample_verification_enabled"] = bool(getattr(runner, "perturbation_verify_candidates", 0))
                if cfg["save_diagnostics"] and cfg["prune_method"].startswith("q_flowcover_perturbation"):
                    from .perturbation_diagnostics import save_perturbation_diagnostics

                    started = time.perf_counter()
                    views = getattr(output, "q_flowcover", getattr(output, "q_flowcover_debug", []))
                    paths = save_perturbation_diagnostics(
                        Path(cfg["output"]).parent / "diagnostics", f"{case.index:06d}", images, views,
                        question=case.question, options=case.options, ground_truth=case.gt,
                        answer=output.answer, extra={
                            "sample_id": case.id, "score_layer": cfg["score_layer"],
                            "perturbation_layer": cfg["score_layer"] + 1,
                            "gather_after_layer": cfg["score_layer"] + 1,
                            "flow_comparison_attention_layer": cfg["score_layer"] + 1,
                            "native_verification": getattr(output, "native_verification", None),
                        },
                    )
                    item["diagnostics"] = paths
                    item["diagnostics_wall_seconds"] = time.perf_counter() - started
                del output, images
            except Exception as exc:
                item = case.record(
                    answer="", seconds=0., flops=0, result="", status="error", correct=False,
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                    sample_id=case.id, prune_method=cfg["prune_method"], device=device,
                    exception_count=1, score_layer=cfg["score_layer"],
                    perturbation_layer=cfg["score_layer"] + 1,
                    actual_prune_after_layer=cfg["score_layer"] + 1,
                )
                print(item["traceback"], flush=True)
            out.put(("row", item))
    except Exception as exc:
        out.put(("worker_error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        out.put(("done", device))
