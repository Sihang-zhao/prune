"""Persistent per-device inference for perturbation pruning and native base."""

from __future__ import annotations

import gc
import random
import traceback
from typing import Any

from .data import Case, load_image
from .flops import text_flops_components, visual_flops_components
from .metrics import score_prediction


def _dtype(requested, device):
    if requested != "auto":
        return requested
    import torch

    return (
        ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
        if device.startswith("cuda")
        else "fp32"
    )


def build_runner(device, cfg):
    from models.base import BaseQwenVL
    from models.qwen3 import Qwen3VLPerturbationRunner

    common = dict(
        name=cfg["model"],
        device=device,
        dtype=_dtype(cfg["dtype"], device),
        max_new_tokens=cfg["max_new_tokens"],
        max_image_pixels=cfg["max_image_pixels"],
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=cfg["offline"],
        enable_thinking=cfg["enable_thinking"],
    )
    if cfg["method"] == "base":
        return BaseQwenVL(**common)
    layer = cfg["layer"]
    method = (
        "q_flowcover_perturbation_nocover"
        if cfg["method"] == "nocover"
        else "q_flowcover_perturbation"
    )
    return Qwen3VLPerturbationRunner(
        **common,
        visual_pruning=True,
        visual_keep_ratio=1 - cfg["prune_percent"] / 100,
        visual_keep_tokens=cfg["keep_tokens"],
        vision_layers=(min(6, layer - 1), layer, cfg["vision_depth"]),
        pruning_mode="middle_text_early",
        pruning_layer=layer,
        pruning_strategy="topk",
        compression_mode="full19_gather",
        prune_method=method,
        decision_layer_human=layer,
        perturbation_lambda=cfg["perturbation_lambda"],
        perturbation_temperature=cfg["temperature"],
        perturbation_query_chunk=cfg["query_chunk_size"],
        perturbation_group_chunk=cfg["group_chunk_size"],
        perturbation_key_chunk=cfg["source_chunk_size"],
        perturbation_local_radius=cfg["local_radius"],
        perturbation_verify_candidates=0,
        perturbation_compare_flow=False,
        perturbation_selector_impl=cfg["implementation"],
        perturbation_max_attention_mb=cfg["max_attention_mb"],
        perturbation_greedy_backend=cfg["greedy_backend"],
    )


def _load(case, cfg):
    return load_image(
        case,
        cfg["image_root"],
        max_patches=cfg["max_image_patches"],
        include_global=cfg["include_global_view"],
        selection=cfg["patch_selection"],
        tile_single=cfg["tile_single_images"],
        patch_size=cfg["patch_size"],
        patch_stride=cfg["patch_stride"],
        global_size=cfg["global_view_size"],
        max_source_pixels=cfg["max_image_pixels"],
        decode_hf_images=cfg["decode_hf_images"],
    )


def make_record(output, case, cfg, model, device):
    base = cfg["method"] == "base"
    grid = getattr(output, "image_grid_thw", None) or getattr(model, "last_image_grid_thw", None)
    if grid is None:
        raise RuntimeError("native processor grid was not captured")
    vision, text = model.config.vision_config, model.config.text_config
    unit = int(vision.spatial_merge_size) ** 2
    raw_counts = [int(t) * int(h) * int(w) for t, h, w in grid]
    original = sum(raw_counts) // unit
    selections = [] if base else output.selected_indices
    keep = [n // unit for n in raw_counts] if base else [len(ids) for ids in selections]
    if sum(keep) != int(output.image_tokens):
        raise RuntimeError("retained group count differs from native LLM visual slots")
    visual = visual_flops_components(
        vision,
        raw_counts,
        keep,
        min(cfg["layer"], int(vision.depth) - 1),
        include_score_merger=not base,
        full_baseline=base,
    )
    llm = text_flops_components(text, output.input_tokens, output.output_tokens)
    selector = 0 if base else int(getattr(output, "selector_flops", 0))
    verify = 0 if base else int(getattr(output, "verification_flops", 0))
    components = dict(
        llm_flops=llm["total"],
        llm_prefill_flops=llm["prefill"],
        llm_decode_flops=llm["decode"],
        vision_linear_flops=visual["linear"],
        vision_attention_flops=visual["attention"],
        score_merger_flops=visual["score_merger"],
        final_merger_flops=visual["final_merger"],
        deepstack_mergers_flops=visual["deepstack_mergers"],
        selector_flops=selector,
        verification_flops=verify,
    )
    components["total_flops"] = llm["total"] + visual["total"] + selector + verify
    answer, correct, details = score_prediction(
        case.gt, output.answer, case.options, case.question, case.category
    )
    record = case.record(
        answer=answer,
        seconds=output.seconds,
        flops=components["total_flops"],
        status=details["status"],
        correct=correct,
        raw_model_answer=output.answer,
        method=cfg["method"],
        implementation=cfg["implementation"],
        device=device,
        dtype=_dtype(cfg["dtype"], device),
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
        original_image_tokens=original,
        retained_tokens=sum(keep),
        image_tokens=output.image_tokens,
        retention_ratio=sum(keep) / original,
        image_grid_thw=grid,
        per_view_group_counts=[n // unit for n in raw_counts],
        flops_accounting="raw_patch_attention_mergers_v1",
        flops_breakdown=components,
        selector_flops=selector,
        pruning_latency=0.0 if base else output.pruning_latency,
        prefill_latency=None if base else output.prefill_latency,
        stage_timings=getattr(output, "stage_timings", {}),
        timing_scope="visual_encoding_to_answer_decode",
        **{key: value for key, value in details.items() if key != "status"},
    )
    if not base:
        record.update(
            selected_indices=selections,
            selection_budgets=output.selection_budgets,
            vision_merge_unit=unit,
            vision_layer_token_counts=output.vision_layer_token_counts,
            selection_layer=cfg["layer"],
        )
        debug = getattr(output, "q_flowcover_debug", [])
        if cfg["diagnostics"] == "summary":
            keys = (
                "algorithm_id",
                "N",
                "K",
                "query_group_count",
                "source_query_count",
                "selector_flops",
                "selector_seconds",
                "perturbation_seconds",
                "greedy_seconds",
                "selected_group_hash",
                "keep_idx_hash",
                "query_chunk_count",
                "cancellation_recomputed_group_ids",
                "stable_recomputed_group_ids",
                "statistics_implementation",
            )
            debug = [{key: view[key] for key in keys if key in view} for view in debug]
        record["pruning_diagnostics"] = debug
    return record


def worker(device: str, rows: list[dict[str, Any]], cfg: dict[str, Any], output_queue):
    try:
        import numpy as np
        import torch

        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(cfg["seed"])
        print(f"[{device}] loading {cfg['method']}: {cfg['model']}", flush=True)
        model = build_runner(device, cfg)
        for position, raw_case in enumerate(rows):
            case = Case(**raw_case)
            images = output = None
            try:
                images = _load(case, cfg)
                kwargs = {} if cfg["method"] == "base" else {"scoring_text": case.question}
                if position < cfg["warmup_cases"]:
                    model.answer(images, case.eval_prompt(), **kwargs)
                if hasattr(model, "perturbation_verify_candidates"):
                    model.perturbation_verify_candidates = (
                        cfg["verify_candidates"] if position == 0 else 0
                    )
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                output = model.answer(images, case.eval_prompt(), **kwargs)
                record = make_record(output, case, cfg, model, device)
                if device.startswith("cuda"):
                    record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            except Exception as error:
                record = case.record(
                    answer="",
                    seconds=0,
                    flops=0,
                    status="error",
                    correct=False,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                    device=device,
                )
            output_queue.put(("row", record))
            images = output = None
            if (position + 1) % 64 == 0:
                gc.collect()
    except Exception as error:
        output_queue.put(
            ("worker_error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}")
        )
    finally:
        output_queue.put(("done", device))
