"""Persistent native Qwen3-VL worker; no selection or scoring projection."""
from __future__ import annotations

import gc
import sys
from typing import Any, Mapping, Sequence

from model.qwen_vl import QwenVL
from utils.data import Case, load_image
from utils.metrics import score_prediction
from .flops_accounting import text_flops_components, visual_flops_components


ACCOUNTING_ID = "qwen3_vl_raw_tokens_native_mergers_v1"
FLOPS_NOTE = (
    "Partial analytical estimate, not measured hardware FLOPs. Same current convention as the "
    "perturbation comparison: full native raw-patch visual blocks, per-view QK/AV, final and "
    "DeepStack merger shapes, plus the shared historical text-tower estimate. No score merger, "
    "selector, deletion statistics or verification pass is executed. Patch Conv3d, normalization, "
    "softmax/activation, RoPE, some reductions/indexing, sampling, memory traffic and kernel "
    "launches are not fully covered."
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


class BaseQwenVL(QwenVL):
    """The plain inference wrapper with processor-grid metadata capture only."""

    def _inputs(self, image: Any, prompt: str) -> Mapping[str, Any]:
        inputs = super()._inputs(image, prompt)
        grid = inputs.get("image_grid_thw")
        if grid is None:
            raise ValueError("Native baseline processor produced no image_grid_thw")
        self.last_image_grid_thw = grid.detach().reshape(-1, 3).cpu().tolist()
        return inputs


def current_flops_breakdown(config: Any, input_tokens: int, output_tokens: int,
                            image_grid_thw: Sequence[Sequence[int]]) -> dict[str, int]:
    vision = _get(config, "vision_config")
    text = _get(config, "text_config", config)
    raw = [int(t) * int(h) * int(w) for t, h, w in image_grid_thw]
    unit = int(_get(vision, "spatial_merge_size", 1)) ** 2
    depth = int(_get(vision, "depth", _get(vision, "num_hidden_layers", 0)))
    # This nominal boundary is unused by full_baseline=True; it only meets
    # the common accounting helper's input contract.
    boundary = min(18, depth - 1)
    visual = visual_flops_components(
        vision, raw, [n // unit for n in raw], boundary,
        include_score_merger=False, full_baseline=True,
    )
    llm = text_flops_components(text, input_tokens, output_tokens)
    return {
        "llm_prefill_flops": llm["prefill"], "llm_decode_flops": llm["decode"],
        "llm_flops": llm["total"], "vision_linear_flops": visual["linear"],
        "vision_attention_flops": visual["attention"], "score_merger_flops": 0,
        "final_merger_flops": visual["final_merger"],
        "deepstack_mergers_flops": visual["deepstack_mergers"],
        "visual_flops": visual["total"], "selector_flops": 0, "verification_flops": 0,
        "estimated_total_flops": llm["total"] + visual["total"],
    }


def _load(case: Case, cfg: Mapping[str, Any]):
    return load_image(
        case, cfg.get("image_root"), max_patches=cfg["max_image_patches"],
        include_global=cfg["include_global_view"], selection=cfg["patch_selection"],
        tile_single=cfg["tile_single_images"], patch_size=cfg["patch_size"],
        patch_stride=cfg["patch_stride"], global_size=cfg["global_view_size"],
        max_source_pixels=cfg["max_image_pixels"], decode_hf_images=cfg.get("decode_hf_images", False),
    )


def worker(device: str, rows: list[dict[str, Any]], cfg: dict[str, Any], out: Any) -> None:
    try:
        import torch

        dtype = str(cfg.get("dtype", "bf16"))
        if dtype == "auto":
            dtype = "bf16" if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else (
                "fp16" if str(device).startswith("cuda") else "fp32")
        print(f"[base] loading native Qwen3-VL: {cfg['model']} device={device} dtype={dtype}; no pruning", file=sys.stderr, flush=True)
        runner = BaseQwenVL(
            name=cfg["model"], device=device, dtype=dtype,
            max_new_tokens=cfg["max_new_tokens"], trust_remote_code=cfg["trust_remote_code"],
            local_files_only=cfg["offline"], max_image_pixels=cfg["max_image_pixels"],
            enable_thinking=cfg["enable_thinking"],
        )
        vision = _get(runner.config, "vision_config")
        depth = int(_get(vision, "depth", _get(vision, "num_hidden_layers", 0)))
        unit = int(_get(vision, "spatial_merge_size", 1)) ** 2
        warmup_cases = int(cfg.get("warmup_cases", 1))
        for position, row in enumerate(rows):
            case, images, output = Case(**row), None, None
            try:
                images = _load(case, cfg)
                if position < warmup_cases:
                    runner.answer(images, case.eval_prompt())
                    print(f"[base] warmup {position + 1}/{min(warmup_cases, len(rows))} completed; excluded from case timing", file=sys.stderr, flush=True)
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                output = runner.answer(images, case.eval_prompt())
                grids = runner.last_image_grid_thw
                components = current_flops_breakdown(runner.config, output.input_tokens, output.output_tokens, grids)
                raw_counts = [int(t) * int(h) * int(w) for t, h, w in grids]
                original_groups = sum(raw_counts) // unit
                if output.image_tokens != original_groups or output.original_image_tokens != original_groups:
                    raise RuntimeError("native baseline did not retain every original visual group")
                answer, correct, details = score_prediction(case.gt, output.answer, case.options, case.question, case.category)
                peak = int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else None
                item = case.record(
                    answer=answer, seconds=output.seconds, flops=components["estimated_total_flops"],
                    raw_model_answer=output.answer, result=answer, status=details["status"], correct=correct,
                    sample_id=case.id, task_category=case.category, dataset_split=cfg.get("split"),
                    model_revision=cfg["model"], device=device, dtype=dtype,
                    evaluation_variant="base", prune_method="qwen3_base", compression_mode="off",
                    visual_pruning=False, visual_prune_percent=0.0,
                    input_tokens=output.input_tokens, output_tokens=output.output_tokens,
                    image_tokens=original_groups, original_image_tokens=original_groups,
                    pruned_image_tokens=original_groups, retained_tokens=original_groups,
                    retention_ratio=1.0, pruning_latency=0.0, pruning_seconds=0.0,
                    prefill_latency=None, prefill_time_measured=False,
                    generation_seconds=output.generation_seconds, decode_seconds=output.decode_seconds,
                    total_wall_time=output.seconds, timing_scope="visual_encoding_to_answer_text_decode",
                    timing_note="Native generate includes visual encoding and language generation; tokenizer answer decoding is separate. Prefill/decode token times are not separately measured.",
                    image_grid_thw=grids, raw_patch_count=sum(raw_counts), merged_group_count=original_groups,
                    retained_group_count=original_groups, actual_llm_visual_tokens=original_groups,
                    vision_merge_unit=unit, vision_layer_token_counts=[sum(raw_counts)] * depth,
                    per_view_raw_patch_counts=raw_counts, per_view_group_counts=[n // unit for n in raw_counts],
                    peak_memory=peak, patch_count=len(images), stages=list(runner.stages()),
                    flops_accounting=ACCOUNTING_ID, flops_note=FLOPS_NOTE,
                    flops_breakdown=components, legacy_estimated_flops=output.flops,
                    selector_flops=0, verification_flops=0, full_baseline=True,
                    **{key: value for key, value in details.items() if key != "status"},
                )
            except Exception as exc:
                item = case.record(
                    answer="", seconds=0.0, flops=0, result="", status="error", correct=False,
                    error=f"{type(exc).__name__}: {exc}", sample_id=case.id, task_category=case.category,
                    device=device, dtype=dtype, evaluation_variant="base", prune_method="qwen3_base",
                    compression_mode="off", flops_accounting=ACCOUNTING_ID,
                )
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            out.put(("row", item))
            images = output = None
            if (position + 1) % 64 == 0:
                gc.collect()
    except Exception as exc:
        out.put(("worker_error", f"{type(exc).__name__}: {exc}"))
    finally:
        gc.collect()
        out.put(("done", device))


__all__ = ["BaseQwenVL", "current_flops_breakdown", "worker", "ACCOUNTING_ID", "FLOPS_NOTE"]
