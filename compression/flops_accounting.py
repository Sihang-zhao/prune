"""Comparable, partial operation estimates for full and pruned Qwen3-VL.

These are architecture estimates, not hardware profiler measurements. The
text estimate deliberately retains the repository's historical convention;
the visual estimate counts raw patch rows, QK/AV and native merger shapes.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def text_flops_components(text_config: Any, input_tokens: int, output_tokens: int) -> dict[str, int]:
    """Use exactly the existing estimator's text-tower convention."""
    layers = int(_get(text_config, "num_hidden_layers", 0))
    hidden = int(_get(text_config, "hidden_size", 0))
    intermediate = int(_get(text_config, "intermediate_size", 0))
    seq, new = max(1, int(input_tokens)), max(0, int(output_tokens))
    per_token = 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
    prefill = layers * seq * (per_token + 2 * hidden * seq)
    decode = layers * new * (per_token + 4 * hidden * seq)
    return {"prefill": prefill, "decode": decode, "total": prefill + decode}


def legacy_visual_flops(vision_config: Any, original_groups: int, retained_groups: int,
                        score_layer: int | None = None) -> int:
    """Historical merged-group linear-only visual estimate, for comparison."""
    depth = int(_get(vision_config, "depth", _get(vision_config, "num_hidden_layers", 0)))
    hidden = int(_get(vision_config, "hidden_size", 0))
    intermediate = int(_get(vision_config, "intermediate_size", 0))
    if score_layer is None:
        token_layers = depth * int(original_groups)
    else:
        layer = int(score_layer)
        if not 0 < layer < depth:
            raise ValueError("score_layer must precede the final visual block")
        token_layers = layer * int(original_groups) + (depth - layer) * int(retained_groups)
    return token_layers * 2 * (4 * hidden * hidden + 2 * hidden * intermediate)


def visual_flops_components(
    vision_config: Any, raw_counts: Sequence[int], keep_counts: Sequence[int],
    score_layer: int, deepstack_indexes: Sequence[int] | None = None,
    include_score_merger: bool = True, *, full_baseline: bool = False,
) -> dict[str, int]:
    """Count native raw-token visual matrix operations.

    ``keep_counts`` are merged-group counts, while ``raw_counts`` are patch
    rows per independent image view. The perturbation entry gathers after
    score layer L+1. A full baseline runs every block at full length and has
    no extra score merger (pass ``include_score_merger=False`` explicitly).
    """
    depth = int(_get(vision_config, "depth", _get(vision_config, "num_hidden_layers", 0)))
    hidden = int(_get(vision_config, "hidden_size", 0))
    intermediate = int(_get(vision_config, "intermediate_size", 0))
    out_hidden = int(_get(vision_config, "out_hidden_size", 0))
    unit = int(_get(vision_config, "spatial_merge_size", 1)) ** 2
    raw, keep = [int(n) for n in raw_counts], [int(n) for n in keep_counts]
    if not raw or len(raw) != len(keep) or any(n <= 0 or n % unit for n in raw):
        raise ValueError("raw_counts must describe nonempty complete-group views")
    if any(k < 0 or k * unit > n for n, k in zip(raw, keep)):
        raise ValueError("keep_counts must fit the original per-view group counts")
    if not 0 < int(score_layer) < depth:
        raise ValueError("score_layer must precede the final visual block")
    indexes = tuple(int(n) for n in (deepstack_indexes if deepstack_indexes is not None
                                     else _get(vision_config, "deepstack_visual_indexes", ())))
    if any(n < 0 or n >= depth for n in indexes):
        raise ValueError("DeepStack indices must lie within the native visual blocks")
    full_layers = depth if full_baseline else int(score_layer) + 1
    compact = raw if full_baseline else [n * unit for n in keep]
    per_token = 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
    linear = (full_layers * sum(raw) + (depth - full_layers) * sum(compact)) * per_token
    attention = 4 * hidden * (full_layers * sum(n * n for n in raw)
                              + (depth - full_layers) * sum(n * n for n in compact))
    group_width = hidden * unit
    merger_row = 2 * group_width * (group_width + out_hidden)
    original_groups = sum(raw) // unit
    final_groups = original_groups if full_baseline else sum(keep)
    score_merger = original_groups * merger_row if include_score_merger else 0
    final_merger = final_groups * merger_row
    deepstack_mergers = sum((original_groups if n < full_layers else final_groups) * merger_row
                           for n in indexes)
    result = dict(linear=linear, attention=attention, score_merger=score_merger,
                  final_merger=final_merger, deepstack_mergers=deepstack_mergers)
    result["total"] = sum(result.values())
    return result


def decompose_record(model_config: Any, row: Mapping[str, Any], *,
                     score_layer: int | None = None) -> dict[str, int | float]:
    """Reconstruct a perturbation row and its controlled full-input baseline.

    Full-baseline FLOPs are counterfactual: restore visual prompt slots while
    holding text tokens and generated length fixed. No baseline answer,
    timing or accuracy is inferred. Changing score_layer also represents a
    budget-only counterfactual and holds the recorded selector estimate fixed.
    """
    vision, text = _get(model_config, "vision_config"), _get(model_config, "text_config", model_config)
    layer = int(score_layer if score_layer is not None else row.get("score_layer", row.get("decision_layer_human", 18)))
    grids = row.get("image_grid_thw")
    if not grids:
        raise ValueError("row has no image_grid_thw; cannot infer native per-view lengths")
    raw = [int(t) * int(h) * int(w) for t, h, w in grids]
    indices = row.get("selected_indices")
    if isinstance(indices, list) and len(indices) == len(raw):
        keep = [len(value) for value in indices]
    elif len(raw) == 1:
        keep = [int(row["retained_tokens"])]
    else:
        raise ValueError("multi-view row has no per-view selected_indices")
    unit = int(_get(vision, "spatial_merge_size", 1)) ** 2
    original, retained = sum(raw) // unit, sum(keep)
    input_tokens, output_tokens = int(row["input_tokens"]), int(row["output_tokens"])
    llm = text_flops_components(text, input_tokens, output_tokens)
    visual = visual_flops_components(vision, raw, keep, layer)
    selector = int(row.get("selector_flops", 0) or 0)
    verification = int(row.get("verification_flops", 0) or 0)
    total = llm["total"] + visual["total"] + selector + verification
    full_input = input_tokens + original - retained
    full_llm = text_flops_components(text, full_input, output_tokens)["total"]
    full_visual = visual_flops_components(vision, raw, keep, layer,
                                          include_score_merger=False, full_baseline=True)
    full_total = full_llm + full_visual["total"]
    recorded = int(row.get("flops", total))
    return {
        "llm_prefill_flops": llm["prefill"], "llm_decode_flops": llm["decode"],
        "llm_flops": llm["total"], "vision_linear_flops": visual["linear"],
        "vision_attention_flops": visual["attention"], "score_merger_flops": visual["score_merger"],
        "final_merger_flops": visual["final_merger"], "deepstack_mergers_flops": visual["deepstack_mergers"],
        "visual_flops": visual["total"], "selector_flops": selector, "verification_flops": verification,
        "estimated_total_flops": total, "recorded_total_flops": recorded,
        "reconstruction_delta_flops": recorded - total,
        "legacy_pruned_flops": llm["total"] + legacy_visual_flops(vision, original, retained, layer),
        "legacy_full_flops": full_llm + legacy_visual_flops(vision, original, original),
        "unified_full_flops": full_total,
        "unified_reduction_fraction": 1 - total / full_total if full_total else 0.0,
        "full_baseline_input_tokens": full_input,
    }


__all__ = ["text_flops_components", "legacy_visual_flops", "visual_flops_components", "decompose_record"]
