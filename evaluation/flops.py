"""Comparable, partial operation estimates for full and pruned Qwen3-VL.

These are architecture estimates, not hardware profiler measurements. The
text estimate deliberately retains the repository's historical convention;
the visual estimate counts raw patch rows, QK/AV and native merger shapes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def text_flops_components(
    text_config: Any, input_tokens: int, output_tokens: int
) -> dict[str, int]:
    """Use exactly the existing estimator's text-tower convention."""
    layers = int(_get(text_config, "num_hidden_layers", 0))
    hidden = int(_get(text_config, "hidden_size", 0))
    intermediate = int(_get(text_config, "intermediate_size", 0))
    seq, new = max(1, int(input_tokens)), max(0, int(output_tokens))
    per_token = 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
    prefill = layers * seq * (per_token + 2 * hidden * seq)
    decode = layers * new * (per_token + 4 * hidden * seq)
    return {"prefill": prefill, "decode": decode, "total": prefill + decode}


def visual_flops_components(
    vision_config: Any,
    raw_counts: Sequence[int],
    keep_counts: Sequence[int],
    score_layer: int,
    deepstack_indexes: Sequence[int] | None = None,
    include_score_merger: bool = True,
    *,
    full_baseline: bool = False,
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
    indexes = tuple(
        int(n)
        for n in (
            deepstack_indexes
            if deepstack_indexes is not None
            else _get(vision_config, "deepstack_visual_indexes", ())
        )
    )
    if any(n < 0 or n >= depth for n in indexes):
        raise ValueError("DeepStack indices must lie within the native visual blocks")
    full_layers = depth if full_baseline else int(score_layer) + 1
    compact = raw if full_baseline else [n * unit for n in keep]
    per_token = 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
    linear = (full_layers * sum(raw) + (depth - full_layers) * sum(compact)) * per_token
    attention = (
        4
        * hidden
        * (
            full_layers * sum(n * n for n in raw)
            + (depth - full_layers) * sum(n * n for n in compact)
        )
    )
    group_width = hidden * unit
    merger_row = 2 * group_width * (group_width + out_hidden)
    original_groups = sum(raw) // unit
    final_groups = original_groups if full_baseline else sum(keep)
    score_merger = original_groups * merger_row if include_score_merger else 0
    final_merger = final_groups * merger_row
    deepstack_mergers = sum(
        (original_groups if n < full_layers else final_groups) * merger_row for n in indexes
    )
    result = dict(
        linear=linear,
        attention=attention,
        score_merger=score_merger,
        final_merger=final_merger,
        deepstack_mergers=deepstack_mergers,
    )
    result["total"] = sum(result.values())
    return result


__all__ = ["text_flops_components", "visual_flops_components"]
