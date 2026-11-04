"""Reference fixed-query deletion statistics and exact local coverage."""

from __future__ import annotations

import hashlib
import math
import time
from functools import lru_cache
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .common import SelectionResult, _stable_topk, _text_cosine


PERTURBATION_METHODS = {
    "q_flowcover_perturbation",
    "q_flowcover_perturbation_nocover",
}


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")


def _mask_slice(
    allowed: torch.Tensor | None,
    query_ids: torch.Tensor,
    key_ids: torch.Tensor,
    heads: int,
) -> torch.Tensor | None:
    if allowed is None:
        return None
    if allowed.ndim == 1:
        return allowed.index_select(0, key_ids)[None, None, :]
    if allowed.ndim == 2:
        return allowed.index_select(0, query_ids).index_select(1, key_ids)[None]
    mask = allowed.index_select(1, query_ids).index_select(2, key_ids)
    if int(mask.shape[0]) == 1:
        return mask
    return mask


def _validate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_groups: torch.Tensor,
    query_weights: torch.Tensor,
    merge_unit: int,
    allowed_mask: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if any(x.ndim != 3 for x in (q, k, v)):
        raise ValueError("native Q/K/V must each have shape [raw_tokens,heads,head_dim]")
    raw, heads, dim = map(int, q.shape)
    unit = int(merge_unit)
    if unit < 1 or raw < 1 or raw % unit:
        raise ValueError("raw Q count must be a positive multiple of merge_unit")
    if k.shape != v.shape or int(k.shape[0]) != raw or int(k.shape[2]) != dim:
        raise ValueError("K/V shape or raw token/head dimension does not match Q")
    if int(k.shape[1]) < 1 or heads % int(k.shape[1]):
        raise ValueError("Q heads must be an integer multiple of KV heads (GQA/MQA)")
    if any(x.device != q.device for x in (k, v, query_groups, query_weights)):
        raise ValueError("Q/K/V, query IDs and weights must share a device")
    if query_groups.ndim != 1 or query_groups.dtype != torch.long or not query_groups.numel():
        raise ValueError("query_groups must be a nonempty int64 vector")
    groups = raw // unit
    if bool(((query_groups < 0) | (query_groups >= groups)).any()):
        raise ValueError("query_groups contains invalid original group IDs")
    if query_groups.unique().numel() != query_groups.numel():
        raise ValueError("query_groups contains repeated group IDs")
    if query_weights.shape != (groups,):
        raise ValueError("query_weights must be the full group-length pi vector")
    _finite("Q", q)
    _finite("K", k)
    _finite("V", v)
    _finite("pi", query_weights)
    if bool((query_weights < 0).any()):
        raise ValueError("query weights must be nonnegative")
    outside = torch.ones(groups, dtype=torch.bool, device=q.device)
    outside.index_fill_(0, query_groups, False)
    if bool((query_weights.masked_select(outside) != 0).any()):
        raise ValueError("query weights outside query_groups must be zero")
    if not torch.isclose(query_weights.sum(), query_weights.new_tensor(1.0), atol=2e-6, rtol=2e-6):
        raise ValueError("query weights must sum to one")
    if allowed_mask is not None:
        if allowed_mask.dtype != torch.bool:
            raise TypeError(
                "allowed_mask must be bool: True means an allowed source; floating masks are rejected"
            )
        if allowed_mask.device != q.device:
            raise ValueError("allowed_mask must be on the Q device")
        shape = tuple(allowed_mask.shape)
        if not (
            shape == (raw,)
            or shape == (raw, raw)
            or shape == (1, raw, raw)
            or shape == (heads, raw, raw)
        ):
            raise ValueError(f"unsupported bool allowed_mask shape {shape}")
        raw_queries = (query_groups[:, None] * unit + torch.arange(unit, device=q.device)).flatten()
        if allowed_mask.ndim == 1:
            allowed_rows = allowed_mask.any().reshape(1)
        elif allowed_mask.ndim == 2:
            allowed_rows = allowed_mask.index_select(0, raw_queries).any(-1)
        else:
            allowed_rows = allowed_mask.index_select(1, raw_queries).any(-1)
        if not bool(allowed_rows.all()):
            raise ValueError("attention has a fully masked Query")
    return raw, heads, dim, groups


def _readout_statistics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    scale: float,
    source_chunk_size: int,
    allowed_mask: torch.Tensor | None,
    excluded_raw: torch.Tensor | None = None,
    validate_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable global-source log normalizer and readout, with online softmax.

    Inputs use [H,C,D]/[H,T,D].  Source chunks share the *same* accumulated
    normalizer.  At most [H,C,source_chunk] attention logits are retained.
    """
    heads, count, dim = query.shape
    normalizer_max = torch.full((heads, count), -float("inf"), device=query.device)
    normalizer_sum = torch.zeros_like(normalizer_max)
    numerator = torch.zeros((heads, count, dim), dtype=torch.float32, device=query.device)
    raw = int(key.shape[1])
    for start in range(0, raw, int(source_chunk_size)):
        end = min(raw, start + int(source_chunk_size))
        ids = torch.arange(start, end, device=query.device)
        logits = torch.bmm(query, key[:, start:end].transpose(1, 2)) * float(scale)
        allowed = _mask_slice(allowed_mask, query_ids, ids, heads)
        if allowed is not None:
            logits.masked_fill_(~allowed, -float("inf"))
        if excluded_raw is not None:
            logits.masked_fill_(torch.isin(ids, excluded_raw)[None, None], -float("inf"))
        next_max = torch.maximum(normalizer_max, logits.amax(dim=-1))
        # A completely masked *chunk* is legal; a completely masked complete
        # query is rejected below after all chunks have been scanned.
        safe_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
        old_weight = torch.exp(normalizer_max - safe_max)
        probability_numerator = torch.exp(logits - safe_max[..., None])
        normalizer_sum = normalizer_sum * old_weight + probability_numerator.sum(-1)
        numerator = numerator * old_weight[..., None] + torch.bmm(
            probability_numerator, value[:, start:end]
        )
        normalizer_max = next_max
    # Normal-path mask legality is validated once before chunking.  Avoid
    # synchronizing the GPU with a Python bool on every Query chunk.  Rare
    # explicit-deletion rescans additionally check the new source mask here.
    if validate_output and (
        bool((normalizer_sum <= 0).any()) or not bool(torch.isfinite(normalizer_sum).all())
    ):
        raise ValueError(
            "attention has a fully masked Query or deletion removes every legal source"
        )
    output = numerator / normalizer_sum[..., None]
    log_normalizer = normalizer_max + normalizer_sum.log()
    return log_normalizer, output


@torch.no_grad()
def group_deletion_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_groups: torch.Tensor,
    query_weights: torch.Tensor,
    merge_unit: int,
    *,
    scale: float,
    query_chunk_size: int = 256,
    group_chunk_size: int = 32,
    source_chunk_size: int = 256,
    allowed_mask: torch.Tensor | None = None,
    near_one_threshold: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return ``D[group]`` and flow support from one shared attention scan.

    Softmax/statistics use FP32.  Native projected/rotated Q/K/V remain intact;
    no learned projections or model calls are made here.  Each candidate is
    deleted independently from the same complete per-view source sequence.
    Query groups are fixed (including when a candidate itself is in P).
    Their errors are diagnosed but receive zero residual weight downstream.

    A first key scan obtains the global-source normalizer and readout.  A
    second, group-chunked scan immediately reduces m/b to scalar group scores
    and support.  No full N*N attention or all-query/all-group hidden tensor
    is materialized.  If 1-m is too small for safe subtraction, that candidate
    is recomputed with an explicit source mask and global stable softmax.
    """
    raw, heads, dim, groups = _validate_attention(
        q, k, v, query_groups, query_weights, merge_unit, allowed_mask
    )
    unit = int(merge_unit)
    for name, chunk in (
        ("query_chunk_size", query_chunk_size),
        ("group_chunk_size", group_chunk_size),
        ("source_chunk_size", source_chunk_size),
    ):
        if int(chunk) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("attention scale must be positive and finite")
    if not 0 < float(near_one_threshold) < 1:
        raise ValueError("near_one_threshold must be in (0,1)")
    if groups <= 1:
        # No valid independent full-minus-one attention exists in this case.
        # The full-budget selector returns before requesting statistics.
        raise ValueError("deletion statistics need at least two source groups")
    raw_queries = (query_groups[:, None] * unit + torch.arange(unit, device=q.device)).flatten()
    weights = query_weights.index_select(0, query_groups).float().repeat_interleave(unit) / float(
        unit
    )
    q_heads = q.transpose(0, 1)
    kv_repeats = heads // int(k.shape[1])
    k_heads = k.transpose(0, 1).float()
    v_heads = v.transpose(0, 1).float()
    if kv_repeats != 1:
        k_heads = k_heads.repeat_interleave(kv_repeats, dim=0)
        v_heads = v_heads.repeat_interleave(kv_repeats, dim=0)
    scores = torch.zeros(groups, dtype=torch.float32, device=q.device)
    support = torch.zeros_like(scores)
    unstable = torch.zeros(groups, dtype=torch.bool, device=q.device)
    # Explicitly cap Query chunks below the complete raw sequence.  Even an
    # all-group P must not accidentally request an N*N attention temporary.
    query_chunk = min(int(query_chunk_size), max(1, raw - 1))
    max_logits = [heads, 0, 0]
    for q_start in range(0, int(raw_queries.numel()), query_chunk):
        query_ids = raw_queries[q_start : q_start + query_chunk]
        local_q = q_heads.index_select(1, query_ids).float()
        lse, full_output = _readout_statistics(
            local_q,
            k_heads,
            v_heads,
            query_ids,
            scale=scale,
            source_chunk_size=source_chunk_size,
            allowed_mask=allowed_mask,
        )
        local_weight = weights[q_start : q_start + int(query_ids.numel())]
        for g_start in range(0, groups, int(group_chunk_size)):
            g_end = min(groups, g_start + int(group_chunk_size))
            group_count = g_end - g_start
            key_ids = torch.arange(g_start * unit, g_end * unit, device=q.device)
            logits = torch.bmm(
                local_q, k_heads[:, g_start * unit : g_end * unit].transpose(1, 2)
            ) * float(scale)
            allowed = _mask_slice(allowed_mask, query_ids, key_ids, heads)
            if allowed is not None:
                logits.masked_fill_(~allowed, -float("inf"))
            attention = torch.exp(logits - lse[..., None]).reshape(heads, -1, group_count, unit)
            mass = attention.sum(-1)
            grouped_value = v_heads[:, g_start * unit : g_end * unit].reshape(
                heads, group_count, unit, dim
            )
            source_readout = torch.einsum("hcbu,hbud->hcbd", attention, grouped_value)
            remaining = 1.0 - mass
            safe = remaining > float(near_one_threshold)
            # Invalid entries are explicitly deferred to stable source-mask
            # recomputation; no arbitrary epsilon can inflate their scores.
            denominator = torch.where(safe, remaining, torch.ones_like(remaining))
            error = (source_readout - mass[..., None] * full_output[:, :, None]) / denominator[
                ..., None
            ]
            error.masked_fill_(~safe[..., None], 0.0)
            local_score = error.square().sum(-1).mean(0)
            scores[g_start:g_end] += (local_score * local_weight[:, None]).sum(0)
            support[g_start:g_end] += (mass.mean(0) * local_weight[:, None]).sum(0)
            unstable[g_start:g_end] |= (~safe).any(dim=(0, 1))
            max_logits[1] = max(max_logits[1], int(query_ids.numel()))
            max_logits[2] = max(
                max_logits[2], int(key_ids.numel()), min(raw, int(source_chunk_size))
            )
    # One diagnostics transfer, outside the normal GPU chunk loop.  Rare
    # pathological groups are explicitly re-read with their source mask.
    unstable_groups = unstable.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    for group in unstable_groups:
        excluded = torch.arange(group * unit, (group + 1) * unit, device=q.device)
        score = torch.zeros((), dtype=torch.float32, device=q.device)
        for q_start in range(0, int(raw_queries.numel()), query_chunk):
            query_ids = raw_queries[q_start : q_start + query_chunk]
            local_q = q_heads.index_select(1, query_ids).float()
            _, full_output = _readout_statistics(
                local_q,
                k_heads,
                v_heads,
                query_ids,
                scale=scale,
                source_chunk_size=source_chunk_size,
                allowed_mask=allowed_mask,
            )
            _, deleted_output = _readout_statistics(
                local_q,
                k_heads,
                v_heads,
                query_ids,
                scale=scale,
                source_chunk_size=source_chunk_size,
                allowed_mask=allowed_mask,
                excluded_raw=excluded,
                validate_output=True,
            )
            error = (full_output - deleted_output).square().sum(-1).mean(0)
            score += (error * weights[q_start : q_start + int(query_ids.numel())]).sum()
        scores[group] = score
    _finite("deletion scores", scores)
    _finite("support", support)
    if not torch.isclose(support.sum(), support.new_tensor(1.0), atol=2e-5, rtol=2e-5):
        raise RuntimeError("shared attention source support does not sum to one")
    raw_query_count = int(raw_queries.numel())
    qk_flops = 4 * heads * raw_query_count * raw * dim  # two source scans
    readout_flops = 4 * heads * raw_query_count * raw * dim
    elementwise_flops = heads * raw_query_count * (9 * raw + groups * (7 * dim + 10))
    fallback_flops = len(unstable_groups) * heads * raw_query_count * raw * (8 * dim + 18)
    return (
        scores,
        support,
        {
            "source_query_count": raw_query_count,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "kv_repeat_count": kv_repeats,
            "max_attention_chunk_shape": max_logits,
            "full_attention_matrix_saved": False,
            "stable_recomputed_group_ids": unstable_groups,
            "attention_source_raw_count": raw,
            "statistics_dtype": "float32",
            "error_definition": "pi-weighted mean raw-query mean-head squared pre-output-projection readout difference",
            "attention_flops_estimated": int(
                qk_flops + readout_flops + elementwise_flops + fallback_flops
            ),
            "flops_note": "Partial analytical scalar-operation estimate, not measured hardware FLOPs. Includes two QK scans, full and group AV, estimated softmax/statistics and rare stable rescan. Excludes some normalization/reduction/index comparisons, memory traffic, indexing, kernel launches and diagnostics.",
        },
    )


@lru_cache(maxsize=64)
def _neighbor_graph(
    height: int, width: int, radius: int, device_string: str
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_string)
    count = int(height) * int(width)
    ids = torch.arange(count, device=device)
    ys, xs = ids.div(int(width), rounding_mode="floor"), ids.remainder(int(width))
    offsets = torch.cartesian_prod(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
    )
    y = ys[:, None] + offsets[:, 0]
    x = xs[:, None] + offsets[:, 1]
    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    return (y * width + x).clamp(0, count - 1), valid


def sparse_local_affinity(
    z: torch.Tensor, height: int, width: int, radius: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Candidate-row/source-column sparse ordinary nonnegative cosine."""
    if z.ndim != 2 or height * width != int(z.shape[0]) or radius < 0:
        raise ValueError("features/grid/local_radius do not describe one regular group view")
    z = F.normalize(z.float(), dim=-1)
    table, valid = _neighbor_graph(height, width, radius, str(z.device))
    affinity = torch.empty(table.shape, dtype=torch.float32, device=z.device)
    # A fixed (2R+1)^2 offset loop, never a per-token/candidate Python loop.
    # It avoids allocating [N,neighbours,feature_dim].
    for offset in range(int(table.shape[1])):
        affinity[:, offset] = (z * z.index_select(0, table[:, offset])).sum(-1).clamp_min(0)
    affinity.masked_fill_(~valid, 0.0)
    ids = torch.arange(int(z.shape[0]), device=z.device)
    affinity.masked_fill_(valid & (table == ids[:, None]), 1.0)
    return affinity, table, valid


def greedy_direct_cover(
    text_mass: torch.Tensor,
    residual_mass: torch.Tensor,
    keep: int,
    affinity: torch.Tensor,
    table: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The historical direct objective, with 1-D device-side selected IDs.

    This is exact sequential greedy from the empty set.  It is not a static
    score Top-K.  Ties select the smallest original group ID (torch.argmax).
    """
    groups = int(text_mass.numel())
    if not 0 < int(keep) <= groups:
        raise ValueError("keep must be in [1,groups]")
    selected = torch.zeros(groups, dtype=torch.bool, device=text_mass.device)
    coverage = torch.zeros_like(text_mass)
    chosen = torch.empty(int(keep), dtype=torch.long, device=text_mass.device)
    gains = torch.empty(int(keep), dtype=torch.float32, device=text_mass.device)
    source_weight = residual_mass.index_select(0, table.flatten()).reshape(table.shape)
    for step in range(int(keep)):
        current = coverage.index_select(0, table.flatten()).reshape(table.shape)
        gain = text_mass + (source_weight * torch.relu(affinity - current)).sum(-1)
        gain.masked_fill_(selected, -float("inf"))
        best = gain.argmax().reshape(1)
        chosen[step : step + 1] = best
        gains[step : step + 1] = gain.index_select(0, best)
        selected.scatter_(0, best, True)
        source = table.index_select(0, best).flatten()
        value = affinity.index_select(0, best).flatten()
        permitted = valid.index_select(0, best).flatten()
        source = source.masked_fill(~permitted, 0)
        value = value.masked_fill(~permitted, -float("inf"))
        coverage.scatter_reduce_(0, source, value, reduce="amax", include_self=True)
    return chosen.sort().values, gains


class _Stages:
    """CUDA event stages resolved once, avoiding per-stage synchronizations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.rows: dict[str, Any] = {}

    def begin(self, name: str) -> None:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.rows[name] = [event, None]
        else:
            self.rows[name] = [time.perf_counter(), None]

    def end(self, name: str) -> None:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.rows[name][1] = event
        else:
            self.rows[name][1] = time.perf_counter()

    def resolve(self) -> dict[str, float]:
        if self.device.type == "cuda":
            # A single synchronization reads all completed stage events.
            torch.cuda.current_stream(self.device).synchronize()
            return {
                name + "_seconds": float(pair[0].elapsed_time(pair[1])) / 1000
                for name, pair in self.rows.items()
            }
        return {name + "_seconds": float(pair[1] - pair[0]) for name, pair in self.rows.items()}


@torch.no_grad()
def select_q_flowcover_perturbation_views(
    projected_views: Sequence[torch.Tensor],
    text_tokens: torch.Tensor,
    q_views: Sequence[torch.Tensor],
    k_views: Sequence[torch.Tensor],
    v_views: Sequence[torch.Tensor],
    hidden_views: Sequence[torch.Tensor],
    keep_counts: Sequence[int],
    raw_counts: Sequence[int],
    merge_unit: int,
    method: str,
    grid_shapes: Sequence[tuple[int, int]],
    *,
    scale: float,
    query_chunk_size: int = 256,
    group_chunk_size: int = 32,
    source_chunk_size: int = 256,
    temperature: float = 1.0,
    perturbation_lambda: float = 1.0,
    local_radius: int = 1,
    allowed_masks: Sequence[torch.Tensor | None] | None = None,
    compare_flow: bool = False,
) -> tuple[list[SelectionResult], list[dict[str, Any]]]:
    """Select independently per view, retaining exactly the baseline budget.

    Production scores D stay in their original units.  The normalized D is
    provided only as a diagnostic, and never silently replaces the objective.
    ``compare_flow`` is an explicitly charged diagnostic extra greedy pass;
    its attention support comes from the existing shared statistics scan.
    """
    if method not in PERTURBATION_METHODS:
        raise ValueError(f"unsupported perturbation selector {method!r}; no fallback exists")
    view_count = len(projected_views)
    if not view_count or any(
        len(value) != view_count
        for value in (q_views, k_views, v_views, hidden_views, keep_counts, raw_counts, grid_shapes)
    ):
        raise ValueError("all per-view collections must have identical nonzero lengths")
    if allowed_masks is not None and len(allowed_masks) != view_count:
        raise ValueError("allowed_masks must contain one mask per independent image view")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(float(perturbation_lambda)) or float(perturbation_lambda) < 0:
        raise ValueError("perturbation_lambda must be nonnegative and finite")
    if int(local_radius) < 0:
        raise ValueError("local_radius must be nonnegative")
    results: list[SelectionResult] = []
    records: list[dict[str, Any]] = []
    for view_index in range(view_count):
        projected, hidden = projected_views[view_index], hidden_views[view_index]
        q, k, v = q_views[view_index], k_views[view_index], v_views[view_index]
        keep, raw = int(keep_counts[view_index]), int(raw_counts[view_index])
        height, width = map(int, grid_shapes[view_index])
        unit = int(merge_unit)
        groups = int(projected.shape[0])
        if groups < 1 or height * width != groups or raw != groups * unit or not 0 < keep <= groups:
            raise ValueError("invalid per-view N/K/grid/raw_count/merge_unit")
        if hidden.ndim != 2 or int(hidden.shape[0]) != raw:
            raise ValueError("group hidden must follow the complete native raw-token packing")
        _finite("projected group features", projected)
        _finite("group hidden", hidden)
        started = time.perf_counter()
        stages = _Stages(projected.device)
        stages.begin("text_cos")
        text_scores = _text_cosine(projected, text_tokens).float()
        _finite("text scores", text_scores)
        text_mass = torch.softmax(text_scores / float(temperature), dim=0)
        query = _stable_topk(text_scores, keep)
        pi = torch.zeros_like(text_mass)
        pi.index_copy_(
            0, query, torch.softmax(text_scores.index_select(0, query) / float(temperature), dim=0)
        )
        stages.end("text_cos")
        radius = 0 if method.endswith("_nocover") else int(local_radius)
        extra: dict[str, Any] = {}
        if keep == groups:
            # Retaining all groups is an exact no-op.  No deletion statistic
            # can affect the result (and N=1 has no legal deletion source).
            chosen = torch.arange(groups, device=projected.device)
            # Unmeasured arrays must remain absent/empty, not a fabricated
            # measured zero map.  Native Q/K/V capture is skipped upstream.
            support = torch.empty(0, dtype=torch.float32, device=projected.device)
            deletion = torch.empty_like(support)
            omega = torch.empty_like(support)
            greedy_gains = torch.empty(0, dtype=torch.float32, device=projected.device)
            extra = {
                "statistics_skipped": True,
                "statistics_computed": False,
                "statistics_skipped_reason": "full budget; D/support are not measured and their arrays are empty",
                "source_query_count": 0,
                "attention_flops_estimated": 0,
                "q_shape": None if q is None else list(q.shape),
                "k_shape": None if k is None else list(k.shape),
                "v_shape": None if v is None else list(v.shape),
            }
            if compare_flow:
                extra["flow_group_index"] = chosen
        else:
            if q is None or k is None or v is None:
                raise ValueError("native Q/K/V are required for a nontrivial perturbation budget")
            stages.begin("perturbation")
            deletion, support, extra = group_deletion_statistics(
                q,
                k,
                v,
                query,
                pi,
                unit,
                scale=scale,
                query_chunk_size=query_chunk_size,
                group_chunk_size=group_chunk_size,
                source_chunk_size=source_chunk_size,
                allowed_mask=None if allowed_masks is None else allowed_masks[view_index],
            )
            stages.end("perturbation")
            extra["statistics_computed"] = True
            extra["statistics_skipped"] = False
            omega = deletion.clone().index_fill_(0, query, 0.0)
            stages.begin("local_graph")
            z = hidden.reshape(groups, unit * int(hidden.shape[-1])).float()
            affinity, table, valid = sparse_local_affinity(z, height, width, radius)
            stages.end("local_graph")
            stages.begin("greedy")
            chosen, greedy_gains = greedy_direct_cover(
                text_mass, omega * float(perturbation_lambda), keep, affinity, table, valid
            )
            stages.end("greedy")
            extra["local_edge_slots"] = int(table.numel())
            extra["greedy_flops_estimated"] = int(keep * table.numel() * 5)
            extra["local_affinity_flops_estimated"] = int(table.numel() * z.shape[-1] * 2)
            if compare_flow:
                stages.begin("flow_comparison")
                # The historical B objective uses the same ordinary local
                # affinity and empty-set greedy.  Source support is shared;
                # there is no extra QK scan or repeat native projection.
                residual_flow = support.clone().index_fill_(0, query, 0.0)
                if radius != int(local_radius):
                    flow_affinity, flow_table, flow_valid = sparse_local_affinity(
                        z, height, width, int(local_radius)
                    )
                else:
                    flow_affinity, flow_table, flow_valid = affinity, table, valid
                flow_chosen, _ = greedy_direct_cover(
                    text_mass, residual_flow, keep, flow_affinity, flow_table, flow_valid
                )
                extra["flow_group_index"] = flow_chosen
                extra["flow_comparison_flops_estimated"] = int(keep * flow_table.numel() * 5)
                stages.end("flow_comparison")
        timings = stages.resolve()
        selector_seconds = time.perf_counter() - started
        if chosen.numel() != keep or chosen.unique().numel() != keep:
            raise RuntimeError(
                "perturbation selector did not return exactly K unique original groups"
            )
        record: dict[str, Any] = {
            "algorithm_id": method,
            "method": method,
            "implementation_file": __file__,
            "implementation_function": "select_q_flowcover_perturbation_views",
            "view_index": view_index,
            "N": groups,
            "K": keep,
            "raw_count": raw,
            "retention_ratio": keep / groups,
            "grid": [height, width],
            "merge_unit": unit,
            "temperature": float(temperature),
            "perturbation_lambda": float(perturbation_lambda),
            "local_radius": radius,
            "text_scores": text_scores,
            "text_mass": text_mass,
            "pi": pi,
            "text_topk": query,
            "query_group_index": query,
            "query_group_count": int(query.numel()),
            "deletion_scores": deletion,
            "deletion_scores_normalized": deletion
            / deletion.sum().clamp_min(torch.finfo(torch.float32).tiny),
            "support": support,
            "omega": omega,
            "residual_support": omega,
            "final_group_index": chosen,
            "greedy_gains": greedy_gains,
            "replacement_group_count": int((~torch.isin(query, chosen)).sum().item()),
            "text_topk_overlap": float(torch.isin(chosen, query).float().mean().item()),
            "selector_seconds": selector_seconds,
            **timings,
            **extra,
        }
        record["flow_attention_seconds"] = record.get("perturbation_seconds")
        record["shared_flow_and_perturbation_scan"] = True
        record["selector_primary_seconds"] = max(
            0.0, selector_seconds - float(record.get("flow_comparison_seconds", 0.0))
        )
        record["selector_flops"] = int(
            sum(
                record.get(key, 0)
                for key in (
                    "attention_flops_estimated",
                    "greedy_flops_estimated",
                    "local_affinity_flops_estimated",
                    "flow_comparison_flops_estimated",
                )
            )
        ) + int(2 * groups * projected.shape[-1])
        # Existing evaluator records are JSON-serializable.  This conversion
        # is separately reported and follows the measured selector boundary.
        serialization_started = time.perf_counter()
        for key, value in tuple(record.items()):
            if isinstance(value, torch.Tensor):
                record[key] = value.detach().cpu().tolist()
        selected_cpu = chosen.detach().cpu().to(dtype=torch.int64)
        query_cpu = query.detach().cpu().to(dtype=torch.int64)
        raw_idx = (selected_cpu[:, None] * unit + torch.arange(unit)).reshape(-1)
        record["S0_hash"] = hashlib.sha256(query_cpu.numpy().tobytes()).hexdigest()
        record["selected_group_hash"] = hashlib.sha256(selected_cpu.numpy().tobytes()).hexdigest()
        record["keep_idx_hash"] = hashlib.sha256(raw_idx.numpy().tobytes()).hexdigest()
        record["raw_keep_idx_hash"] = record["keep_idx_hash"]
        record["diagnostic_serialization_seconds"] = time.perf_counter() - serialization_started
        empty = torch.empty(0, dtype=torch.long, device=chosen.device)
        results.append(
            SelectionResult(
                indices=chosen,
                final_indices=empty,
                middle_indices=chosen,
                shallow_indices=empty,
                budget=(0, keep, 0),
            )
        )
        records.append(record)
    return results, records


__all__ = [
    "PERTURBATION_METHODS",
    "group_deletion_statistics",
    "sparse_local_affinity",
    "greedy_direct_cover",
    "select_q_flowcover_perturbation_views",
]
