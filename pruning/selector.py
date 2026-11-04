"""Perturbation selection using optimized statistics and exact local coverage."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import torch

from .common import SelectionResult, _stable_topk, _text_cosine
from .reference import PERTURBATION_METHODS, _Stages, _finite, sparse_local_affinity
from .statistics import group_deletion_statistics_optimized
from .coverage import greedy_direct_cover_optimized


@torch.no_grad()
def select_q_flowcover_perturbation_optimized_views(
    projected_views: Sequence[torch.Tensor],
    text_tokens: torch.Tensor,
    q_views: Sequence[torch.Tensor | None],
    k_views: Sequence[torch.Tensor | None],
    v_views: Sequence[torch.Tensor | None],
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
    max_attention_mb: float = 512.0,
    greedy_backend: str = "reference",
) -> tuple[list[SelectionResult], list[dict[str, Any]]]:
    import math

    if method not in PERTURBATION_METHODS:
        raise ValueError(f"unregistered perturbation method {method!r}; no fallback exists")
    count = len(projected_views)
    if not count or any(
        len(x) != count
        for x in (q_views, k_views, v_views, hidden_views, keep_counts, raw_counts, grid_shapes)
    ):
        raise ValueError("all per-view collections must have equal nonzero lengths")
    if allowed_masks is not None and len(allowed_masks) != count:
        raise ValueError("allowed_masks must contain one mask per image view")
    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(perturbation_lambda)
        or perturbation_lambda < 0
    ):
        raise ValueError("temperature must be positive and lambda nonnegative, both finite")
    if local_radius < 0 or min(query_chunk_size, group_chunk_size, source_chunk_size) < 1:
        raise ValueError("invalid radius/chunk sizes")
    if greedy_backend not in {"reference", "local_refresh", "scripted_local_refresh"}:
        raise ValueError("unknown greedy backend; no fallback exists")
    if not math.isfinite(max_attention_mb) or max_attention_mb <= 0:
        raise ValueError("attention memory budget must be finite and positive")
    results, records = [], []
    for view in range(count):
        projected, hidden = projected_views[view], hidden_views[view]
        groups = int(projected.shape[0])
        keep, raw, unit = int(keep_counts[view]), int(raw_counts[view]), int(merge_unit)
        height, width = map(int, grid_shapes[view])
        if (
            projected.ndim != 2
            or hidden.ndim != 2
            or groups < 1
            or height * width != groups
            or not 0 < keep <= groups
            or unit < 1
            or raw != groups * unit
            or hidden.shape[0] != raw
        ):
            raise ValueError("invalid group features, native grid, complete packing or budget")
        _finite("projected group features", projected)
        _finite("native group hidden", hidden)
        started = time.perf_counter()
        stages = _Stages(projected.device)
        stages.begin("text_cos")
        scores = _text_cosine(projected, text_tokens).float()
        _finite("text scores", scores)
        p = torch.softmax(scores / temperature, 0)
        query = _stable_topk(scores, keep)
        pi = torch.zeros_like(p).index_copy_(
            0, query, torch.softmax(scores.index_select(0, query) / temperature, 0)
        )
        stages.end("text_cos")
        radius = 0 if method.endswith("_nocover") else int(local_radius)
        extra: dict[str, Any] = {}
        if keep == groups:
            selected = torch.arange(groups, device=projected.device)
            deletion = support = omega = torch.empty(0, device=projected.device)
            gains = torch.empty_like(deletion)
            extra.update(
                statistics_computed=False,
                statistics_skipped=True,
                statistics_skipped_reason="full budget; D/support not measured",
                source_query_count=0,
                attention_flops_estimated=0,
                q_shape=None,
                k_shape=None,
                v_shape=None,
            )
            if compare_flow:
                extra["flow_group_index"] = selected
        else:
            q, k, v = q_views[view], k_views[view], v_views[view]
            if q is None or k is None or v is None:
                raise ValueError("nontrivial budgets require the actual native next-block Q/K/V")
            stages.begin("perturbation")
            deletion, support, extra = group_deletion_statistics_optimized(
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
                max_attention_mb=max_attention_mb,
                allowed_mask=None if allowed_masks is None else allowed_masks[view],
            )
            stages.end("perturbation")
            extra.update(statistics_computed=True, statistics_skipped=False)
            omega = deletion.clone().index_fill_(0, query, 0.0)
            stages.begin("local_graph")
            z = hidden.reshape(groups, unit * hidden.shape[-1]).float()
            affinity, table, valid = sparse_local_affinity(z, height, width, radius)
            stages.end("local_graph")
            stages.begin("greedy")
            selected, gains, greedy_meta = greedy_direct_cover_optimized(
                p,
                omega * perturbation_lambda,
                keep,
                affinity,
                table,
                valid,
                backend=greedy_backend,
                return_diagnostics=True,
                grid_shape=(height, width),
                local_radius=radius,
            )
            stages.end("greedy")
            extra["greedy_implementation"] = greedy_meta
            extra["local_edge_slots"] = int(table.numel())
            extra["greedy_flops_estimated"] = int(
                greedy_meta.get("estimated_flops", keep * table.numel() * 5)
            )
            extra["local_affinity_flops_estimated"] = int(table.numel() * z.shape[-1] * 2)
            if compare_flow:
                stages.begin("flow_comparison")
                flow_aff, flow_tab, flow_valid = (
                    (affinity, table, valid)
                    if radius == local_radius
                    else sparse_local_affinity(z, height, width, local_radius)
                )
                flow, _ = greedy_direct_cover_optimized(
                    p,
                    support.clone().index_fill_(0, query, 0.0),
                    keep,
                    flow_aff,
                    flow_tab,
                    flow_valid,
                    backend=greedy_backend,
                    grid_shape=(height, width),
                    local_radius=local_radius,
                )
                extra["flow_group_index"] = flow
                extra["flow_comparison_flops_estimated"] = int(keep * flow_tab.numel() * 5)
                stages.end("flow_comparison")
        timings = stages.resolve()
        elapsed = time.perf_counter() - started
        if selected.numel() != keep or selected.unique().numel() != keep:
            raise RuntimeError("optimized selector must return exactly K unique original IDs")
        record = dict(
            algorithm_id=method,
            method=method,
            selector_impl="optimized",
            implementation_file=__file__,
            implementation_function="select_q_flowcover_perturbation_optimized_views",
            view_index=view,
            N=groups,
            K=keep,
            raw_count=raw,
            retention_ratio=keep / groups,
            grid=[height, width],
            merge_unit=unit,
            temperature=temperature,
            perturbation_lambda=perturbation_lambda,
            local_radius=radius,
            text_scores=scores,
            text_mass=p,
            pi=pi,
            text_topk=query,
            query_group_index=query,
            query_group_count=query.numel(),
            deletion_scores=deletion,
            deletion_scores_normalized=deletion
            / deletion.sum().clamp_min(torch.finfo(torch.float32).tiny),
            support=support,
            omega=omega,
            residual_support=omega,
            final_group_index=selected,
            greedy_gains=gains,
            replacement_group_count=int((~torch.isin(query, selected)).sum().item()),
            text_topk_overlap=float(torch.isin(selected, query).float().mean().item()),
            selector_seconds=elapsed,
            **timings,
            **extra,
        )
        record["max_attention_mb"] = max_attention_mb
        record["flow_attention_seconds"] = record.get("perturbation_seconds")
        record["shared_flow_and_perturbation_scan"] = True
        record["selector_primary_seconds"] = max(
            0.0, elapsed - record.get("flow_comparison_seconds", 0.0)
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
        ) + 2 * groups * int(projected.shape[-1])
        serialization = time.perf_counter()
        for key, value in tuple(record.items()):
            if torch.is_tensor(value):
                record[key] = value.detach().cpu().tolist()
        selected_cpu, query_cpu = selected.detach().cpu(), query.detach().cpu()
        raw_index = (selected_cpu[:, None] * unit + torch.arange(unit)).flatten()
        record["S0_hash"] = hashlib.sha256(query_cpu.numpy().tobytes()).hexdigest()
        record["selected_group_hash"] = hashlib.sha256(selected_cpu.numpy().tobytes()).hexdigest()
        record["keep_idx_hash"] = record["raw_keep_idx_hash"] = hashlib.sha256(
            raw_index.numpy().tobytes()
        ).hexdigest()
        record["diagnostic_serialization_seconds"] = time.perf_counter() - serialization
        empty = selected.new_empty(0)
        results.append(
            SelectionResult(
                indices=selected,
                final_indices=empty,
                middle_indices=selected,
                shallow_indices=empty,
                budget=(0, keep, 0),
            )
        )
        records.append(record)
    return results, records
