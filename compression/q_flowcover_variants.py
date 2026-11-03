"""Independent direct-evidence Q-FlowCover variants.

This module intentionally does not alter :mod:`compression.q_flowcover`.
The original ``q_flowcover`` method continues to use that module; the
selectors here are only selected by the five ``q_flowcover_direct_*`` method
names.  All computations are per image view and use a sparse local graph.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any, Sequence

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - keeps --help importable
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from model.training_free_pruning import SelectionResult
except Exception:  # pragma: no cover
    from dataclasses import dataclass

    @dataclass
    class SelectionResult:  # type: ignore[no-redef]
        indices: Any
        final_indices: Any
        middle_indices: Any
        shallow_indices: Any
        budget: tuple[int, int, int]

from .q_flowcover import _group_source_support, _stable_topk, _text_cosine


DIRECT_METHODS = {
    "q_flowcover_direct_residual",
    "q_flowcover_direct_querymass",
    "q_flowcover_direct_centered",
    "q_flowcover_direct_adaptive_r",
    "q_flowcover_direct_all",
}


def _require() -> None:
    if torch is None or F is None:
        raise RuntimeError("direct Q-FlowCover requires PyTorch")


@lru_cache(maxsize=64)
def _cached_neighbor_graph(height: int, width: int, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a CPU sparse graph once for each grid/radius combination."""

    if height <= 0 or width <= 0 or radius < 0:
        raise ValueError(f"invalid local graph ({height},{width},R={radius})")
    groups = int(height) * int(width)
    rows: list[list[int]] = []
    for index in range(groups):
        row, col = divmod(index, int(width))
        values = [
            rr * int(width) + cc
            for rr in range(max(0, row - radius), min(int(height), row + radius + 1))
            for cc in range(max(0, col - radius), min(int(width), col + radius + 1))
            if max(abs(rr - row), abs(cc - col)) <= radius
        ]
        rows.append(values)
    degree = max(len(row) for row in rows)
    table = torch.zeros((groups, degree), dtype=torch.long)
    valid = torch.zeros((groups, degree), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row)
        if count:
            table[index, :count] = torch.tensor(row, dtype=torch.long)
            valid[index, :count] = True
    return table, valid


def _graph_on_device(height: int, width: int, radius: int, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    table, valid = _cached_neighbor_graph(int(height), int(width), int(radius))
    return table.to(device=device), valid.to(device=device)


def _group_hidden(hidden: torch.Tensor, merge_unit: int) -> torch.Tensor:
    n = int(hidden.shape[0])
    unit = int(merge_unit)
    if hidden.ndim != 2:
        hidden = hidden.reshape(-1, int(hidden.shape[-1]))
        n = int(hidden.shape[0])
    if unit <= 0 or n % unit:
        raise ValueError(f"hidden token count {n} is not divisible by merge_unit={unit}")
    return F.normalize(hidden.reshape(n // unit, unit * int(hidden.shape[-1])).float(), dim=-1)


def _sparse_affinity(
    z: torch.Tensor,
    height: int,
    width: int,
    radius: int,
    mode: str,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return candidate-row sparse kappa, source indices and validity mask.

    ``table[i]`` lists source evidence groups j covered by candidate i, so the
    returned kappa has the same orientation as ``kappa[j, i]``.  Only E=G*d
    values are retained.  The centered mode computes each source group's
    median from its own cached R-neighborhood and then gathers that scalar for
    every sparse edge.
    """

    _require()
    groups = int(z.shape[0])
    if int(height) * int(width) != groups:
        raise ValueError(f"grid ({height},{width}) does not match groups={groups}")
    table, valid = _graph_on_device(height, width, radius, z.device)
    degree = int(table.shape[1])
    edge_raw = torch.zeros((groups, degree), dtype=torch.float32, device=z.device)
    # Keep the temporary feature gather bounded for high-dimensional visual
    # states; the retained graph itself remains only O(G*degree).
    for start in range(0, groups, 256):
        local = torch.arange(start, min(groups, start + 256), device=z.device)
        source = table.index_select(0, local)
        source_z = z.index_select(0, source.reshape(-1)).reshape(source.shape[0], source.shape[1], -1)
        edge_raw[local] = (
            source_z * z.index_select(0, local).unsqueeze(1)
        ).sum(dim=-1)

    mode = str(mode).strip().lower()
    if mode not in {"raw", "local_centered"}:
        raise ValueError(f"unsupported affinity_mode={mode!r}")
    if mode == "raw":
        kappa = edge_raw.clamp_min(0.0)
    else:
        # edge_raw[j, v] is available in row j because the graph is symmetric.
        # Exclude self edges when estimating tau[j].  A one-group/isolated
        # neighborhood has no non-self observations, so tau defaults to zero.
        source_ids = torch.arange(groups, device=z.device).unsqueeze(1)
        nonself = valid & (table != source_ids)
        neg_inf = torch.full_like(edge_raw, float("nan"))
        nonself_values = torch.where(nonself, edge_raw, neg_inf)
        tau = torch.nanmedian(nonself_values, dim=1).values
        tau = torch.nan_to_num(tau, nan=0.0)
        tau_for_edge = tau.index_select(0, table.reshape(-1)).reshape(table.shape)
        kappa = torch.relu((edge_raw - tau_for_edge) / (1.0 - tau_for_edge + float(eps)))
        self_edge = valid & (table == source_ids)
        kappa = torch.where(self_edge, torch.ones_like(kappa), kappa)
    kappa = kappa.masked_fill(~valid, 0.0)
    return kappa, table, valid, groups * degree


def _direct_cover(
    z: torch.Tensor,
    text_mass: torch.Tensor,
    residual_mass: torch.Tensor,
    keep: int,
    height: int,
    width: int,
    radius: int,
    affinity_mode: str,
) -> tuple[torch.Tensor, int]:
    """Exact greedy direct-text plus residual-flow objective."""

    kappa, table, valid, edge_work = _sparse_affinity(
        z, height, width, radius, affinity_mode
    )
    groups = int(z.shape[0])
    selected = torch.zeros(groups, dtype=torch.bool, device=z.device)
    coverage = torch.zeros(groups, dtype=torch.float32, device=z.device)
    chosen: list[torch.Tensor] = []
    for _ in range(int(keep)):
        # Evaluate the fixed sparse edge table for every group and mask the
        # selected rows.  This avoids a dynamic nonzero/candidate tensor on
        # every iteration while retaining only O(G*degree) edge state.
        source = table
        edge_valid = valid
        similarity = kappa
        current = coverage.index_select(0, source.reshape(-1)).reshape(source.shape)
        source_weight = residual_mass.index_select(0, source.reshape(-1)).reshape(source.shape)
        flow_gain = (
            source_weight
            * torch.relu(similarity - current).masked_fill(~edge_valid, 0.0)
        ).sum(dim=1)
        # The direct text term is awarded only when the candidate itself is
        # selected.  Masking the already-selected rows gives the exact
        # marginal gain without a dense direct-evidence matrix.
        gain = (flow_gain + text_mass).masked_fill(selected, -float("inf"))
        best = torch.argmax(gain)
        selected[best] = True
        chosen.append(best)
        source = table[best]
        similarity = kappa[best]
        # Use a max scatter so padded invalid slots (which reuse index 0) can
        # never overwrite a valid edge update at a boundary cell.
        scatter_source = source.masked_fill(~valid[best], 0)
        scatter_value = similarity.masked_fill(~valid[best], -float("inf"))
        coverage.scatter_reduce_(0, scatter_source, scatter_value, reduce="amax", include_self=True)
    return torch.stack(chosen).sort().values, int(edge_work)


def _query_set(p: torch.Tensor, eta: float) -> torch.Tensor:
    if not 0.0 < float(eta) <= 1.0:
        raise ValueError(f"query_mass must be in (0,1], got {eta}")
    order = torch.argsort(p, descending=True, stable=True)
    cumulative = torch.cumsum(p.index_select(0, order), dim=0)
    # One scalar synchronization is needed to turn the mass threshold into a
    # variable-length query list; the greedy loop itself remains on-device.
    count = int(torch.searchsorted(cumulative, torch.tensor(float(eta), device=p.device)).item()) + 1
    count = min(int(order.numel()), max(1, count))
    return order[:count]


def _select_one(
    projected: torch.Tensor,
    text_tokens: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    hidden: torch.Tensor,
    keep: int,
    raw_count: int,
    merge_unit: int,
    grid: tuple[int, int],
    method: str,
    *,
    scale: float,
    query_chunk_size: int,
    query_mass: float,
    local_radius: int,
    adaptive_radius_max: int,
    affinity_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _require()
    groups = int(projected.shape[0])
    height, width = (int(grid[0]), int(grid[1]))
    if groups <= 0 or height * width != groups:
        raise ValueError(f"grid ({height},{width}) does not match groups={groups}")
    if int(raw_count) != groups * int(merge_unit):
        raise ValueError("raw_count/group count/merge_unit are inconsistent")
    if not 0 < int(keep) <= groups:
        raise ValueError(f"keep={keep} must be in [1,{groups}]")
    if method not in DIRECT_METHODS:
        raise ValueError(f"unsupported direct Q-FlowCover method {method!r}")

    text_score = _text_cosine(projected, text_tokens).float()
    text_mass = torch.softmax(text_score, dim=0)
    text_topk = _stable_topk(text_score, int(keep))
    if method in {"q_flowcover_direct_querymass", "q_flowcover_direct_all"}:
        query_groups = _query_set(text_mass, float(query_mass))
    else:
        query_groups = text_topk
    p_hat = torch.zeros_like(text_mass)
    query_mass_total = text_mass.index_select(0, query_groups).sum()
    p_hat[query_groups] = text_mass.index_select(0, query_groups) / query_mass_total.clamp_min(1e-12)
    support, source_query_count = _group_source_support(
        q,
        k,
        query_groups,
        p_hat,
        int(merge_unit),
        scale=float(scale),
        query_chunk_size=int(query_chunk_size),
    )
    residual = support.index_fill(0, query_groups, 0.0)

    if method in {"q_flowcover_direct_centered", "q_flowcover_direct_all"}:
        effective_affinity = "local_centered"
    else:
        effective_affinity = str(affinity_mode).strip().lower()
    if effective_affinity not in {"raw", "local_centered"}:
        raise ValueError(f"unsupported affinity_mode={affinity_mode!r}")
    if method in {"q_flowcover_direct_adaptive_r", "q_flowcover_direct_all"}:
        radius = max(1, int(math.ceil((math.sqrt(float(groups) / float(keep)) - 1.0) / 2.0)))
        radius = min(int(adaptive_radius_max), radius)
    else:
        radius = int(local_radius)
    if radius < 0:
        raise ValueError("local_radius must be non-negative")
    selected, edge_work = _direct_cover(
        _group_hidden(hidden, int(merge_unit)),
        text_mass,
        residual,
        int(keep),
        height,
        width,
        radius,
        effective_affinity,
    )
    if int(selected.numel()) != int(keep) or int(torch.unique(selected).numel()) != int(keep):
        raise RuntimeError("direct Q-FlowCover did not return exactly K unique groups")
    diagnostics = {
        "method": method,
        "N": groups,
        "K": int(keep),
        "text_topk": text_topk.detach().cpu().tolist(),
        "query_group_index": query_groups.detach().cpu().tolist(),
        "final_group_index": selected.detach().cpu().tolist(),
        "pi": p_hat.detach().cpu().tolist(),
        "text_mass": text_mass.detach().cpu().tolist(),
        "support": support.detach().cpu().tolist(),
        "residual_support": residual.detach().cpu().tolist(),
        "omega": residual.detach().cpu().tolist(),
        "query_mass": float(query_mass),
        "query_mass_actual": float(text_mass.index_select(0, query_groups).sum().detach().cpu().item()),
        "query_group_count": int(query_groups.numel()),
        "grid": [height, width],
        "merge_unit": int(merge_unit),
        "raw_count": int(raw_count),
        "source_query_count": int(source_query_count),
        "edge_work": int(edge_work),
        "local_radius": int(radius),
        "affinity_mode": effective_affinity,
        "text_topk_overlap": float(torch.isin(selected, text_topk).float().mean().detach().cpu().item()),
        "replacement_group_count": int((~torch.isin(text_topk, selected)).sum().detach().cpu().item()),
    }
    selector_flops = 2 * int(projected.shape[-1]) * groups
    selector_flops += 2 * int(q.shape[1]) * int(q.shape[2]) * int(raw_count) * int(source_query_count)
    selector_flops += 2 * int(merge_unit) * int(hidden.shape[-1]) * int(edge_work)
    diagnostics["selector_flops"] = int(selector_flops)
    return selected, diagnostics


def select_q_flowcover_variant_views(
    projected_views: Sequence[torch.Tensor],
    text_tokens: torch.Tensor,
    q_views: Sequence[torch.Tensor],
    k_views: Sequence[torch.Tensor],
    hidden_views: Sequence[torch.Tensor],
    keep_counts: Sequence[int],
    raw_counts: Sequence[int],
    merge_unit: int,
    method: str,
    grid_shapes: Sequence[tuple[int, int]],
    *,
    scale: float,
    query_chunk_size: int = 256,
    query_mass: float = 0.9,
    local_radius: int = 1,
    adaptive_radius_max: int = 3,
    affinity_mode: str = "raw",
) -> tuple[list[SelectionResult], list[dict[str, Any]]]:
    """Run one direct variant independently for each image view."""

    _require()
    method = str(method).strip().lower()
    if method not in DIRECT_METHODS:
        raise ValueError(f"unsupported direct method {method!r}")
    if len(projected_views) != len(q_views) or len(projected_views) != len(k_views):
        raise ValueError("direct selector received mismatched Q/K views")
    if not 0 < float(query_mass) <= 1.0:
        raise ValueError("query_mass must be in (0,1]")
    if int(adaptive_radius_max) < 1:
        raise ValueError("adaptive_radius_max must be at least 1")
    results: list[SelectionResult] = []
    diagnostics: list[dict[str, Any]] = []
    for projected, q, k, hidden, keep, raw_count, grid in zip(
        projected_views,
        q_views,
        k_views,
        hidden_views,
        keep_counts,
        raw_counts,
        grid_shapes,
    ):
        started = time.perf_counter()
        selected, debug = _select_one(
            projected,
            text_tokens,
            q,
            k,
            hidden,
            int(keep),
            int(raw_count),
            int(merge_unit),
            grid,
            method,
            scale=float(scale),
            query_chunk_size=int(query_chunk_size),
            query_mass=float(query_mass),
            local_radius=int(local_radius),
            adaptive_radius_max=int(adaptive_radius_max),
            affinity_mode=str(affinity_mode),
        )
        debug["selector_seconds"] = time.perf_counter() - started
        empty = torch.empty(0, dtype=torch.long, device=selected.device)
        results.append(
            SelectionResult(
                indices=selected,
                final_indices=empty,
                middle_indices=selected,
                shallow_indices=empty,
                budget=(0, int(selected.numel()), 0),
            )
        )
        diagnostics.append(debug)
    return results, diagnostics


__all__ = ["DIRECT_METHODS", "select_q_flowcover_variant_views"]
