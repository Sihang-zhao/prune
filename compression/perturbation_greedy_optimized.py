"""Exact local-refresh greedy backends for the perturbation selector.

The reference implementation remains in ``q_flowcover_perturbation.py``.
This module reuses the independently implemented sparse local refresh and
offers an explicitly named TorchScript backend for its sequential loop.
Neither backend approximates greedy with a static Top-K.  No image-dependent
input, support, feature or coverage tensor is cached across invocations.

The local graph must be symmetric (the native square-radius grid is), while
edge affinities may be asymmetric.  Refreshing complete affected candidate
rows preserves the original floating-point reduction order rather than
incrementally subtracting rounded marginal-gain deltas.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from .q_flowcover_perturbation import greedy_direct_cover as _reference


BACKENDS = ("reference", "local_refresh", "scripted_local_refresh")


def _local_loop(
    text_mass: torch.Tensor,
    residual_mass: torch.Tensor,
    keep: int,
    affinity: torch.Tensor,
    table: torch.Tensor,
    valid: torch.Tensor,
    affected_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ATen updates, optionally using a cached unique 2R grid table."""
    groups = text_mass.numel()
    source_weight = residual_mass.index_select(0, table.flatten()).reshape(table.shape)
    selected = torch.zeros(groups, dtype=torch.bool, device=text_mass.device)
    coverage = torch.zeros_like(text_mass)
    chosen = torch.empty(keep, dtype=torch.long, device=text_mass.device)
    gains = torch.empty(keep, dtype=text_mass.dtype, device=text_mass.device)
    similarity = affinity.masked_fill(~valid, 0.0)
    cached_gain = text_mass + (source_weight * torch.relu(similarity)).sum(-1)
    for step in range(keep):
        best = cached_gain.argmax().reshape(1)
        chosen[step:step + 1] = best
        gains[step:step + 1] = cached_gain.index_select(0, best)
        selected.scatter_(0, best, True)
        if step + 1 == keep:
            break
        source = table.index_select(0, best).flatten()
        values = similarity.index_select(0, best).flatten()
        permitted = valid.index_select(0, best).flatten()
        coverage.scatter_reduce_(
            0, source.masked_fill(~permitted, 0),
            values.masked_fill(~permitted, -float("inf")),
            reduce="amax", include_self=True,
        )
        cached_gain.scatter_(0, best, -float("inf"))
        if affected_table.numel() > 0:
            affected = affected_table.index_select(0, best).flatten()
        else:
            affected = table.index_select(0, source).flatten()
        candidate_sources = table.index_select(0, affected)
        current = coverage.index_select(0, candidate_sources.flatten()).reshape(candidate_sources.shape)
        updated = text_mass.index_select(0, affected) + (
            source_weight.index_select(0, affected)
            * torch.relu(similarity.index_select(0, affected) - current)
        ).sum(-1)
        updated.masked_fill_(selected.index_select(0, affected), -float("inf"))
        cached_gain.scatter_reduce_(0, affected, updated, reduce="amax", include_self=False)
    return chosen.sort().values, gains


_scripted_loop = torch.jit.script(_local_loop)


@lru_cache(maxsize=64)
def _grid_affected_table(height: int, width: int, radius: int, device_string: str) -> torch.Tensor:
    """Cache integer structure only: square-radius graph's 2-hop superset.

    A source changed by candidate i lies at distance <=R. Any other candidate
    whose gain depends on that source therefore lies at distance <=2R from i.
    R=1 has 25 fixed slots rather than 9*9 duplicate two-hop paths. Invalid
    padded entries can refresh unrelated legal rows harmlessly; no such row's
    exact gain changes simply because it was recomputed.
    """
    device = torch.device(device_string)
    ids = torch.arange(height * width, device=device)
    y = ids.div(width, rounding_mode="floor")
    x = ids.remainder(width)
    offsets = torch.cartesian_prod(
        torch.arange(-2 * radius, 2 * radius + 1, device=device),
        torch.arange(-2 * radius, 2 * radius + 1, device=device),
    )
    table = (y[:, None] + offsets[:, 0]) * width + x[:, None] + offsets[:, 1]
    return table.clamp(0, height * width - 1)


def _validate(
    text_mass: torch.Tensor,
    residual_mass: torch.Tensor,
    keep: int,
    affinity: torch.Tensor,
    table: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    if text_mass.ndim != 1 or not text_mass.numel() or residual_mass.shape != text_mass.shape:
        raise ValueError("text/residual masses must be nonempty equal-length vectors")
    if not text_mass.is_floating_point() or residual_mass.dtype != text_mass.dtype:
        raise TypeError("text/residual masses must have the same floating dtype")
    groups = int(text_mass.numel())
    if not 0 < int(keep) <= groups:
        raise ValueError("keep must be in [1, groups]")
    if table.ndim != 2 or table.shape[0] != groups or table.shape != affinity.shape or table.shape != valid.shape:
        raise ValueError("affinity/table/valid must share [groups,neighbor_slots] shape")
    if not table.shape[1]:
        raise ValueError("local graph needs at least one slot")
    if table.dtype != torch.long or valid.dtype != torch.bool:
        raise TypeError("table must be int64 and valid must be bool")
    if affinity.dtype != text_mass.dtype:
        raise TypeError("affinity and masses must have the same floating dtype")
    if any(x.device != text_mass.device for x in (residual_mass, affinity, table, valid)):
        raise ValueError("all inputs must share a device")


@torch.no_grad()
def greedy_direct_cover_optimized(
    text_mass: torch.Tensor,
    residual_mass: torch.Tensor,
    keep: int,
    affinity: torch.Tensor,
    table: torch.Tensor,
    valid: torch.Tensor,
    *,
    backend: str = "local_refresh",
    grid_shape: tuple[int, int] | None = None,
    local_radius: int = 1,
    return_diagnostics: bool = False,
) -> Any:
    """Return sorted original group IDs and selection-order exact gains.

    ``backend=reference`` directly calls the unchanged original full-refresh
    greedy. This is an important GPU performance control: less arithmetic
    does not necessarily compensate for extra local-update kernel launches.
    ``backend=local_refresh`` calls the existing sparse local-refresh code.
    ``backend=scripted_local_refresh`` uses the same operations in a scripted
    C++ loop, avoiding Python per-round dispatch without a CUDA-graph feature
    cache or first-shape graph capture. Neither silently falls back.

    R=0/self-only graphs have a modular objective. Stable Top-K in that
    explicitly detected case is mathematically exact sequential greedy,
    including smallest-ID ties. General local graphs retain K updates.

    Set ``return_diagnostics=True`` for a third returned metadata dictionary.
    CUDA correctness/performance still needs validation on the target GPU.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unsupported perturbation greedy backend {backend!r}; expected {BACKENDS}")
    _validate(text_mass, residual_mass, keep, affinity, table, valid)
    groups, degree = map(int, table.shape)
    affected_table = torch.empty((0, 0), dtype=torch.long, device=table.device)
    affected_slots = degree ** 2
    if grid_shape is not None and backend != "reference":
        height, width = map(int, grid_shape)
        radius = int(local_radius)
        if min(height, width) <= 0 or height * width != groups or radius < 0:
            raise ValueError("grid_shape/local_radius do not describe this image view")
        if degree != (2 * radius + 1) ** 2:
            raise ValueError("local graph slots disagree with the explicitly supplied grid radius")
        affected_table = _grid_affected_table(height, width, radius, str(table.device))
        affected_slots = int(affected_table.shape[1])
    self_only = backend != "reference" and degree == 1 and torch.equal(table[:, 0], torch.arange(groups, device=table.device))
    if backend == "reference":
        chosen, gains = _reference(text_mass, residual_mass, int(keep), affinity, table, valid)
        implementation = "compression.q_flowcover_perturbation.greedy_direct_cover"
        affected_slots = 0  # Original backend refreshes all candidate rows.
    elif self_only:
        score = text_mass + residual_mass * affinity[:, 0].clamp_min(0).masked_fill(~valid[:, 0], 0)
        order = torch.argsort(score, descending=True, stable=True)[:int(keep)]
        chosen, gains = order.sort().values, score.index_select(0, order)
        implementation = "stable_topk_exact_self_only_objective"
    elif backend == "local_refresh":
        chosen, gains = _local_loop(text_mass, residual_mass, int(keep), affinity, table, valid, affected_table)
        implementation = __name__ + "._local_loop"
    else:
        chosen, gains = _scripted_loop(text_mass, residual_mass, int(keep), affinity, table, valid, affected_table)
        implementation = __name__ + "._scripted_loop"
    # Match the original perturbation reference's diagnostic gain dtype.
    # Production scores/coverage use FP32 already, so this is normally a no-op.
    gains = gains.to(dtype=torch.float32)
    if not return_diagnostics:
        return chosen, gains
    # Counts describe executed sparse slots, not a complete hardware FLOPs
    # measurement. Argmax scans remain O(K*N) in either exact backend.
    refresh_rounds = 0 if self_only else max(0, int(keep) - 1)
    candidate_edge_visits = (
        int(keep) * groups * degree if backend == "reference"
        else groups if self_only else groups * degree + refresh_rounds * affected_slots * degree
    )
    argmax_comparisons = 0 if self_only else int(keep) * max(0, groups - 1)
    metadata = {
        "greedy_backend": backend,
        "greedy_implementation": implementation,
        "greedy_exact": True,
        "greedy_tie_break": "smallest_original_group_id",
        "greedy_local_slots_per_candidate": degree,
        "greedy_affected_candidate_slots": affected_slots,
        "greedy_cached_grid_shape": list(grid_shape) if grid_shape is not None and backend != "reference" else None,
        "greedy_reference_candidate_edge_visits": int(keep) * groups * degree,
        "greedy_optimized_candidate_edge_visits_bound": candidate_edge_visits,
        "greedy_argmax_candidate_visits": 0 if self_only else int(keep) * groups,
        "estimated_flops": 5 * candidate_edge_visits,
        "greedy_flops_estimated": 5 * candidate_edge_visits,
        "estimated_argmax_comparisons": argmax_comparisons,
        "estimated_flops_note": (
            "Partial analytical estimate: five scalar operations per visited local evidence slot; "
            "argmax comparisons are listed separately. Does not count index/sort comparisons, "
            "all scatter updates, memory traffic, allocation or kernel dispatch."
        ),
        "greedy_sequential_rounds": 0 if self_only else int(keep),
        "greedy_modular_self_only": self_only,
        "greedy_cross_sample_tensor_cache": False,
        "greedy_cost_note": (
            "Unchanged reference full-refresh greedy: K rounds, N candidate gains per round, "
            "d local evidence slots per candidate, O(K*N*d). No 2R graph build or local "
            "refresh is executed in this backend."
        ) if backend == "reference" else (
            "Exact local refresh: initial O(N*d), K argmax scans O(K*N), and "
            "K*d^3 edge refreshes for generic graphs, or K*(4R+1)^2*d "
            "with an explicitly supplied grid and cached integer 2R neighborhoods; "
            "no dense N*N similarity and no per-round CPU scalar synchronization. "
            "Scripted backend still launches ATen GPU operations; no GPU speedup is assumed."
        ),
    }
    return chosen, gains, metadata


__all__ = ["BACKENDS", "greedy_direct_cover_optimized"]
