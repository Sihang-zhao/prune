"""Equivalent bounded rectangular attention and Value-Gram deletion statistics."""

from __future__ import annotations

import math
from typing import Any

import torch

from .reference import (
    _finite,
    _mask_slice,
    _readout_statistics,
    _validate_attention,
)


def _rectangle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_ids: torch.Tensor,
    key_ids: torch.Tensor,
    *,
    scale: float,
    allowed_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Heads-major contiguous K/V avoid implicit repacking of interleaved
    # native [raw,heads,dim] strides on every matrix multiplication.
    logits = torch.bmm(query, key.transpose(1, 2))
    logits.mul_(float(scale))
    allowed = _mask_slice(allowed_mask, query_ids, key_ids, int(query.shape[0]))
    if allowed is not None:
        logits.masked_fill_(~allowed, -float("inf"))
    attention = torch.softmax(logits, dim=-1)
    del logits
    output = torch.bmm(attention, value)
    return attention, output


def _repair_inline(
    error_squared: torch.Tensor,
    suspicious: torch.Tensor,
    unsafe: torch.Tensor,
    grouped_attention: torch.Tensor,
    output: torch.Tensor,
    value_groups: torch.Tensor,
) -> int:
    """Re-evaluate flagged readouts while their full attention is available."""
    positions = (suspicious & ~unsafe).nonzero(as_tuple=False)
    count = int(positions.shape[0])
    # Only the sparse readout vectors are materialized. Dynamic nonzero can
    # synchronize CUDA once per query chunk; no model/QK/AV replay is needed.
    for begin in range(0, count, 16384):
        head, query, group = positions[begin : begin + 16384].unbind(1)
        local_attention = grouped_attention[head, query, group]
        source = value_groups[head, group]
        mass = local_attention.sum(-1)
        readout = torch.bmm(local_attention.unsqueeze(1), source).squeeze(1)
        error = (readout - mass[:, None] * output[head, query]) / (1.0 - mass)[:, None]
        error_squared[head, query, group] = error.square().sum(-1)
    return count


@torch.no_grad()
def group_deletion_statistics_optimized(
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
    max_attention_mb: float = 512.0,
    cancellation_threshold: float = 1e-3,
    cancellation_recovery: str = "deferred",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute the reference score/support with one full-source QK per chunk.

    ``max_attention_mb`` bounds one [heads,query_chunk,raw_source] FP32
    rectangle. It is not a claim about total workspace, which has several
    such bounded temporaries. ``group_chunk_size`` applies to rare direct
    deferred cancellation recovery; normal Gram scoring is vectorized across groups.
    ``source_chunk_size`` applies to the rare stable masked-source rescans.
    All legal Key rows share one softmax denominator. No N x N matrix or
    all-group hidden-vector difference tensor is constructed.

    ``inline`` repairs only ill-conditioned raw-query/head/group tuples
    before reducing the current chunk. ``deferred`` retains the original
    whole-group rescoring pass. Both use the same direct readout formula;
    near-unit attention mass always uses independently masked attention.
    """
    raw, heads, dim, groups = _validate_attention(
        q,
        k,
        v,
        query_groups,
        query_weights,
        merge_unit,
        allowed_mask,
    )
    for name, count in (
        ("query_chunk_size", query_chunk_size),
        ("group_chunk_size", group_chunk_size),
        ("source_chunk_size", source_chunk_size),
    ):
        if int(count) <= 0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("attention scale must be positive and finite")
    if not math.isfinite(float(max_attention_mb)) or float(max_attention_mb) <= 0:
        raise ValueError("max_attention_mb must be positive and finite")
    if not 0 < float(near_one_threshold) < 1:
        raise ValueError("near_one_threshold must be in (0,1)")
    if not 0 < float(cancellation_threshold) < 1:
        raise ValueError("cancellation_threshold must be in (0,1)")
    if cancellation_recovery not in {"deferred", "inline"}:
        raise ValueError("cancellation_recovery must be deferred or inline")
    if groups <= 1:
        raise ValueError("deletion statistics need at least two source groups")
    unit = int(merge_unit)
    per_query_bytes = heads * raw * torch.tensor([], dtype=torch.float32).element_size()
    budget_bytes = int(float(max_attention_mb) * 1024 * 1024)
    budget_queries = budget_bytes // per_query_bytes
    if budget_queries < 1:
        raise ValueError("max_attention_mb cannot hold even one full-source raw Query rectangle")
    chunk = min(int(query_chunk_size), max(1, raw - 1), budget_queries)
    raw_queries = (query_groups[:, None] * unit + torch.arange(unit, device=q.device)).flatten()
    weights = query_weights.index_select(0, query_groups).float().repeat_interleave(unit) / float(
        unit
    )
    query_heads = q.transpose(0, 1)
    repeats = heads // int(k.shape[1])
    key = k.transpose(0, 1).float().contiguous()
    value = v.transpose(0, 1).float().contiguous()
    if repeats != 1:
        key = key.repeat_interleave(repeats, dim=0)
        value = value.repeat_interleave(repeats, dim=0)
    # A shared translation of every source Value cancels exactly in b-m*o.
    # Centering removes common offsets before the Gram identity, improving
    # conditioning while leaving native tensors and the mathematical score
    # unchanged. The final FP32 score is tested against explicit deletion.
    center = value.mean(1, keepdim=True)
    value = (value - center).contiguous()
    value_groups = value.reshape(heads, groups, unit, dim)
    gram = torch.matmul(value_groups, value_groups.transpose(-1, -2))
    key_ids = torch.arange(raw, device=q.device)
    scores = torch.zeros(groups, dtype=torch.float32, device=q.device)
    support = torch.zeros_like(scores)
    near_one = torch.zeros(groups, dtype=torch.bool, device=q.device)
    cancellation = torch.zeros_like(near_one)
    inline_tuples = 0
    for start in range(0, raw_queries.numel(), chunk):
        ids = raw_queries[start : start + chunk]
        local_q = query_heads.index_select(1, ids).float().contiguous()
        attention, output = _rectangle(
            local_q,
            key,
            value,
            ids,
            key_ids,
            scale=scale,
            allowed_mask=allowed_mask,
        )
        local_count = int(ids.numel())
        grouped_attention = attention.reshape(heads, local_count, groups, unit)
        mass = grouped_attention.sum(-1)
        # Contract four source Values via a 4x4 Gram block rather than
        # materializing one head_dim-wide readout for every query/group.
        gram_attention = torch.einsum("hcgu,hguv->hcgv", grouped_attention, gram)
        gram_attention.mul_(grouped_attention)
        b_squared = gram_attention.sum(-1)
        del gram_attention
        ov = torch.bmm(output, value.transpose(1, 2))
        ov.mul_(attention)
        cross = ov.reshape(heads, local_count, groups, unit).sum(-1)
        del ov
        cross.mul_(mass).mul_(2.0)
        output_squared = output.square().sum(-1)[..., None]
        output_term = mass.square().mul_(output_squared)
        numerator = b_squared - cross + output_term
        magnitude = b_squared.abs() + cross.abs() + output_term.abs()
        suspicious = (numerator < 0) | (
            (numerator <= float(cancellation_threshold) * magnitude)
            & (magnitude > torch.finfo(torch.float32).tiny)
        )
        remaining = 1.0 - mass
        unsafe = remaining <= float(near_one_threshold)
        near_one |= unsafe.any(dim=(0, 1))
        cancellation |= suspicious.any(dim=(0, 1))
        denominator = torch.where(unsafe, torch.ones_like(remaining), remaining)
        error_squared = numerator.clamp_min(0).div_(denominator.square())
        error_squared.masked_fill_(unsafe | suspicious, 0.0)
        if cancellation_recovery == "inline":
            inline_tuples += _repair_inline(
                error_squared, suspicious, unsafe, grouped_attention, output, value_groups
            )
        local_weight = weights[start : start + local_count]
        scores.add_((error_squared.mean(0) * local_weight[:, None]).sum(0))
        support.add_((mass.mean(0) * local_weight[:, None]).sum(0))
        del attention, grouped_attention, mass, output, b_squared, cross
        del output_squared, output_term, numerator, magnitude, suspicious
        del remaining, unsafe, denominator, error_squared
    # Deferred recovery transfers only these small flags after the main scan.
    # Inline recovery also resolves sparse tuple counts in each query chunk.
    flags = torch.stack((near_one, cancellation & ~near_one)).detach().cpu()
    near_ids = flags[0].nonzero(as_tuple=False).flatten().tolist()
    cancel_ids = flags[1].nonzero(as_tuple=False).flatten().tolist()
    if cancel_ids and cancellation_recovery == "deferred":
        cancel_groups = torch.tensor(cancel_ids, dtype=torch.long, device=q.device)
        recovered = torch.zeros(len(cancel_ids), dtype=torch.float32, device=q.device)
        for start in range(0, raw_queries.numel(), chunk):
            ids = raw_queries[start : start + chunk]
            local_q = query_heads.index_select(1, ids).float().contiguous()
            attention, output = _rectangle(
                local_q,
                key,
                value,
                ids,
                key_ids,
                scale=scale,
                allowed_mask=allowed_mask,
            )
            all_groups = attention.reshape(heads, int(ids.numel()), groups, unit)
            local_weight = weights[start : start + int(ids.numel())]
            for begin in range(0, len(cancel_ids), int(group_chunk_size)):
                selected = cancel_groups[begin : begin + int(group_chunk_size)]
                local_a = all_groups.index_select(2, selected)
                mass = local_a.sum(-1)
                source = value_groups.index_select(1, selected)
                b = torch.einsum("hcgu,hgud->hcgd", local_a, source)
                error = (b - mass[..., None] * output[:, :, None]) / (1 - mass)[..., None]
                score = error.square().sum(-1).mean(0)
                recovered[begin : begin + int(selected.numel())].add_(
                    (score * local_weight[:, None]).sum(0)
                )
        scores.index_copy_(0, cancel_groups, recovered)
    for group in near_ids:
        excluded = torch.arange(group * unit, (group + 1) * unit, device=q.device)
        recovered = torch.zeros((), dtype=torch.float32, device=q.device)
        for start in range(0, raw_queries.numel(), chunk):
            ids = raw_queries[start : start + chunk]
            local_q = query_heads.index_select(1, ids).float().contiguous()
            _, full = _readout_statistics(
                local_q,
                key,
                value,
                ids,
                scale=scale,
                source_chunk_size=source_chunk_size,
                allowed_mask=allowed_mask,
            )
            _, deleted = _readout_statistics(
                local_q,
                key,
                value,
                ids,
                scale=scale,
                source_chunk_size=source_chunk_size,
                allowed_mask=allowed_mask,
                excluded_raw=excluded,
                validate_output=True,
            )
            score = (full - deleted).square().sum(-1).mean(0)
            recovered.add_((score * weights[start : start + int(ids.numel())]).sum())
        scores[group] = recovered
    _finite("deletion scores", scores)
    _finite("support", support)
    if not torch.isclose(support.sum(), support.new_tensor(1.0), atol=2e-5, rtol=2e-5):
        raise RuntimeError("shared attention source support does not sum to one")
    queries = int(raw_queries.numel())
    # Each of QK, AV and oV is a matrix product. The tiny per-group Value
    # Gram and Gram/a contractions are additionally counted. This remains a
    # partial analytical estimate; indexing, allocation and launch costs are
    # intentionally not disguised as measured hardware FLOPs.
    ordinary_flops = 6 * heads * queries * raw * dim
    ordinary_flops += 2 * heads * groups * unit * unit * dim
    ordinary_flops += 2 * heads * queries * groups * unit * unit
    ordinary_flops += heads * queries * (8 * raw + 21 * groups)
    recovery_flops = 0
    if cancel_ids and cancellation_recovery == "deferred":
        recovery_flops += 4 * heads * queries * raw * dim
        recovery_flops += len(cancel_ids) * heads * queries * (2 * unit * dim + 8 * dim + 8)
    elif cancellation_recovery == "inline":
        recovery_flops += inline_tuples * (2 * unit * dim + 8 * dim + 8)
    recovery_flops += len(near_ids) * heads * queries * raw * (8 * dim + 18)
    return (
        scores,
        support,
        {
            "statistics_implementation": "optimized_rectangular_gram_fp32",
            "statistics_implementation_file": __file__,
            "source_query_count": queries,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "kv_repeat_count": repeats,
            "max_attention_chunk_shape": [heads, min(chunk, queries), raw],
            "full_attention_matrix_saved": False,
            "query_chunk_effective": chunk,
            "query_chunk_count": math.ceil(queries / chunk),
            "max_attention_mb": float(max_attention_mb),
            "group_chunk_setting_used_for": (
                "unused by inline sparse recovery; readout tuple batches are bounded at 16384"
                if cancellation_recovery == "inline"
                else "rare direct-vector cancellation recovery only; ordinary Gram scoring is vectorized across groups"
            ),
            "source_chunk_setting_used_for": "rare stable independently masked attention recovery only",
            "ordinary_qk_source_scans": 1,
            "cancellation_recovery": cancellation_recovery,
            "cancellation_qk_source_scans": int(
                bool(cancel_ids) and cancellation_recovery == "deferred"
            ),
            "inline_recovered_tuple_count": inline_tuples,
            "inline_readout_tuple_chunk_limit": (
                16384 if cancellation_recovery == "inline" else None
            ),
            "value_centering": "per-head source mean; exact common-translation identity",
            "stable_recomputed_group_ids": near_ids,
            "cancellation_recomputed_group_ids": cancel_ids,
            "attention_source_raw_count": raw,
            "statistics_dtype": "float32",
            "error_definition": "pi-weighted mean raw-query mean-head squared pre-output-projection readout difference",
            "attention_flops_estimated": int(ordinary_flops + recovery_flops),
            "flops_note": "Partial analytical scalar-operation estimate. Counts one QK scan, AV, oV, per-group Value Gram, Gram/a products, estimated normalization/reductions, and executed direct/masked recoveries. Inline cancellation repair reuses the current rectangle and counts only repaired readout tuples; deferred cancellation repair includes the repeated QK/AV scan. Excludes some reductions/index checks, allocation, transfers, memory traffic and launches; not measured hardware FLOPs.",
        },
    )


__all__ = ["group_deletion_statistics_optimized"]
