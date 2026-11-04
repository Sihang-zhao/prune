"""Optional persistent CUDA kernel for sequential local-coverage greedy."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _persistent_greedy(
        Text,
        Residual,
        Affinity,
        Table,
        Valid,
        Chosen,
        Gains,
        N: tl.constexpr,
        D: tl.constexpr,
        K,
        B: tl.constexpr,
    ):
        ids = tl.arange(0, B)
        text = tl.load(Text + ids, ids < N, 0.0)
        coverage = tl.full((B,), 0.0, tl.float32)
        selected = ids >= N
        for step in range(K):
            flow = tl.full((B,), 0.0, tl.float32)
            for slot in tl.static_range(D):
                source = tl.load(Table + ids * D + slot, ids < N, 0).to(tl.int32)
                valid = tl.load(Valid + ids * D + slot, ids < N, 0)
                similarity = tl.load(Affinity + ids * D + slot, ids < N, 0.0)
                weight = tl.load(Residual + source)
                current = tl.gather(coverage, source, axis=0)
                delta = tl.maximum(similarity - current, 0.0)
                flow = flow + tl.where(valid, weight * delta, 0.0)
            gain = tl.where(selected, -float("inf"), text + flow)
            highest = tl.max(gain, axis=0)
            best = tl.min(tl.where(gain == highest, ids, 2147483647), axis=0)
            tl.store(Chosen + step, best.to(tl.int64))
            tl.store(Gains + step, highest)
            selected = selected | (ids == best)
            for slot in tl.static_range(D):
                source = tl.load(Table + best * D + slot).to(tl.int32)
                valid = tl.load(Valid + best * D + slot)
                similarity = tl.load(Affinity + best * D + slot)
                coverage = tl.where(
                    valid & (ids == source), tl.maximum(coverage, similarity), coverage
                )

    @triton.jit
    def _persistent_local_greedy(
        Text,
        Residual,
        Affinity,
        Table,
        Valid,
        Cache,
        Chosen,
        Gains,
        N: tl.constexpr,
        D: tl.constexpr,
        K,
        B: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        R: tl.constexpr,
        A: tl.constexpr,
    ):
        ids = tl.arange(0, B)
        text = tl.load(Text + ids, ids < N, 0.0)
        coverage = tl.full((B,), 0.0, tl.float32)
        selected = ids >= N
        flow = tl.full((B,), 0.0, tl.float32)
        for slot in tl.static_range(D):
            initial_source = tl.load(Table + ids * D + slot, ids < N, 0).to(tl.int32)
            initial_permitted = tl.load(Valid + ids * D + slot, ids < N, 0)
            initial_affinity = tl.load(Affinity + ids * D + slot, ids < N, 0.0)
            initial_weight = tl.load(Residual + initial_source)
            flow += tl.where(
                initial_permitted, initial_weight * tl.maximum(initial_affinity, 0.0), 0.0
            )
        tl.store(Cache + ids, tl.where(selected, -float("inf"), text + flow), ids < N)
        tl.debug_barrier()
        offsets = tl.arange(0, A)
        side: tl.constexpr = 4 * R + 1
        for step in range(K):
            gain = tl.load(Cache + ids, ids < N, -float("inf"), volatile=True)
            highest = tl.max(gain, axis=0)
            best = tl.min(tl.where(gain == highest, ids, 2147483647), axis=0)
            tl.store(Chosen + step, best.to(tl.int64))
            tl.store(Gains + step, highest)
            selected = selected | (ids == best)
            if step + 1 < K:
                for slot in tl.static_range(D):
                    source = tl.load(Table + best * D + slot).to(tl.int32)
                    permitted = tl.load(Valid + best * D + slot)
                    affinity = tl.load(Affinity + best * D + slot)
                    coverage = tl.where(
                        permitted & (ids == source), tl.maximum(coverage, affinity), coverage
                    )
                ys = best // W + offsets // side - 2 * R
                xs = best % W + offsets % side - 2 * R
                permitted = (offsets < side * side) & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                affected = tl.where(permitted, ys * W + xs, 0)
                flow_local = tl.full((A,), 0.0, tl.float32)
                for slot in tl.static_range(D):
                    source = tl.load(Table + affected * D + slot).to(tl.int32)
                    valid_edge = tl.load(Valid + affected * D + slot)
                    affinity = tl.load(Affinity + affected * D + slot)
                    weight = tl.load(Residual + source)
                    current = tl.gather(coverage, source, axis=0)
                    flow_local += tl.where(
                        valid_edge, weight * tl.maximum(affinity - current, 0.0), 0.0
                    )
                direct = tl.load(Text + affected)
                taken = tl.gather(selected.to(tl.int32), affected, axis=0) != 0
                updated = tl.where(taken, -float("inf"), direct + flow_local)
                tl.store(Cache + affected, updated, permitted)
                tl.debug_barrier()


@torch.no_grad()
def greedy_direct_cover_cuda(
    text_mass, residual_mass, keep, affinity, table, valid, *, grid_shape=None, local_radius=1
):
    """Return sorted original IDs and sequential gains without host round trips.

    The objective and K sequential decisions are unchanged. Arithmetic uses
    FP32 with fusion disabled; small rounding differences from ATen reduction
    order are possible. All sample-dependent state is fresh for every call.
    """
    if triton is None:
        raise RuntimeError("cuda_fused requires Triton in the CUDA PyTorch environment")
    if text_mass.device.type != "cuda":
        raise ValueError("cuda_fused requires CUDA tensors")
    if text_mass.ndim != 1 or residual_mass.shape != text_mass.shape:
        raise ValueError("text/residual masses must be equal-length vectors")
    n = text_mass.numel()
    if not 0 < int(keep) <= n:
        raise ValueError("keep must lie in [1,N]")
    if (
        table.ndim != 2
        or table.shape[0] != n
        or affinity.shape != table.shape
        or valid.shape != table.shape
    ):
        raise ValueError("affinity/table/valid must share [N,degree] shape")
    if (
        text_mass.dtype != torch.float32
        or residual_mass.dtype != torch.float32
        or affinity.dtype != torch.float32
    ):
        raise TypeError("cuda_fused masses and affinities must be FP32")
    if table.dtype != torch.long or valid.dtype != torch.bool:
        raise TypeError("table must be int64 and valid must be bool")
    if any(x.device != text_mass.device for x in (residual_mass, affinity, table, valid)):
        raise ValueError("all inputs must share the same CUDA device")
    if n > 16384 or table.shape[1] > 25 or table.shape[1] < 1:
        raise ValueError("cuda_fused supports up to 16384 groups and 25 neighbor slots")
    tensors = [x.contiguous() for x in (text_mass, residual_mass, affinity, table, valid)]
    chosen = torch.empty(int(keep), device=text_mass.device, dtype=torch.long)
    gains = torch.empty(int(keep), device=text_mass.device, dtype=torch.float32)
    if grid_shape is None:
        _persistent_greedy[(1,)](
            *tensors,
            chosen,
            gains,
            n,
            table.shape[1],
            int(keep),
            triton.next_power_of_2(n),
            num_warps=8,
            enable_fp_fusion=False,
        )
    else:
        height, width = map(int, grid_shape)
        radius = int(local_radius)
        if (
            min(height, width) < 1
            or height * width != n
            or radius < 0
            or table.shape[1] != (2 * radius + 1) ** 2
        ):
            raise ValueError(
                "grid_shape/radius must describe the provided square-neighborhood graph"
            )
        cache = torch.empty_like(text_mass)
        _persistent_local_greedy[(1,)](
            *tensors,
            cache,
            chosen,
            gains,
            n,
            table.shape[1],
            int(keep),
            triton.next_power_of_2(n),
            height,
            width,
            radius,
            triton.next_power_of_2((4 * radius + 1) ** 2),
            num_warps=8,
            enable_fp_fusion=False,
        )
    return chosen.sort().values, gains


__all__ = ["greedy_direct_cover_cuda"]
