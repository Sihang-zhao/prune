"""Independent native-Qwen3-VL adapter for perturbation-guided coverage."""

from __future__ import annotations
import hashlib
import json
import math
import time
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None
from .base import _get
from .pruned import PrunedQwenVL, _inference_mode


def _require_runtime():
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for Qwen3-VL perturbation pruning")


def _sync(device):
    if torch is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


PERTURBATION_METHODS = {"q_flowcover_perturbation", "q_flowcover_perturbation_nocover"}


def decode_native_qkv(attn: Any, projected: Any, position_embeddings: Any):
    """Decode the already-computed native Qwen3-VL vision projection."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    kv_groups = int(getattr(attn, "num_key_value_groups", 1))
    if kv_groups != 1:
        raise RuntimeError(
            "native Qwen3-VL vision adapter requires its MHA qkv layout; GQA must have an explicit adapter"
        )
    if projected.ndim != 2 or int(projected.shape[1]) != 3 * heads * head_dim:
        raise RuntimeError(f"unexpected native vision qkv shape {tuple(projected.shape)}")
    q, k, v = (
        projected.reshape(projected.shape[0], 3, heads, head_dim).permute(1, 0, 2, 3).unbind(0)
    )
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    return (q, k, v)


@_inference_mode()
def verify_native_deletion(
    q,
    k,
    v,
    output_weight,
    query_groups,
    merge_unit,
    *,
    scale,
    candidate_count=2,
    query_weights=None,
    expected_scores=None,
    query_chunk_size=64,
):
    """One-off production-score check on real native tensors."""
    raw, heads, dim = map(int, q.shape)
    unit = int(merge_unit)
    groups = raw // unit
    query_ids = torch.tensor(list(query_groups), device=q.device, dtype=torch.long)
    if not query_ids.numel() or groups <= 1:
        return {"status": "skipped", "reason": "no legal full-minus-one reference"}
    raw_queries = (query_ids[:, None] * unit + torch.arange(unit, device=q.device)).flatten()
    if query_weights is None:
        pi = torch.zeros(groups, device=q.device).index_fill_(0, query_ids, 1.0 / query_ids.numel())
    else:
        pi = torch.as_tensor(query_weights, device=q.device, dtype=torch.float32)
    raw_weights = pi.index_select(0, query_ids).repeat_interleave(unit) / float(unit)
    query_set = set(map(int, query_groups))
    candidates = [group for group in range(groups) if group not in query_set][
        : int(candidate_count)
    ]
    if len(candidates) < min(int(candidate_count), groups):
        candidates.extend((group for group in range(groups) if group not in candidates))
        candidates = candidates[: min(int(candidate_count), groups)]
    key, value = (k.transpose(0, 1).float(), v.transpose(0, 1).float())
    weight = output_weight.float()
    full_score = torch.zeros(len(candidates), device=q.device)
    formula_score = torch.zeros_like(full_score)
    max_readout = torch.zeros_like(full_score)
    max_projected = torch.zeros_like(full_score)
    readout_matches = torch.ones(len(candidates), dtype=torch.bool, device=q.device)
    projected_matches = torch.ones_like(readout_matches)
    near_one_counts = torch.zeros(len(candidates), dtype=torch.long, device=q.device)
    query_chunk = min(int(query_chunk_size), 64, max(1, raw - 1))
    for start in range(0, raw_queries.numel(), query_chunk):
        query = q.index_select(0, raw_queries[start : start + query_chunk]).transpose(0, 1).float()
        logits = torch.matmul(query, key.transpose(-1, -2)) * float(scale)
        attention = torch.softmax(logits, dim=-1)
        full = torch.matmul(attention, value)
        local_weights = raw_weights[start : start + query_chunk]
        for ordinal, group in enumerate(candidates):
            removed = torch.arange(group * unit, (group + 1) * unit, device=q.device)
            allowed = torch.ones(raw, dtype=torch.bool, device=q.device).index_fill_(
                0, removed, False
            )
            deleted = torch.matmul(torch.softmax(logits[..., allowed], dim=-1), value[:, allowed])
            mass = attention.index_select(-1, removed).sum(-1)
            weighted_value = torch.matmul(
                attention.index_select(-1, removed), value.index_select(1, removed)
            )
            remaining = 1.0 - mass
            safe = remaining > 1e-05
            explicit = full - deleted
            formula = (weighted_value - mass[..., None] * full) / torch.where(
                safe, remaining, torch.ones_like(remaining)
            )[..., None]
            formula = torch.where(safe[..., None], formula, explicit)
            near_one_counts[ordinal] += (~safe).sum()
            projected_formula = F.linear(
                formula.transpose(0, 1).reshape(query.shape[1], heads * dim), weight, bias=None
            )
            projected_explicit = F.linear(
                explicit.transpose(0, 1).reshape(query.shape[1], heads * dim), weight, bias=None
            )
            max_readout[ordinal] = torch.maximum(
                max_readout[ordinal], (formula - explicit).abs().max()
            )
            max_projected[ordinal] = torch.maximum(
                max_projected[ordinal], (projected_formula - projected_explicit).abs().max()
            )
            readout_matches[ordinal] &= torch.isclose(
                formula, explicit, atol=2e-05, rtol=0.0002
            ).all()
            projected_matches[ordinal] &= torch.isclose(
                projected_formula, projected_explicit, atol=5e-05, rtol=0.0005
            ).all()
            formula_score[ordinal] += (formula.square().sum(-1).mean(0) * local_weights).sum()
            full_score[ordinal] += (explicit.square().sum(-1).mean(0) * local_weights).sum()
    if not bool(readout_matches.all()) or not bool(projected_matches.all()):
        raise RuntimeError(
            "closed-form deletion does not match independently deleted native attention output"
        )
    torch.testing.assert_close(formula_score, full_score, atol=2e-05, rtol=0.0005)
    if expected_scores is not None:
        production = torch.as_tensor(
            expected_scores, device=q.device, dtype=torch.float32
        ).index_select(0, torch.tensor(candidates, device=q.device))
        torch.testing.assert_close(production, full_score, atol=2e-05, rtol=0.0005)
    else:
        production = formula_score
    rows = [
        {
            "group_id": int(group),
            "max_abs_readout_error": float(max_readout[ordinal].item()),
            "max_abs_projected_error": float(max_projected[ordinal].item()),
            "near_one_explicit_count": int(near_one_counts[ordinal].item()),
            "production_deletion_score": float(production[ordinal].item()),
            "explicit_deletion_score": float(full_score[ordinal].item()),
        }
        for ordinal, group in enumerate(candidates)
    ]
    projection_flops = (
        4 * len(candidates) * raw_queries.numel() * (heads * dim) * output_weight.shape[0]
    )
    readout_flops = 2 * heads * raw_queries.numel() * raw * dim * (2 + len(candidates))
    readout_flops += 2 * len(candidates) * heads * raw_queries.numel() * unit * dim
    readout_flops += heads * raw_queries.numel() * (1 + len(candidates)) * raw * 5
    return {
        "status": "passed",
        "raw_query_count": int(raw_queries.numel()),
        "candidate_count": len(rows),
        "uses_all_production_queries_and_pi": True,
        "comparison": "FP32 fixed native-query readout, W_O difference and production D versus independent Key/Value deletion",
        "readout_atol": 2e-05,
        "readout_rtol": 0.0002,
        "projected_atol": 5e-05,
        "projected_rtol": 0.0005,
        "verification_flops_estimated": int(projection_flops + readout_flops),
        "candidates": rows,
    }


class Qwen3VLPerturbationRunner(PrunedQwenVL):
    """Explicit new-selector branch, with no dispatch through an old selector."""

    def __init__(
        self,
        *args: Any,
        compression_mode: str = "full19_gather",
        prune_method: str = "q_flowcover_perturbation",
        decision_layer_human: int = 18,
        perturbation_lambda: float = 1.0,
        perturbation_temperature: float = 1.0,
        perturbation_query_chunk: int = 256,
        perturbation_group_chunk: int = 32,
        perturbation_key_chunk: int = 1024,
        perturbation_local_radius: int = 1,
        perturbation_verify_candidates: int = 0,
        perturbation_compare_flow: bool = False,
        perturbation_selector_impl: str = "reference",
        perturbation_max_attention_mb: float = 512.0,
        perturbation_greedy_backend: str = "reference",
        **kwargs: Any,
    ) -> None:
        _require_runtime()
        method = str(prune_method).strip().lower()
        if method not in PERTURBATION_METHODS:
            raise ValueError(f"unsupported perturbation method {method!r}")
        if str(compression_mode) != "full19_gather":
            raise ValueError(
                "perturbation runner requires full19_gather: native block L+1 runs once before gathering; hard18/bridge are unsupported"
            )
        if int(decision_layer_human) <= 0:
            raise ValueError("decision_layer_human must be positive")
        if not math.isfinite(float(perturbation_lambda)) or float(perturbation_lambda) < 0:
            raise ValueError("perturbation_lambda must be finite and nonnegative")
        if (
            not math.isfinite(float(perturbation_temperature))
            or float(perturbation_temperature) <= 0
        ):
            raise ValueError("perturbation_temperature must be finite and positive")
        for name, value in (
            ("query", perturbation_query_chunk),
            ("group", perturbation_group_chunk),
            ("key", perturbation_key_chunk),
        ):
            if int(value) <= 0:
                raise ValueError(f"perturbation {name} chunk must be positive")
        if int(perturbation_local_radius) < 0:
            raise ValueError("perturbation_local_radius must be nonnegative")
        if int(perturbation_verify_candidates) < 0:
            raise ValueError("perturbation_verify_candidates must be nonnegative")
        if perturbation_selector_impl not in {"reference", "optimized"}:
            raise ValueError("selector implementation must be reference or optimized")
        if perturbation_greedy_backend not in {
            "reference",
            "local_refresh",
            "scripted_local_refresh",
        }:
            raise ValueError("unknown optimized greedy backend")
        if not math.isfinite(perturbation_max_attention_mb) or perturbation_max_attention_mb <= 0:
            raise ValueError("optimized attention memory budget must be finite and positive")
        if kwargs.get("visual_pruning", True) is not True:
            raise ValueError("the explicit perturbation runner requires visual_pruning=True")
        if kwargs.get("pruning_mode", "middle_text_early") != "middle_text_early":
            raise ValueError(
                "perturbation runner requires its explicit middle_text_early visual path"
            )
        if int(kwargs.get("pruning_layer", decision_layer_human)) != int(decision_layer_human):
            raise ValueError(
                "pruning_layer must equal decision_layer_human; layer labels cannot silently diverge"
            )
        kwargs.setdefault("visual_pruning", True)
        kwargs.setdefault("pruning_mode", "middle_text_early")
        kwargs.setdefault("pruning_layer", int(decision_layer_human))
        kwargs.setdefault("pruning_strategy", "topk")
        PrunedQwenVL.__init__(self, *args, **kwargs)
        self.compression_mode = "full19_gather"
        self.prune_method = method
        self.decision_layer_human = int(decision_layer_human)
        self.decision_block_index = self.decision_layer_human - 1
        self.bridge_block_index = self.decision_layer_human
        depth = int(_get(_get(self.config, "vision_config"), "depth", len(self.vit.blocks)))
        if depth != len(self.vit.blocks):
            raise RuntimeError(
                "native vision depth does not match its blocks; refusing to guess layer mapping"
            )
        if self.bridge_block_index >= len(self.vit.blocks):
            raise ValueError(
                f"score layer {self.decision_layer_human} requires a following visual block; depth={len(self.vit.blocks)}"
            )
        self.perturbation_lambda = float(perturbation_lambda)
        self.perturbation_temperature = float(perturbation_temperature)
        self.perturbation_query_chunk = int(perturbation_query_chunk)
        self.perturbation_group_chunk = int(perturbation_group_chunk)
        self.perturbation_key_chunk = int(perturbation_key_chunk)
        self.perturbation_local_radius = int(perturbation_local_radius)
        self.perturbation_verify_candidates = int(perturbation_verify_candidates)
        self.perturbation_compare_flow = bool(perturbation_compare_flow)
        self.perturbation_selector_impl = perturbation_selector_impl
        self.perturbation_max_attention_mb = float(perturbation_max_attention_mb)
        self.perturbation_greedy_backend = perturbation_greedy_backend
        self.att_stats_mode = "none"
        self.save_att_stats = False
        self.max_debug_samples = 0
        self._debug_samples_seen = 0
        self._last_compression_metadata = {}
        self._last_stage_timings = {}
        self._last_att_stats = {}
        self._last_q_flowcover_debug = []
        self._last_perturbation_shape_flops = 0

    def stages(self):
        return (
            "Full visual prefix",
            f"Native block {self.decision_layer_human + 1}",
            "Perturbation coverage selector",
            "One complete-group gather",
            "Compact suffix",
            "LLM",
        )

    def _vision_forward_middle_early(self, pixel_values, image_grid_thw, text_tokens):
        _require_runtime()
        if self.prune_method not in PERTURBATION_METHODS:
            raise RuntimeError("perturbation runner did not hit its explicit new selector branch")
        if self.compression_mode != "full19_gather":
            raise RuntimeError("perturbation runner execution boundary changed unexpectedly")
        vit = self.vit
        if torch.is_grad_enabled():
            raise RuntimeError(
                "perturbation inference requires torch.no_grad() or torch.inference_mode()"
            )
        if bool(vit.training):
            raise RuntimeError("perturbation inference requires model.eval()")
        merge_size = int(
            getattr(vit, "spatial_merge_size", 0)
            or _get(_get(self.config, "vision_config"), "spatial_merge_size", 1)
        )
        if merge_size <= 0:
            raise RuntimeError("invalid native spatial_merge_size")
        merge_unit = merge_size**2
        rows = image_grid_thw.detach().reshape(-1, 3).cpu().tolist()
        if not rows or any(
            (int(t) != 1 or int(h) % merge_size or int(w) % merge_size for t, h, w in rows)
        ):
            raise RuntimeError(
                "perturbation adapter requires still-image T=1 grids divisible by spatial_merge_size"
            )
        grids = [(int(h) // merge_size, int(w) // merge_size) for _, h, w in rows]
        counts = [h * w for h, w in grids]
        raw_counts = [n * merge_unit for n in counts]
        budgets = [self._keep_count(n) for n in counts]
        self._last_native_final_features = []
        self._last_native_deepstack = []
        self._last_early_selections = []
        self._last_cached_visual = None
        self._last_early_original_counts = list(counts)
        self._last_vision_layer_token_counts = []
        stage = {}
        qkv_times = []
        _sync(self.device)
        prefix_started = time.perf_counter()
        hidden = vit.patch_embed(pixel_values)
        if int(hidden.shape[0]) != sum(raw_counts):
            raise RuntimeError("native patch sequence does not match image_grid_thw")
        hidden = hidden + vit.fast_pos_embed_interpolate(image_grid_thw)
        rotary = vit.rot_pos_emb(image_grid_thw).reshape(hidden.shape[0], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        positions = (emb.cos(), emb.sin())
        cu = F.pad(
            torch.tensor(raw_counts, dtype=torch.int32, device=hidden.device).cumsum(
                0, dtype=torch.int32
            ),
            (1, 0),
            value=0,
        )
        deep_indexes = tuple((int(i) for i in vit.deepstack_visual_indexes))
        deepstack = []
        score_hidden = None
        score_projected = None
        selections = None
        debug = []
        compact_counts = None
        keep_raw = None
        capture_count = 0
        full_budget = all((k == n for k, n in zip(budgets, counts)))
        suffix_started = None
        for index, block in enumerate(vit.blocks):
            self._last_vision_layer_token_counts.append(int(hidden.shape[0]))
            if index == self.bridge_block_index:
                _sync(self.device)
                stage["visual_prefix"] = time.perf_counter() - prefix_started
                stage["prefix"] = stage["visual_prefix"]
                captured = []
                qkv_start = []
                qkv_events = []

                def before_projection(_module, _inputs):
                    if hidden.is_cuda:
                        first = torch.cuda.Event(enable_timing=True)
                        first.record(torch.cuda.current_stream(self.device))
                        qkv_events.append([first])
                    else:
                        qkv_start.append(time.perf_counter())

                def after_projection(_module, _inputs, output):
                    nonlocal capture_count
                    if len(captured):
                        raise RuntimeError("native next block repeated its QKV projection")
                    captured.append(output)
                    capture_count += 1
                    if hidden.is_cuda:
                        last = torch.cuda.Event(enable_timing=True)
                        last.record(torch.cuda.current_stream(self.device))
                        qkv_events[-1].append(last)
                    else:
                        qkv_times.append(time.perf_counter() - qkv_start.pop())

                handles = (
                    []
                    if full_budget
                    else [
                        block.attn.qkv.register_forward_pre_hook(before_projection),
                        block.attn.qkv.register_forward_hook(after_projection),
                    ]
                )
                full_started = time.perf_counter()
                try:
                    hidden = block(hidden, cu_seqlens=cu, position_embeddings=positions)
                finally:
                    for handle in handles:
                        handle.remove()
                _sync(self.device)
                stage["visual_full_next"] = time.perf_counter() - full_started
                if qkv_events:
                    qkv_times.extend(
                        (float(pair[0].elapsed_time(pair[1])) / 1000.0 for pair in qkv_events)
                    )
                if index in deep_indexes:
                    deepstack.append(vit.deepstack_merger_list[deep_indexes.index(index)](hidden))
                if score_hidden is None or score_projected is None:
                    raise RuntimeError("next block was reached without complete score-layer state")
                selector_started = time.perf_counter()
                decode_events = None
                decode_seconds = 0.0
                if full_budget:
                    q_views = k_views = v_views = [None] * len(counts)
                else:
                    if len(captured) != 1:
                        raise RuntimeError(
                            "native next-block QKV capture did not execute exactly once"
                        )
                    decode_started = time.perf_counter()
                    if hidden.is_cuda:
                        decode_events = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        decode_events[0].record(torch.cuda.current_stream(self.device))
                    q, k, v = decode_native_qkv(block.attn, captured.pop(), positions)
                    if decode_events is not None:
                        decode_events[1].record(torch.cuda.current_stream(self.device))
                    else:
                        decode_seconds = time.perf_counter() - decode_started
                    q_views, k_views, v_views = (
                        list(value.split(raw_counts, dim=0)) for value in (q, k, v)
                    )
                implementation = getattr(self, "perturbation_selector_impl", "reference")
                if implementation == "reference":
                    from pruning.reference import (
                        select_q_flowcover_perturbation_views as select_views,
                    )

                    implementation_kwargs = {}
                elif implementation == "optimized":
                    from pruning.selector import (
                        select_q_flowcover_perturbation_optimized_views as select_views,
                    )

                    implementation_kwargs = {
                        "max_attention_mb": self.perturbation_max_attention_mb,
                        "greedy_backend": self.perturbation_greedy_backend,
                    }
                else:
                    raise RuntimeError(
                        "unknown perturbation implementation; no fallback is permitted"
                    )
                selections, debug = select_views(
                    list(score_projected.split(counts, dim=0)),
                    text_tokens,
                    q_views,
                    k_views,
                    v_views,
                    list(score_hidden.split(raw_counts, dim=0)),
                    budgets,
                    raw_counts,
                    merge_unit,
                    self.prune_method,
                    grids,
                    scale=float(block.attn.scaling),
                    query_chunk_size=self.perturbation_query_chunk,
                    group_chunk_size=self.perturbation_group_chunk,
                    source_chunk_size=self.perturbation_key_chunk,
                    temperature=self.perturbation_temperature,
                    perturbation_lambda=self.perturbation_lambda,
                    local_radius=self.perturbation_local_radius,
                    compare_flow=self.perturbation_compare_flow,
                    **implementation_kwargs,
                )
                _sync(self.device)
                stage["selector"] = time.perf_counter() - selector_started
                if decode_events is not None:
                    decode_seconds = float(decode_events[0].elapsed_time(decode_events[1])) / 1000.0
                stage["qkv_native_projection"] = sum(qkv_times) if not full_budget else None
                stage["qkv_rope_decode"] = decode_seconds
                stage["native_projection"] = stage["qkv_native_projection"]
                stage["rope_decode"] = decode_seconds
                stage["qkv_acquisition"] = sum(qkv_times) + decode_seconds
                stage["qkv_acquisition_overlaps_selector"] = bool(not full_budget)
                stage["qkv_acquisition_overlaps_native_block"] = True
                stage["flow_and_perturbation_share_attention_scan"] = True
                stage["selector_flops"] = sum((int(row.get("selector_flops", 0)) for row in debug))
                for key in (
                    "text_cos",
                    "perturbation",
                    "flow_attention",
                    "local_graph",
                    "greedy",
                    "flow_comparison",
                    "diagnostic_serialization",
                    "selector_primary",
                ):
                    values = [row.get(key + "_seconds") for row in debug]
                    stage[key] = (
                        sum((float(value) for value in values))
                        if all((value is not None for value in values))
                        else None
                    )
                for row in debug:
                    row["score_layer"] = self.decision_layer_human
                    row["perturbation_layer"] = self.decision_layer_human + 1
                    row["flow_comparison_layer"] = self.decision_layer_human + 1
                    row["flow_comparison_identity"] = "same-next-layer attention-flow control"
                if not full_budget and self.perturbation_verify_candidates:
                    verify_started = time.perf_counter()
                    verify_view = next(
                        (i for i, row in enumerate(debug) if row.get("statistics_computed"))
                    )
                    debug[verify_view]["numerical_verification"] = verify_native_deletion(
                        q_views[verify_view],
                        k_views[verify_view],
                        v_views[verify_view],
                        block.attn.proj.weight,
                        debug[verify_view]["query_group_index"],
                        merge_unit,
                        scale=float(block.attn.scaling),
                        candidate_count=self.perturbation_verify_candidates,
                        query_weights=debug[verify_view]["pi"],
                        expected_scores=debug[verify_view]["deletion_scores"],
                        query_chunk_size=self.perturbation_query_chunk,
                    )
                    _sync(self.device)
                    stage["verification"] = time.perf_counter() - verify_started
                stage.setdefault("verification", None)
                q_views = k_views = v_views = None
                score_hidden = score_projected = None
                if not full_budget:
                    del q, k, v
                index_started = time.perf_counter()
                compact_counts = [int(sel.indices.numel()) for sel in selections]
                _, keep_raw = self._raw_indices(selections, raw_counts, merge_unit, hidden.device)
                if keep_raw.numel() != sum(compact_counts) * merge_unit or int(
                    torch.unique(keep_raw).numel()
                ) != int(keep_raw.numel()):
                    raise RuntimeError(
                        "complete-group expansion is not unique or has an incorrect length"
                    )
                if keep_raw.numel() > 1 and (not bool(torch.all(keep_raw[1:] > keep_raw[:-1]))):
                    raise RuntimeError("gather indices must preserve original native packing order")
                hidden = hidden.index_select(0, keep_raw)
                deepstack = [
                    torch.cat(
                        [
                            view.index_select(0, sel.indices)
                            for view, sel in zip(feature.split(counts, dim=0), selections)
                        ]
                    )
                    for feature in deepstack
                ]
                positions = tuple((value.index_select(0, keep_raw) for value in positions))
                compact_raw = [n * merge_unit for n in compact_counts]
                cu = F.pad(
                    torch.tensor(compact_raw, dtype=torch.int32, device=hidden.device).cumsum(
                        0, dtype=torch.int32
                    ),
                    (1, 0),
                    value=0,
                )
                _sync(self.device)
                stage["index_expand"] = time.perf_counter() - index_started
                stage["selector_and_index"] = stage["selector"] + stage["index_expand"]
                stage["total_pruning"] = stage["selector_and_index"] + stage["text_projection"]
                stage["selection_including_text_projection"] = stage["total_pruning"]
                stage["primary_total_pruning"] = (
                    stage["selector_primary"] + stage["index_expand"] + stage["text_projection"]
                    if stage["selector_primary"] is not None
                    else None
                )
                suffix_started = time.perf_counter()
                continue
            hidden = block(hidden, cu_seqlens=cu, position_embeddings=positions)
            if index in deep_indexes:
                deepstack.append(vit.deepstack_merger_list[deep_indexes.index(index)](hidden))
            if index == self.decision_block_index:
                score_hidden = hidden
                _sync(self.device)
                score_started = time.perf_counter()
                score_projected = self._project_layer(hidden, self.pruning_layer, image_grid_thw)
                _sync(self.device)
                stage["text_projection"] = time.perf_counter() - score_started
        if selections is None or compact_counts is None or keep_raw is None:
            raise RuntimeError("new perturbation selector branch did not execute")
        _sync(self.device)
        stage["visual_suffix"] = time.perf_counter() - suffix_started
        merge_started = time.perf_counter()
        final = vit.merger(hidden)
        _sync(self.device)
        stage["merger"] = time.perf_counter() - merge_started
        if final.shape[0] != sum(compact_counts):
            raise RuntimeError("native merger does not match selected group budget")
        self._last_native_final_features = list(final.split(compact_counts, dim=0))
        self._last_native_deepstack = [
            list(feature.split(compact_counts, dim=0)) for feature in deepstack
        ]
        self._last_early_selections = list(selections)
        self._last_q_flowcover_debug = debug
        self._last_stage_timings = stage
        self._last_att_stats = {}
        self._last_compression_metadata = {
            "algorithm_id": self.prune_method,
            "selector_impl": getattr(self, "perturbation_selector_impl", "reference"),
            "algorithm_file": __file__,
            "selector_file": debug[0].get("implementation_file") if debug else None,
            "selector_function": debug[0].get("implementation_function") if debug else None,
            "native_verification": next(
                (row["numerical_verification"] for row in debug if "numerical_verification" in row),
                None,
            ),
            "algorithm_function": f"{type(self).__name__}._vision_forward_middle_early",
            "compression_mode": self.compression_mode,
            "prune_method": self.prune_method,
            "decision_layer_human": self.decision_layer_human,
            "decision_block_index": self.decision_block_index,
            "score_layer": self.decision_layer_human,
            "perturbation_layer": self.decision_layer_human + 1,
            "actual_prune_after_layer": self.decision_layer_human + 1,
            "bridge_block_index": self.bridge_block_index,
            "raw_patch_count": sum(raw_counts),
            "merged_group_count": sum(counts),
            "retained_group_count": sum(compact_counts),
            "actual_llm_visual_tokens": sum(compact_counts),
            "keep_idx": [sel.indices.detach().cpu().tolist() for sel in selections],
            "image_grid_thw": rows,
            "layer_token_counts": list(self._last_vision_layer_token_counts),
            "raw_keep_idx": keep_raw.detach().cpu().tolist(),
            "raw_keep_idx_hash": hashlib.sha256(
                json.dumps(keep_raw.detach().cpu().tolist(), separators=(",", ":")).encode()
            ).hexdigest(),
            "q_flowcover": debug,
            "qkv_capture_count": capture_count,
            "qkv_projection_replay_count": 0,
            "actual_gather_count": 1,
            "flops_coverage_note": "Executed-shape matrix-operation estimate includes raw visual attention, native QKV/MLP, score/final/DeepStack mergers and selector operations; omits patch Conv3d, normalization, native softmax/activation, RoPE, allocation and data movement.",
        }
        self._last_compression_metadata["flops_note"] = self._last_compression_metadata[
            "flops_coverage_note"
        ]
        self._last_perturbation_shape_flops = self._visual_shape_flops(
            raw_counts, compact_counts, merge_unit, deep_indexes
        )
        return {}

    def _visual_shape_flops(self, raw_counts, compact_counts, merge_unit, deep_indexes):
        """Count the actual full/compact matrix shapes, rather than old cut=L."""
        vision = _get(self.config, "vision_config")
        hidden = int(_get(vision, "hidden_size"))
        intermediate = int(_get(vision, "intermediate_size"))
        out_hidden = int(_get(vision, "out_hidden_size"))
        full_layers = self.decision_layer_human + 1
        compact_raw = [n * merge_unit for n in compact_counts]
        per_token = 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
        flops = 0
        for layer in range(len(self.vit.blocks)):
            lengths = raw_counts if layer < full_layers else compact_raw
            flops += sum(lengths) * per_token + 4 * hidden * sum((n * n for n in lengths))
        group_width = hidden * merge_unit
        merger_row = 2 * group_width * (group_width + out_hidden)
        flops += sum(raw_counts) // merge_unit * merger_row
        flops += sum(compact_counts) * merger_row
        for index in deep_indexes:
            n = sum(raw_counts) // merge_unit if index < full_layers else sum(compact_counts)
            flops += n * merger_row
        return int(flops)

    def answer(self, *args: Any, **kwargs: Any):
        output = super().answer(*args, **kwargs)
        for key, value in self._last_compression_metadata.items():
            setattr(output, key, value)
        output.stage_timings = dict(self._last_stage_timings)
        output.att_stats = dict(self._last_att_stats)
        output.q_flowcover_debug = list(self._last_q_flowcover_debug)
        output.selector_flops = float(
            sum((int(row.get("selector_flops", 0)) for row in self._last_q_flowcover_debug))
        )
        output.flops = int(output.flops) + int(output.selector_flops)
        output.peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else None
        )
        if self._last_compression_metadata.get("algorithm_id") != self.prune_method:
            raise RuntimeError(
                "new perturbation selector branch did not execute; no fallback is permitted"
            )
        vision = _get(self.config, "vision_config")
        hidden = int(_get(vision, "hidden_size"))
        intermediate = int(_get(vision, "intermediate_size"))
        depth = int(_get(vision, "depth", len(self.vit.blocks)))
        old_token_layers = (
            self.decision_layer_human * output.original_image_tokens
            + (depth - self.decision_layer_human) * output.image_tokens
        )
        old_visual = old_token_layers * 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
        verify_flops = sum(
            (
                int(row.get("numerical_verification", {}).get("verification_flops_estimated", 0))
                for row in self._last_q_flowcover_debug
            )
        )
        output.flops = (
            int(output.flops)
            - int(old_visual)
            + int(self._last_perturbation_shape_flops)
            + verify_flops
        )
        output.verification_flops = verify_flops
        output.stage_timings.update(
            {
                "llm": float(output.generation_seconds),
                "end_to_end": float(output.seconds),
                "llm_prefill_is_host_hook_estimate": True,
                "llm_timing_scope": "native generation wall time with final GPU synchronization; text decoding separately reported",
            }
        )
        self._last_stage_timings = dict(output.stage_timings)
        return output


__all__ = [
    "Qwen3VLPerturbationRunner",
    "PERTURBATION_METHODS",
    "decode_native_qkv",
    "verify_native_deletion",
]
