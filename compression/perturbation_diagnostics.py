"""CPU-only, post-inference diagnostics for the perturbation selector.

No tensors or image features are cached here.  The adapter supplies one
record per *actual input image view*, with native row-major group IDs and its
verified group grid.  Rendering never infers a square grid from token count.
This optional analysis runs outside the measured inference interval.
"""
from __future__ import annotations

import colorsys
import csv
import hashlib
import json
import math
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _vector(record: Mapping[str, Any], name: str, count: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"diagnostics require {name}; unavailable scores cannot be fabricated")
    value = _array(record[name])
    if value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {count} finite group values, got {value.shape}")
    return value


def _indices(record: Mapping[str, Any], name: str, count: int, *, optional: bool = False) -> np.ndarray | None:
    if name not in record:
        if optional:
            return None
        raise ValueError(f"diagnostics require {name}")
    raw = _array(record[name])
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
        raise ValueError(f"{name} must contain integer original group IDs")
    value = raw.astype(np.int64)
    if np.unique(value).size != value.size or np.any(value < 0) or np.any(value >= count):
        raise ValueError(f"{name} has duplicate or invalid original group IDs")
    return value


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ascending 1-based midranks, including exact ties (Spearman contract)."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (float(start) + float(end - 1)) + 1.0
    return ranks


def spearman_correlation(left: Any, right: Any) -> float | None:
    """Return None for constant/undefined data rather than inventing zero."""
    left, right = _array(left), _array(right)
    if left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be equal-length finite vectors")
    if left.size < 2:
        return None
    x, y = _average_ranks(left), _average_ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return None if denominator == 0 else float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _mask(indices: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros(count, dtype=bool)
    result[indices] = True
    return result


def analyze_view(
    record: Mapping[str, Any], *, low_text_ratio: float = 0.6,
    high_perturbation_ratio: float = 0.1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate identity/mapping and calculate explicitly defined diagnostics.

    Quantile thresholds below are *analysis only*, never protection budgets or
    online selector hyperparameters.  Boundary ties are all included.
    """
    if not 0 < low_text_ratio <= 1 or not 0 < high_perturbation_ratio <= 1:
        raise ValueError("diagnostic ratios must lie in (0,1]")
    count = int(record["N"])
    grid = tuple(int(x) for x in record["grid"])
    if count <= 0 or len(grid) != 2 or min(grid) <= 0 or math.prod(grid) != count:
        raise ValueError(f"verified group grid {grid} is inconsistent with N={count}")
    if record.get("group_order", "row_major") != "row_major":
        raise ValueError("image diagnostics require explicit native row_major group order")
    scores = _vector(record, "text_scores", count)
    computed = bool(record.get("statistics_computed", True))
    if computed:
        deletion = _vector(record, "deletion_scores", count)
        support = _vector(record, "support", count)
    else:
        if int(record["K"]) != count or _array(record.get("deletion_scores", [])).size or _array(record.get("support", [])).size:
            raise ValueError("only an explicit full-budget no-op can omit measured D/support")
        # These NA values are a labelled per-ID export of unmeasured data,
        # never measured zero perturbation/support.
        deletion = np.full(count, np.nan)
        support = np.full(count, np.nan)
    if computed and (np.any(deletion < 0) or np.any(support < 0)):
        raise ValueError("deletion errors and incoming attention mass must be nonnegative")
    query = _indices(record, "query_group_index", count)
    text = _indices(record, "text_topk", count)
    selected = _indices(record, "final_group_index", count)
    flow = _indices(record, "flow_group_index", count, optional=True)
    keep = int(record["K"])
    if keep < 1 or keep > count or len(selected) != keep or len(text) != keep:
        raise ValueError("text and final diagnostics must both use the exact K budget")
    if flow is not None and len(flow) != keep:
        raise ValueError("flow comparison must use exactly the same K budget")
    text_mask, selected_mask = _mask(text, count), _mask(selected, count)
    low_threshold = float(np.quantile(scores, low_text_ratio))
    high_threshold = float(np.quantile(deletion, 1.0 - high_perturbation_ratio)) if computed else None
    low = scores <= low_threshold
    scale = float(np.max(deletion)) if computed else 0.0
    # A constant/near-zero score cannot identify unusually sensitive groups.
    informative = computed and scale > np.finfo(np.float32).tiny and np.ptp(deletion) > scale * 1e-6
    high = (deletion >= high_threshold) if informative else np.zeros(count, dtype=bool)
    target = high & low & ~_mask(query, count)
    outside_text = target & ~text_mask
    normalized = (deletion / deletion.sum() if deletion.sum() > 0 else np.zeros_like(deletion)) if computed else np.full(count, np.nan)
    rank = _average_ranks(-deletion) if computed else np.full(count, np.nan)
    rank_q = (rank - 1.0) / max(count - 1, 1)
    summary: dict[str, Any] = {
        "N": count, "K": keep, "K_over_N": keep / count,
        "removed_fraction": 1.0 - keep / count, "grid": list(grid),
        "merge_unit": int(record.get("merge_unit", 0)),
        "group_order": "native_row_major", "group_id_scope": "view_local_original",
        "global_group_offset": record.get("global_group_offset"),
        "query_group_count": len(query), "query_group_index": query.tolist(),
        "text_group_index": text.tolist(), "final_group_index": selected.tolist(),
        "flow_group_index": None if flow is None else flow.tolist(),
        "flow_comparison_available": flow is not None,
        "flow_comparison_attention_layer": record.get("flow_comparison_attention_layer", record.get("flow_comparison_layer")),
        "flow_comparison_semantics": "Source-controlled flow selection from this diagnostic's Q/K; not an archived baseline answer or result.",
        "algorithm_id": record.get("algorithm_id"),
        "score_layer": record.get("score_layer"),
        "perturbation_layer": record.get("perturbation_layer"),
        "gather_after_layer": record.get("gather_after_layer", record.get("actual_prune_after_layer")),
        "statistics_computed": computed,
        "statistics_skipped_reason": record.get("statistics_skipped_reason"),
        "deletion_scores_informative": bool(informative) if computed else None,
        "perturbation_top10": [
            {"group_id": int(index), "deletion_score": float(deletion[index]),
             "midrank": float(rank[index])}
            for index in (np.argsort(-deletion, kind="stable")[:min(10, count)] if computed else [])
        ],
        "spearman_deletion_vs_support": spearman_correlation(deletion, support) if computed else None,
        "high_perturbation_low_text": {
            "definition": "text <= own-view quantile; D >= own-view quantile; outside query P; analysis only; includes boundary ties",
            "low_text_ratio": low_text_ratio, "high_perturbation_ratio": high_perturbation_ratio,
            "text_threshold": low_threshold, "deletion_threshold": high_threshold,
            "count": int(target.sum()) if computed else None,
            "retained_count": int((target & selected_mask).sum()) if computed else None,
            "text_baseline_retained_count": int((target & text_mask).sum()) if computed else None,
            "recovered_outside_text_topk": int((outside_text & selected_mask).sum()) if computed else None,
            "group_index": np.flatnonzero(target).tolist() if computed else None,
        },
        "text_vs_final": {
            "intersection": np.intersect1d(text, selected).tolist(),
            "text_only": np.setdiff1d(text, selected).tolist(),
            "final_only": np.setdiff1d(selected, text).tolist(),
        },
        "flow_vs_final": None if flow is None else {
            "intersection": np.intersect1d(flow, selected).tolist(),
            "flow_only": np.setdiff1d(flow, selected).tolist(),
            "final_only": np.setdiff1d(selected, flow).tolist(),
        },
        "cross_layer_persistence": {
            "measured": False,
            "reason": "One decision/attention layer per inference. Cross-layer persistence requires a separate multi-layer diagnostic run.",
        },
    }
    arrays = {
        "group_id": np.arange(count, dtype=np.int64),
        "group_row": np.arange(count, dtype=np.int64) // grid[1],
        "group_col": np.arange(count, dtype=np.int64) % grid[1],
        "text_scores": scores, "support": support, "deletion_scores": deletion,
        "deletion_scores_normalized": normalized, "deletion_rank": rank,
        "deletion_rank_q": rank_q, "query_mask": _mask(query, count),
        "text_keep_mask": text_mask, "final_keep_mask": selected_mask,
        "high_perturbation_low_text": target if computed else np.full(count, np.nan),
    }
    if flow is not None:
        arrays["flow_keep_mask"] = _mask(flow, count)
    return summary, arrays


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:  # portable Pillow fallback
        return ImageFont.load_default(size=size)


def _rainbow(values: np.ndarray) -> np.ndarray:
    # Blue/violet low -> red high, matching the project's rainbow heatmaps.
    flat = np.clip(values, 0, 1).reshape(-1)
    colors = np.asarray([colorsys.hsv_to_rgb((1.0 - value) * 0.74, 0.85, 0.98) for value in flat])
    return (255 * colors.reshape(*values.shape, 3)).astype(np.uint8)


def _image_panel(
    image: Image.Image, title: str, *, values: np.ndarray | None = None,
    grid: tuple[int, int] | None = None, limits: tuple[float, float] | None = None,
    alpha: float = 0.45, highlights: Sequence[int] = (),
    keep_mask: np.ndarray | None = None, categorical: np.ndarray | None = None,
    subtitle: str = "",
) -> Image.Image:
    panel = Image.new("RGB", (720, 720), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((22, 15), title, fill="#172b43", font=_font(23, True))
    if subtitle:
        draw.text((22, 49), subtitle[:94], fill="#566575", font=_font(14))
    base = ImageOps.contain(image.convert("RGB"), (676, 565), Image.Resampling.LANCZOS)
    if values is not None:
        if grid is None or limits is None:
            raise ValueError("heatmap requires verified grid and explicit limits")
        lower, upper = limits
        normalized = np.zeros_like(values) if upper == lower else np.clip((values - lower) / (upper - lower), 0, 1)
        color = Image.fromarray(_rainbow(normalized.reshape(grid))).resize(base.size, Image.Resampling.NEAREST)
        base = Image.blend(base, color, alpha)
    if keep_mask is not None:
        assert grid is not None
        # White means removed, exactly as the selection-mask convention.
        removed = Image.fromarray((~keep_mask.reshape(grid) * 180).astype(np.uint8)).resize(base.size, Image.Resampling.NEAREST)
        base.paste(Image.new("RGB", base.size, "white"), mask=removed)
    if categorical is not None:
        assert grid is not None
        palette = np.asarray([[0, 0, 0], [74, 132, 184], [233, 153, 57], [54, 157, 128]], dtype=np.uint8)
        colors = Image.fromarray(palette[categorical.reshape(grid)]).resize(base.size, Image.Resampling.NEAREST)
        opacity = Image.fromarray((categorical.reshape(grid) > 0).astype(np.uint8) * 140).resize(base.size, Image.Resampling.NEAREST)
        base.paste(colors, mask=opacity)
    x, y = (720 - base.width) // 2, 88 + (565 - base.height) // 2
    panel.paste(base, (x, y))
    if highlights:
        assert grid is not None
        for group in highlights:
            row, col = divmod(int(group), grid[1])
            x0, y0 = x + col * base.width / grid[1], y + row * base.height / grid[0]
            x1, y1 = x + (col + 1) * base.width / grid[1], y + (row + 1) * base.height / grid[0]
            draw.rectangle((x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)), outline="#fff900", width=2)
            label = str(group)
            bbox = draw.textbbox((x0, y0), label, font=_font(11, True))
            draw.rectangle(bbox, fill="#172b43")
            draw.text((x0, y0), label, fill="white", font=_font(11, True))
    if limits is not None:
        gradient = Image.fromarray(_rainbow(np.linspace(0, 1, 620)[None, :])).resize((620, 14))
        panel.paste(gradient, (50, 670))
        draw.text((50, 690), f"{limits[0]:.5g}", fill="#354455", font=_font(13))
        upper_label = f"{limits[1]:.5g}"
        draw.text((670 - draw.textlength(upper_label, font=_font(13)), 690), upper_label, fill="#354455", font=_font(13))
    return panel


def _text_panel(lines: Sequence[str], title: str = "Question / answer / protocol") -> Image.Image:
    panel = Image.new("RGB", (720, 720), "#f7f9fb")
    draw = ImageDraw.Draw(panel)
    draw.text((24, 20), title, fill="#172b43", font=_font(23, True))
    y = 72
    for paragraph in lines:
        for line in textwrap.wrap(str(paragraph), width=66, break_long_words=True) or [""]:
            draw.text((24, y), line, fill="#253c50", font=_font(17))
            y += 26
            if y > 680:
                draw.text((24, 688), "Complete text is saved in summary.json", fill="#566575", font=_font(13))
                return panel
        y += 10
    return panel


def _safe_sample_name(sample_id: Any) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    digest = hashlib.sha256(str(sample_id).encode()).hexdigest()[:8]
    return f"sample_{label[:64]}_{digest}"


def save_perturbation_diagnostics(
    output_dir: str | Path, sample_id: Any, images: Sequence[Image.Image],
    views: Sequence[Mapping[str, Any]], *, question: str,
    options: Any = None, ground_truth: Any = None, answer: str = "",
    extra: Mapping[str, Any] | None = None, alpha: float = 0.45,
) -> dict[str, Any]:
    """Save independent panels, original arrays, CSV and honest comparisons.

    ``images`` must be the exact prepared input views (not the dataset contact
    sheet).  ``views`` is the corresponding per-view selector diagnostic list.
    Call this only after inference timing/peak-memory readings are complete.
    """
    started = time.perf_counter()
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    if len(images) != len(views) or not images:
        raise ValueError("diagnostics need exactly one verified record per actual image view")
    sample_dir = Path(output_dir) / _safe_sample_name(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "sample_id": sample_id, "question": question, "options": options,
        "ground_truth": ground_truth, "answer": answer,
        "timing_scope": "post-inference CPU diagnostics; excluded from inference latency",
        "extra": dict(extra or {}), "views": [],
    }
    for view_index, (image, record) in enumerate(zip(images, views)):
        summary, arrays = analyze_view(record)
        grid = tuple(summary["grid"])
        folder = sample_dir / f"view_{view_index:03d}"
        subfig = folder / "subfig"
        subfig.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(folder / "input.png", dpi=(300, 300))
        np.savez_compressed(folder / "arrays.npz", **arrays)
        count = summary["N"]
        with (folder / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(arrays)
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows([[
                arrays[key][index].item() if np.isfinite(arrays[key][index]) else "NA"
                for key in fields
            ] for index in range(count)])
        query = arrays["query_mask"]
        top10 = [row["group_id"] for row in summary["perturbation_top10"]]
        text_range = max(abs(float(arrays["text_scores"].min())), abs(float(arrays["text_scores"].max())))
        if summary["statistics_computed"]:
            deletion_panel = _image_panel(image, "Deletion attention-output error", values=arrays["deletion_scores"], grid=grid, limits=(0, float(arrays["deletion_scores"].max())), alpha=alpha, highlights=top10, subtitle="Raw squared readout error; no final-answer error claim")
            flow_panel = _image_panel(image, "Attention-flow source support", values=arrays["support"], grid=grid, limits=(0, float(arrays["support"].max())), alpha=alpha, subtitle=f"Same perturbation source layer {summary['perturbation_layer']}; full source softmax")
        else:
            unavailable = ["Full budget K=N: selection is an exact no-op.", "D/support were not computed; exported arrays contain NA, not fabricated zeros."]
            deletion_panel = _text_panel(unavailable, title="Deletion score unmeasured (K=N)")
            flow_panel = _text_panel(unavailable, title="Flow support unmeasured (K=N)")
        panels: list[tuple[str, Image.Image]] = [
            ("original", _image_panel(image, "Actual model input view", subtitle=f"view {view_index}; grid={grid}; {count} original groups")),
            ("question_answer", _text_panel([
                f"Sample: {sample_id}", f"Question: {question}", f"Options: {json.dumps(options, ensure_ascii=False)}",
                f"GT: {ground_truth}", f"Generated answer: {answer}",
                f"N={count}, K={summary['K']} ({summary['K_over_N']:.3%} retained)",
                "Heatmaps preserve native original group IDs. Yellow labels: deletion Top-10 (ties retain midranks).",
            ])),
            ("text_cos", _image_panel(image, "Text-cos relevance", values=arrays["text_scores"], grid=grid, limits=(-text_range, text_range), alpha=alpha)),
            ("query_set", _image_panel(image, "Text Top-K Query set P", grid=grid, keep_mask=query, subtitle="White = not a discovery Query; P is not forced into final S")),
            ("deletion_perturbation", deletion_panel),
            ("flow_support", flow_panel),
            ("text_keep", _image_panel(image, "A: Text-cos Top-K", grid=grid, keep_mask=arrays["text_keep_mask"], subtitle="White = removed; exact K group budget")),
        ]
        if "flow_keep_mask" in arrays:
            panels.append(("flow_keep", _image_panel(image, "Shared-source flow local coverage", grid=grid, keep_mask=arrays["flow_keep_mask"], subtitle=f"Diagnostic comparator, attention L{summary['flow_comparison_attention_layer']}; not archived B")))
        else:
            panels.append(("flow_keep", _text_panel(["No flow comparison was computed.", "This panel is deliberately unavailable rather than copied from another method."], title="Flow comparison unavailable")))
        panels.append(("final_keep", _image_panel(image, "Current method: final selected groups", grid=grid, keep_mask=arrays["final_keep_mask"], highlights=top10, subtitle="White = removed; yellow = raw deletion Top-10 IDs")))
        for name, baseline in (("text", arrays["text_keep_mask"]), ("flow", arrays.get("flow_keep_mask"))):
            if baseline is None:
                panels.append((f"{name}_comparison", _text_panel(["Flow comparison not computed."], title="Flow / current comparison unavailable")))
                continue
            final = arrays["final_keep_mask"]
            categories = np.zeros(count, dtype=np.uint8)
            categories[baseline & ~final] = 1
            categories[~baseline & final] = 2
            categories[baseline & final] = 3
            panels.append((f"{name}_comparison", _image_panel(image, f"{name.title()} / current selection", grid=grid, categorical=categories, subtitle="Blue: baseline only; orange: current only; teal: shared")))
        metrics = summary["high_perturbation_low_text"]
        panels.append(("analysis", _text_panel([
            f"D/support Spearman: {summary['spearman_deletion_vs_support']}",
            f"High-D / low-text outside Query P: {metrics['count']}",
            f"Retained by current selector: {metrics['retained_count']}",
            f"Recovered outside text Top-K: {metrics['recovered_outside_text_topk']}",
            "Diagnostic thresholds only: text bottom 60%, D top 10%; ties included. These do not modify selection.",
            "Cross-layer persistence was not measured by this single-layer inference.",
        ], title="Independent diagnostic statistics")))
        for name, panel in panels:
            panel.save(subfig / f"{name}.png", dpi=(300, 300))
        rows = math.ceil(len(panels) / 4)
        overview = Image.new("RGB", (4 * 720, rows * 720), "white")
        for index, (_, panel) in enumerate(panels):
            overview.paste(panel, (index % 4 * 720, index // 4 * 720))
        overview.save(folder / "overview.png", dpi=(300, 300))
        summary["input_image_size"] = list(image.size)
        summary["overview"] = str(folder / "overview.png")
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        report["views"].append(summary)
    report["diagnostics_wall_seconds"] = time.perf_counter() - started
    (sample_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"directory": str(sample_dir), "summary": str(sample_dir / "summary.json"), "view_count": len(views),
            "diagnostics_wall_seconds": report["diagnostics_wall_seconds"]}


__all__ = ["analyze_view", "spearman_correlation", "save_perturbation_diagnostics"]
