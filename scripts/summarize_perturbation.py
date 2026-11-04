#!/usr/bin/env python3
"""Compare saved native-ID perturbation rankings across actually measured layers.

This script loads scalar diagnostic arrays only.  It does not load a checkpoint
or manufacture a persistence result from a single decision-layer run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from compression.perturbation_diagnostics import spearman_correlation


def _identity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _config(sample_folder: Path) -> dict[str, Any]:
    for folder in (sample_folder, *sample_folder.parents):
        candidate = folder / "config.json"
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        if folder == PROJECT_ROOT:
            break
    return {}


def _profile(config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    # Score layer is the intended varying dimension.  Other meaningful online
    # query/selection settings are separated rather than silently pooled.
    keys = ("data", "model", "split", "dtype", "effective_dtypes", "text_source",
            "temperature", "perturbation_lambda", "local_radius", "max_image_pixels",
            "max_image_patches", "patch_selection", "include_global_view",
            "tile_single_images", "patch_size", "patch_stride", "global_view_size",
            "decode_hf_images", "seed")
    profile = {key: config[key] for key in keys if key in config}
    profile["algorithm_id"] = record.get("algorithm_id", config.get("prune_method"))
    profile["K"] = int(record["K"])
    return profile


def _load_snapshots(roots: Sequence[Path], scope: str) -> tuple[list[dict[str, Any]], list[str]]:
    snapshots: list[dict[str, Any]] = []
    warnings: list[str] = []
    paths = sorted({path.resolve() for root in roots if root.is_dir() for path in root.rglob("summary.json")})
    for path in paths:
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(sample, dict) or not isinstance(sample.get("views"), list):
                continue
            config = _config(path.parent)
            extra = sample.get("extra", {})
            # Older diagnostic directory labels used dataset index strings.
            # Only extra.sample_id contains their actual Case.id.
            actual_id = extra.get("sample_id", sample.get("sample_id"))
            if actual_id is None:
                raise ValueError("missing actual sample_id")
            content = {key: sample.get(key) for key in ("question", "options", "ground_truth")}
            if not content["question"]:
                raise ValueError("missing question needed for safe sample identity checks")
            for view_index, declared in enumerate(sample["views"]):
                try:
                    if not declared.get("statistics_computed", True):
                        raise ValueError("D/support were explicitly not measured (full budget)")
                    score_layer = declared.get("score_layer", extra.get("score_layer"))
                    perturbation_layer = declared.get("perturbation_layer", extra.get("perturbation_layer"))
                    if score_layer is None or perturbation_layer is None or int(perturbation_layer) != int(score_layer) + 1:
                        raise ValueError("score_layer / perturbation_layer must explicitly satisfy l+1")
                    folder = path.parent / f"view_{view_index:03d}"
                    if not (folder / "input.png").is_file():
                        raise ValueError("missing exact input image for cross-run identity validation")
                    count = int(declared["N"])
                    grid = tuple(int(x) for x in declared["grid"])
                    if len(grid) != 2 or min(grid) <= 0 or math.prod(grid) != count:
                        raise ValueError("grid/N mismatch")
                    with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
                        arrays = {key: np.asarray(archive[key]).copy() for key in (
                            "group_id", "group_row", "group_col", "deletion_scores", "query_mask"
                        )}
                    if any(value.shape != (count,) for value in arrays.values()):
                        raise ValueError("scalar array / group ID shape mismatch")
                    ids = np.arange(count, dtype=np.int64)
                    if not np.array_equal(arrays["group_id"], ids):
                        raise ValueError("group IDs must be the exact original native IDs in order")
                    if not np.array_equal(arrays["group_row"], ids // grid[1]) or not np.array_equal(arrays["group_col"], ids % grid[1]):
                        raise ValueError("group spatial coordinates do not match original row-major grid")
                    if arrays["query_mask"].dtype != np.bool_:
                        raise ValueError("query_mask must be bool")
                    keep = int(declared["K"])
                    if not 0 < keep <= count or int(arrays["query_mask"].sum()) != keep:
                        raise ValueError("Query mask must contain exactly the baseline Top-K groups")
                    scores = arrays["deletion_scores"].astype(np.float64)
                    if not np.isfinite(scores).all() or np.any(scores < 0):
                        raise ValueError("measured deletion scores must be finite and nonnegative")
                    eligible = ~arrays["query_mask"] if scope == "outside_query" else np.ones(count, dtype=bool)
                    snapshots.append({
                        "actual_sample_id": actual_id, "view_index": view_index,
                        "source": str(folder), "score_layer": int(score_layer),
                        "perturbation_layer": int(perturbation_layer),
                        "N": count, "grid": grid, "scores": scores,
                        "eligible": eligible, "content_identity": _identity(content),
                        "image_sha256": _digest(folder / "input.png"),
                        "profile": _profile(config, declared),
                    })
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    warnings.append(f"Skipped {path}, view {view_index}: {exc}")
        except (TypeError, ValueError, OSError) as exc:
            warnings.append(f"Skipped {path}: {exc}")
    return snapshots, warnings


def _midranks_descending(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1
    return ranks


def build_report(input_roots: Sequence[str | Path], *, top_fraction: float = .1,
                 scope: str = "outside_query") -> dict[str, Any]:
    if not math.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0,1]")
    if scope not in {"all", "outside_query"}:
        raise ValueError("scope must be all or outside_query")
    roots = [Path(root).expanduser() for root in input_roots]
    snapshots, warnings = _load_snapshots(roots, scope)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in snapshots:
        key = _identity({"profile": row["profile"], "sample": row["actual_sample_id"], "view": row["view_index"]})
        buckets.setdefault(key, []).append(row)
    pairs, ranking_rows = [], []
    profiles: dict[str, dict[str, Any]] = {}
    for key, rows in buckets.items():
        first = rows[0]
        profile_id = hashlib.sha256(_identity(first["profile"]).encode()).hexdigest()[:12]
        series_id = hashlib.sha256(key.encode()).hexdigest()[:12]
        profiles[profile_id] = first["profile"]
        # Inconsistent geometry/content for the same real ID is an error, not
        # evidence that a zero-filled or resized heatmap represents the ID.
        identity_fields = ("N", "grid", "content_identity", "image_sha256")
        if any(any(row[field] != first[field] for field in identity_fields) for row in rows[1:]):
            warnings.append(f"Rejected cross-run identity/mapping mismatch: sample={first['actual_sample_id']!r}, view={first['view_index']}")
            continue
        layers: dict[int, dict[str, Any]] = {}
        ambiguous = set()
        for row in rows:
            layer = row["score_layer"]
            if layer in layers and (not np.array_equal(row["scores"], layers[layer]["scores"]) or
                                    not np.array_equal(row["eligible"], layers[layer]["eligible"])):
                ambiguous.add(layer)
            else:
                layers[layer] = row
        for layer in ambiguous:
            layers.pop(layer, None)
            warnings.append(f"Excluded conflicting repeated layer {layer}: sample={first['actual_sample_id']!r}, view={first['view_index']}")
        for layer, row in sorted(layers.items()):
            ids = np.flatnonzero(row["eligible"])
            ranks = _midranks_descending(row["scores"][ids]) if ids.size else np.empty(0)
            for index, group_id in enumerate(ids):
                ranking_rows.append({
                    "sample_id": row["actual_sample_id"], "view_index": row["view_index"],
                    "algorithm_id": row["profile"]["algorithm_id"],
                    "profile_id": profile_id, "series_id": series_id,
                    "score_layer": layer, "perturbation_layer": layer + 1,
                    "group_id": int(group_id), "group_row": int(group_id // row["grid"][1]),
                    "group_col": int(group_id % row["grid"][1]),
                    "deletion_score": float(row["scores"][group_id]),
                    "rank": float(ranks[index]),
                    "rank_q": float((ranks[index] - 1) / max(ids.size - 1, 1)),
                    "eligible_count": int(ids.size),
                })
        layer_ids = sorted(layers)
        for left_index, left_layer in enumerate(layer_ids):
            for right_layer in layer_ids[left_index + 1:]:
                left, right = layers[left_layer], layers[right_layer]
                common = left["eligible"] & right["eligible"]
                candidate_counts = [int(row["eligible"].sum()) for row in (left, right)]
                informative = all(count >= 2 for count in candidate_counts) and all(
                    np.ptp(row["scores"][row["eligible"]]) > max(float(np.max(row["scores"])), np.finfo(np.float32).tiny) * 1e-6
                    for row in (left, right)
                )
                measured = informative and int(common.sum()) >= 2
                top_sets = [set(np.flatnonzero(row["eligible"] & (row["scores"] >= np.quantile(row["scores"][row["eligible"]], 1 - top_fraction))).tolist())
                            for row in (left, right)] if measured else [set(), set()]
                intersection, union = top_sets[0] & top_sets[1], top_sets[0] | top_sets[1]
                pairs.append({
                    "sample_id": first["actual_sample_id"], "view_index": first["view_index"],
                    "algorithm_id": first["profile"]["algorithm_id"],
                    "profile_id": profile_id, "series_id": series_id,
                    "score_layer_left": left_layer, "score_layer_right": right_layer,
                    "perturbation_layer_left": left_layer + 1,
                    "perturbation_layer_right": right_layer + 1,
                    "measured": measured,
                    "reason": None if measured else "insufficient common eligible IDs or uninformative/near-tied scores",
                    "N": first["N"], "grid": list(first["grid"]),
                    "eligible_left": candidate_counts[0], "eligible_right": candidate_counts[1],
                    "common_eligible": int(common.sum()),
                    "spearman": spearman_correlation(left["scores"][common], right["scores"][common]) if measured else None,
                    "top_left_count": len(top_sets[0]) if measured else None,
                    "top_right_count": len(top_sets[1]) if measured else None,
                    "top_intersection": len(intersection) if measured else None,
                    "top_overlap_coefficient": len(intersection) / min(map(len, top_sets)) if measured else None,
                    "top_jaccard": len(intersection) / len(union) if measured else None,
                    "source_left": left["source"], "source_right": right["source"],
                })
    measured_pairs = sum(row["measured"] for row in pairs)
    return {
        "cross_layer_persistence_measured": measured_pairs > 0,
        "measured_pair_count": measured_pairs, "snapshot_count": len(snapshots),
        "profiles": profiles,
        "scope": scope, "top_fraction": top_fraction,
        "rank_definition": "Descending deletion error, tie-averaged ranks; rank_q=0 most sensitive. Noneligible IDs are omitted, never zero-filled.",
        "top_definition": "Per-layer eligible top-score quantile, boundary ties included; overlap=intersection/min(top set sizes).",
        "identity_validation": "Exact actual sample ID (including JSON type), question/options/GT, view, input PNG SHA256, grid/N, native group IDs/coordinates, budget and matching recorded profile.",
        "limitations": "Descriptive fixed-query attention-readout consistency; not statistical significance or final-answer causal importance. Single-layer or uninformative data is not a measured persistence result.",
        "warnings": warnings, "pairs": pairs, "rankings": ranking_rows,
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("layer_pairs", report["pairs"]), ("rankings", report["rankings"])):
        with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps({key: value for key, value in report.items() if key != "rankings"},
                                                ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/perturbation_cross_layer_summary")
    parser.add_argument("--top-fraction", type=float, default=.1)
    parser.add_argument("--scope", choices=("outside_query", "all"), default="outside_query")
    args = parser.parse_args(argv)
    report = build_report(args.input_roots, top_fraction=args.top_fraction, scope=args.scope)
    write_report(report, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key not in {"pairs", "rankings"}}, ensure_ascii=False, indent=2))
    print(f"Saved {Path(args.output_dir).resolve()}")
    return report


if __name__ == "__main__":
    main()
