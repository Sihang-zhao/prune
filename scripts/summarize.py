#!/usr/bin/env python3
"""Collect completed pruning and base runs into a comparable benchmark table."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.tasks import ALL_TASKS, PERCEPTION_TASKS, REASONING_TASKS


def summarize(input_root, output_prefix):
    candidates, profiles = {}, set()
    profile_keys = (
        "data",
        "model",
        "image_root",
        "dtype",
        "split",
        "limit",
        "case_count",
        "schema",
        "max_image_pixels",
        "max_image_patches",
        "max_new_tokens",
        "enable_thinking",
        "include_global_view",
        "tile_single_images",
        "patch_selection",
        "patch_size",
        "patch_stride",
        "global_view_size",
        "temperature",
        "perturbation_lambda",
        "local_radius",
        "query_chunk_size",
        "group_chunk_size",
        "source_chunk_size",
        "max_attention_mb",
        "greedy_backend",
        "cancellation_recovery",
        "seed",
    )
    for path in Path(input_root).expanduser().rglob("config.json"):
        summary_path = path.with_name("summary.json")
        if not summary_path.is_file():
            continue
        config = json.loads(path.read_text())
        report = json.loads(summary_path.read_text())
        if config.get("schema") != "qwen3_perturbation_v1":
            continue
        if not report.get("total") or report["total"] != config.get("case_count"):
            continue
        profiles.add(
            json.dumps(
                {
                    key: config.get(key, "deferred" if key == "cancellation_recovery" else None)
                    for key in profile_keys
                },
                sort_keys=True,
            )
        )
        key = (
            config["method"],
            config["implementation"],
            0 if config["method"] == "base" else config["layer"],
            config["prune_percent"],
            config["keep_tokens"],
        )
        candidate = (summary_path.stat().st_mtime_ns, str(summary_path), config, report)
        if key not in candidates or candidate[:2] > candidates[key][:2]:
            candidates[key] = candidate
    if len(profiles) > 1:
        raise ValueError(
            "Input mixes different datasets, models or evaluation settings; use separate output roots."
        )
    columns = [
        "Method",
        "Layer",
        "Prune%",
        "K",
        *ALL_TASKS,
        "Avg.",
        "time (s)",
        "TFLOPs",
        "Cases",
        "Errors",
    ]
    rows = []
    for key in sorted(candidates, key=lambda k: (k[0], k[1], k[2], k[3], k[4] or 0)):
        _, _, cfg, report = candidates[key]
        tasks = report.get("taskL3", {})
        values = [
            cfg["method"] + "/" + cfg["implementation"],
            "—" if cfg["method"] == "base" else cfg["layer"],
            "—" if cfg["keep_tokens"] is not None else f"{cfg['prune_percent']:g}",
            cfg["keep_tokens"] or "—",
        ]
        values.extend(
            f"{tasks[t]['acc'] * 100:.2f}" if tasks.get(t, {}).get("total") else "—"
            for t in ALL_TASKS
        )
        values.extend(
            [
                (
                    f"{report['macro_acc'] * 100:.2f}"
                    if report.get("macro_task_count") == len(ALL_TASKS)
                    else "—"
                ),
                f"{report['avg_testtime']:.4f}",
                f"{report['avg_tflops']:.6f}",
                report["total"],
                report["errors"],
            ]
        )
        rows.append(values)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if (prefix.parent / "config.json").exists() and prefix.name in ("summary", "config", "results"):
        raise ValueError("Choose the sweep output root, not an individual run directory.")
    with Path(str(prefix) + ".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    head = "<table>\n<thead><tr>" + "".join(f'<th rowspan="2">{name}</th>' for name in columns[:4])
    head += f'<th colspan="{len(PERCEPTION_TASKS)}">Perception</th><th colspan="{len(REASONING_TASKS)}">Reasoning</th>'
    head += "".join(
        f'<th rowspan="2">{html.escape(name)}</th>' for name in columns[4 + len(ALL_TASKS) :]
    )
    head += "</tr><tr>" + "".join(f"<th>{t}</th>" for t in ALL_TASKS) + "</tr></thead>\n<tbody>\n"
    table = head + "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    table += "\n</tbody></table>\n"
    Path(str(prefix) + ".md").write_text("# Evaluation summary\n\n" + table, encoding="utf-8")
    Path(str(prefix) + ".html").write_text(
        '<!doctype html><meta charset="utf-8">' + table, encoding="utf-8"
    )
    print(f"Summarized {len(rows)} completed configurations: {prefix}.csv / .md / .html")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    summarize(args.input_root, args.output_prefix)
