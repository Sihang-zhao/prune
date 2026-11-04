#!/usr/bin/env python3
"""Summarize completed layer/pruning experiments without loading result JSONL."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.tasks import ALL_TASKS, PERCEPTION_TASKS, REASONING_TASKS

DEFAULT_LAYERS = (15, 16, 17, 18)
DEFAULT_PRUNE_PERCENTS = (50, 60, 70, 80, 85, 90, 95, 98)
DEFAULT_METHOD = "q_flowcover_direct_residual"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    return int(result) if result is not None and result.is_integer() else None


def _accuracy(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and 0.0 <= result <= 1.0 else None


def _path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve()) if value is not None else ""


def _model(value: Any) -> str:
    if value is None:
        return ""
    name = str(value).rstrip("/")
    candidate = Path(name).expanduser()
    return _path(name) if name.startswith(("/", ".", "~")) or candidate.exists() else name


def _dtype_name(value: Any) -> str:
    name = str(value or "auto").lower()
    return {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}.get(name, name)


def _effective_dtypes(config: dict[str, Any]) -> tuple[str, ...]:
    values = config.get("effective_dtypes")
    if isinstance(values, (list, tuple)) and values:
        return tuple(sorted({_dtype_name(value) for value in values}))
    return (_dtype_name(config.get("dtype")),)


def _profile(config: dict[str, Any], cases: int) -> dict[str, Any]:
    """Comparison settings excluding the two intended sweep dimensions."""
    defaults = {
        "split": "train", "limit": None, "visual_keep_tokens": None,
        "max_image_pixels": 4_194_304, "max_image_patches": 1,
        "patch_selection": "uniform", "include_global_view": True,
        "tile_single_images": True, "patch_size": 448, "patch_stride": 336,
        "global_view_size": 448, "decode_hf_images": False,
        "enable_thinking": False, "max_new_tokens": 32,
        "pruning_mode": "middle_text_early", "pruning_strategy": "topk",
        "q_flowcover_local_radius": 1, "q_flowcover_affinity_mode": "raw",
        "q_flowcover_query_chunk": 256, "q_flowcover_query_mass": 0.9,
        "q_flowcover_adaptive_radius_max": 3, "q_flowcover_candidate_chunk": 64,
        "q_flowcover_radius": 1, "text_source": "baseline",
        "att_stats": "none", "max_debug_samples": 0,
    }
    result = {key: config.get(key, default) for key, default in defaults.items()}
    result.update(
        data=_path(config.get("data")), model=_model(config.get("model")),
        image_root=_path(config.get("image_root")), case_count=cases,
        dtype=list(_effective_dtypes(config)),
        exec_mode=config.get("compression_mode", config.get("exec_mode", "off")),
    )
    # The middle role deliberately changes with the chosen pruning layer.
    vision_layers = config.get("vision_layers")
    if vision_layers:
        try:
            values = [int(x.strip()) for x in str(vision_layers).split(",")]
            result["vision_outer_layers"] = [values[0], values[-1]]
        except (TypeError, ValueError, IndexError):
            result["vision_outer_layers"] = str(vision_layers)
    else:
        result["vision_outer_layers"] = [6, 24]
    return result


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _task_values(summary: dict[str, Any]) -> dict[str, float | None]:
    stats = summary.get("taskL3", summary.get("categories", {}))
    stats = stats if isinstance(stats, dict) else {}
    result: dict[str, float | None] = {}
    for task in ALL_TASKS:
        entry = stats.get(task)
        value = None
        if isinstance(entry, dict) and (_integer(entry.get("total")) or 0) > 0:
            value = _accuracy(entry.get("acc"))
            if value is None:
                hits = _integer(entry.get("correct"))
                total = _integer(entry.get("total"))
                if hits is not None and total and 0 <= hits <= total:
                    value = hits / total
        result[task] = None if value is None else value * 100.0
    return result


def _macro(summary: dict[str, Any], tasks: dict[str, float | None]) -> float | None:
    count = _integer(summary.get("macro_task_count"))
    all_tasks = all(value is not None for value in tasks.values())
    if (count is not None and count != len(ALL_TASKS)) or (count is None and not all_tasks):
        return None
    value = _accuracy(summary.get("macro_acc"))
    if value is not None:
        return value * 100.0
    # Never use summary['acc'] or micro_acc for the benchmark Avg. column.
    return sum(tasks.values()) / len(ALL_TASKS) if all_tasks else None


def build_report(
    input_roots: Sequence[str | Path] | str | Path,
    layers: Sequence[int] = DEFAULT_LAYERS,
    prune_percents: Sequence[float] = DEFAULT_PRUNE_PERCENTS,
    method: str = DEFAULT_METHOD,
    data: str | Path | None = None,
    model: str | Path | None = None,
    exec_mode: str | None = "hard18",
    dtype: str | None = None,
) -> dict[str, Any]:
    """Return a complete grid; metric values are percentages, seconds, TFLOPs."""
    roots = [input_roots] if isinstance(input_roots, (str, Path)) else list(input_roots)
    roots = [Path(root).expanduser().resolve() for root in roots]
    layer_values = sorted(set(int(value) for value in layers))
    percent_values = sorted(set(round(float(value), 8) for value in prune_percents))
    if not layer_values or any(value <= 0 for value in layer_values):
        raise ValueError("layers must contain positive layer numbers")
    if not percent_values or any(not math.isfinite(value) or not 0 <= value < 100 for value in percent_values):
        raise ValueError("prune_percents must be in [0,100)")
    warnings: list[str] = []
    candidates: dict[tuple[int, float], list[tuple[int, str, dict[str, Any]]]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    config_paths = sorted({p.resolve() for root in roots if root.is_dir() for p in root.rglob("config.json")})
    for config_path in config_paths:
        summary_path = config_path.with_name("summary.json")
        try:
            config = _read_object(config_path)
        except (OSError, ValueError) as error:
            warnings.append(f"Skipped {config_path}: {error}")
            continue
        if str(config.get("prune_method", "text_cos")) != method:
            continue
        layer = _integer(config.get("decision_layer_human", config.get("pruning_layer")))
        declared_layer = _integer(config.get("pruning_layer"))
        if layer is not None and declared_layer is not None and layer != declared_layer:
            warnings.append(f"Skipped {config_path}: inconsistent pruning layer fields")
            continue
        if config.get("visual_keep_tokens") is not None:
            warnings.append(f"Skipped {config_path}: fixed-K budget is not a pruning-percentage experiment")
            continue
        percent = _number(config.get("visual_prune_percent"))
        if percent is None:
            ratio = _number(config.get("visual_keep_ratio"))
            percent = (1.0 - ratio) * 100 if ratio is not None else None
        percent = round(percent, 8) if percent is not None else None
        mode = config.get("compression_mode", config.get("exec_mode", "off"))
        if layer not in layer_values or percent not in percent_values:
            continue
        if exec_mode is not None and mode != exec_mode:
            continue
        if data is not None and _path(config.get("data")) != _path(data):
            continue
        if model is not None and _model(config.get("model")) != _model(model):
            continue
        effective = _effective_dtypes(config)
        if dtype is not None and effective != (_dtype_name(dtype),):
            continue
        if not summary_path.is_file():
            warnings.append(f"Pending run without summary: {config_path.parent}")
            continue
        try:
            summary = _read_object(summary_path)
            cases = _integer(summary.get("total"))
            expected = _integer(config.get("case_count"))
            errors = _integer(summary.get("errors", 0))
            if cases is None or cases <= 0:
                raise ValueError("summary total is missing or not positive")
            if "case_count" in config and (expected is None or cases != expected):
                raise ValueError(f"incomplete case count: summary={cases}, expected={config.get('case_count')}")
            if errors is None or errors < 0 or errors > cases:
                raise ValueError("invalid summary error count")
            tasks = _task_values(summary)
            task_count = sum(value is not None for value in tasks.values())
            notes: list[str] = []
            if task_count != len(ALL_TASKS):
                notes.append(f"{task_count}/{len(ALL_TASKS)} tasks")
                warnings.append(f"Incomplete task coverage in {summary_path}: {task_count}/{len(ALL_TASKS)}")
            macro_count = _integer(summary.get("macro_task_count"))
            if macro_count is not None and macro_count != len(ALL_TASKS):
                notes.append(f"macro covers {macro_count}/{len(ALL_TASKS)} tasks")
            profile = _profile(config, cases)
            profile_key = json.dumps(profile, sort_keys=True, ensure_ascii=False)
            profiles[profile_key] = profile
            row = {
                "method": method, "layer": layer, "prune_percent": percent,
                "status": "completed_with_errors" if errors else "completed",
                "task_accuracies": tasks,
                "metrics": {"avg": _macro(summary, tasks), "time": _number(summary.get("avg_testtime")),
                            "tflops": _number(summary.get("avg_tflops"))},
                "cases": cases, "errors": errors, "task_count": task_count,
                "source": str(summary_path), "config_source": str(config_path),
                "dataset": _path(config.get("data")), "model": _model(config.get("model")),
                "dtype": ",".join(effective), "notes": notes,
            }
            candidates.setdefault((layer, percent), []).append((summary_path.stat().st_mtime_ns, str(summary_path), row))
        except (OSError, ValueError) as error:
            warnings.append(f"Skipped {summary_path}: {error}")
    if len(profiles) > 1:
        values = list(profiles.values())
        differing = sorted(key for key in values[0] if any(value.get(key) != values[0].get(key) for value in values[1:]))
        raise ValueError(
            "Found different experiment profiles (" + ", ".join(differing) + "). "
            "Use --data/--model/--dtype filters or a narrower --input-root; do not combine unlike experiments."
        )
    rows: list[dict[str, Any]] = []
    for layer in layer_values:
        for percent in percent_values:
            runs = candidates.get((layer, percent), [])
            if runs:
                rows.append(max(runs, key=lambda item: (item[0], item[1]))[2])
            else:
                rows.append({
                    "method": method, "layer": layer, "prune_percent": percent, "status": "pending",
                    "task_accuracies": dict.fromkeys(ALL_TASKS),
                    "metrics": {"avg": None, "time": None, "tflops": None},
                    "cases": None, "errors": None, "task_count": 0, "source": None,
                    "config_source": None, "dataset": _path(data) if data is not None else None,
                    "model": _model(model) if model is not None else None,
                    "dtype": _dtype_name(dtype) if dtype is not None else None, "notes": [],
                })
    return {
        "method": method, "exec_mode": exec_mode, "layers": layer_values, "prune_percents": percent_values,
        "input_roots": [str(root) for root in roots], "profile": next(iter(profiles.values()), None),
        "rows": rows, "completed_count": sum(row["status"] != "pending" for row in rows),
        "expected_count": len(rows), "warnings": warnings,
        "selection_rule": "Latest completed matching run per layer/ratio, by summary file modification time; no averaging across runs.",
        "units": {"task_accuracies": "percent", "avg": "13-task macro accuracy, percent", "time": "seconds per case", "tflops": "estimated TFLOPs per case"},
    }


def _format(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _status(row: dict[str, Any]) -> str:
    result = "Pending" if row["status"] == "pending" else (
        f"Completed ({row['errors']} errors)" if row.get("errors") else "Completed"
    )
    return result + ("; " + "; ".join(row["notes"]) if row.get("notes") else "")


def _table(report: dict[str, Any]) -> str:
    output = [
        '<table>\n<thead>\n<tr><th rowspan="2">Method</th><th rowspan="2">Layer</th><th rowspan="2">Prune%</th>'
        f'<th colspan="{len(PERCEPTION_TASKS)}">Perception</th><th colspan="{len(REASONING_TASKS)}">Reasoning</th>'
        '<th rowspan="2">Avg.</th><th rowspan="2">time (s)</th><th rowspan="2">TFLOPs</th><th rowspan="2">Status</th></tr>',
        "<tr>" + "".join(f"<th>{task}</th>" for task in ALL_TASKS) + "</tr>\n</thead>\n<tbody>",
    ]
    for row in report["rows"]:
        values = [row["method"], str(row["layer"]), f"{row['prune_percent']:g}%"]
        values += [_format(row["task_accuracies"].get(task)) for task in ALL_TASKS]
        values += [_format(row["metrics"]["avg"]), _format(row["metrics"]["time"], 4),
                   _format(row["metrics"]["tflops"], 4), _status(row)]
        css_class = "pending" if row["status"] == "pending" else ("errors" if row.get("errors") else "complete")
        output.append(f'<tr class="{css_class}">' + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    output.append("</tbody>\n</table>")
    return "\n".join(output)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _preserve_report(report: dict[str, Any], previous: dict[str, Any]) -> None:
    """Keep completed cells when current roots only contain part of a sweep."""
    if not isinstance(previous.get("rows"), list):
        raise ValueError("Existing summary JSON has no valid rows; it was left unchanged")
    completed = [row for row in previous["rows"] if row.get("status") in {"completed", "completed_with_errors"}]
    if not completed:
        return
    mismatches = [key for key in ("method", "exec_mode") if report.get(key) != previous.get(key)]
    old_profile, new_profile = previous.get("profile"), report.get("profile")
    if not isinstance(old_profile, dict):
        raise ValueError("Cannot verify the experiment profile of the existing completed summary; it was left unchanged")
    if isinstance(new_profile, dict):
        mismatches.extend(key for key in old_profile.keys() | new_profile.keys()
                          if old_profile.get(key) != new_profile.get(key))
    else:
        # No newly completed runs: explicit filters are recorded on pending
        # rows, and must still agree with the already completed experiment.
        for row in report["rows"]:
            for field, profile_key in (("dataset", "data"), ("model", "model")):
                if row.get(field) is not None and row[field] != old_profile.get(profile_key):
                    mismatches.append(profile_key)
            if row.get("dtype") is not None and row["dtype"].split(",") != old_profile.get("dtype"):
                mismatches.append("dtype")
    if mismatches:
        raise ValueError("Existing summary has a different experiment profile (" + ", ".join(sorted(set(mismatches)))
                         + "); previous tables were left unchanged. Choose a separate output prefix.")
    rows = {(row["layer"], row["prune_percent"]): row for row in report["rows"]}
    kept = 0
    for row in completed:
        key = (row["layer"], row["prune_percent"])
        if key not in rows or rows[key].get("status") == "pending":
            rows[key] = copy.deepcopy(row)
            kept += 1
    report["rows"] = [rows[key] for key in sorted(rows)]
    report["layers"] = sorted({row["layer"] for row in report["rows"]})
    report["prune_percents"] = sorted({row["prune_percent"] for row in report["rows"]})
    report["expected_count"] = len(rows)
    report["completed_count"] = sum(row["status"] != "pending" for row in report["rows"])
    if report.get("profile") is None:
        report["profile"] = copy.deepcopy(old_profile)
    # Retain provenance for old cells when a wrapper only passes its newly
    # active root; the snapshot remains usable after source directories move.
    report["input_roots"] = sorted(set(report.get("input_roots", [])) | set(previous.get("input_roots", [])))
    if kept:
        report["warnings"].append(f"Preserved {kept} completed cells from the existing summary; no pending run replaced them.")


def _backup_tables(paths: dict[str, Path], contents: dict[str, str]) -> None:
    """Snapshot a changed table set once per content hash before replacing it."""
    existing = {suffix: path.read_bytes() for suffix, path in paths.items() if path.is_file()}
    if not existing or all(existing.get(suffix) == content.encode("utf-8") for suffix, content in contents.items()):
        return
    digest = hashlib.sha256()
    for suffix, value in sorted(existing.items()):
        digest.update(suffix.encode("ascii") + b"\0" + value)
    prefix = paths["json"].with_suffix("")
    backup = prefix.parent / "summary_history" / (prefix.name + "_" + digest.hexdigest()[:16])
    backup.mkdir(parents=True, exist_ok=True)
    for suffix, value in existing.items():
        target = backup / paths[suffix].name
        if not target.exists():
            target.write_bytes(value)


def write_report(report: dict[str, Any], output_prefix: str | Path, *, preserve_existing: bool = False) -> dict[str, str]:
    """Write CSV, two-level-header Markdown/HTML, and full provenance JSON."""
    prefix = Path(output_prefix).expanduser()
    paths = {suffix: Path(str(prefix) + "." + suffix) for suffix in ("csv", "md", "html", "json")}
    if (prefix.parent / "config.json").is_file() and (
        paths["json"].name in {"config.json", "summary.json"} or paths["md"].name == "results.md"
    ):
        raise ValueError("Output would overwrite an original run file; choose the sweep root or a different output prefix")
    if preserve_existing and paths["json"].is_file():
        _preserve_report(report, _read_object(paths["json"]))
    csv_text = io.StringIO(newline="")
    writer = csv.writer(csv_text, lineterminator="\n")
    writer.writerow(["Method", "Layer", "Prune%", *ALL_TASKS, "Avg.", "time (s)", "TFLOPs", "Status", "Cases", "Errors", "Source"])
    for row in report["rows"]:
        writer.writerow([
            row["method"], row["layer"], f"{row['prune_percent']:g}",
            *[_format(row["task_accuracies"].get(task)) for task in ALL_TASKS],
            _format(row["metrics"]["avg"]), _format(row["metrics"]["time"], 4),
            _format(row["metrics"]["tflops"], 4), _status(row),
            "—" if row["cases"] is None else row["cases"],
            "—" if row["errors"] is None else row["errors"], row.get("source") or "",
        ])
    heading = "XLRS layer / pruning summary"
    progress = f"Completed: {report['completed_count']} / {report['expected_count']} experiments."
    note = ("Accuracies and Avg. are percentages. Avg. is the macro average of all 13 tasks, never micro accuracy. "
            "time is seconds per case; TFLOPs is estimated operations per case. "
            "— means unavailable or not completed, not zero. The latest completed matching run is used for each cell. "
            "Per-row source paths, cases, errors and configuration profile are recorded in the CSV/JSON files.")
    table = _table(report)
    warning_md = ""
    warning_html = ""
    if report.get("warnings"):
        warning_md = "\n\nWarnings:\n\n" + "\n".join("- " + str(warning) for warning in report["warnings"])
        warning_html = "<details><summary>Warnings</summary><ul>" + "".join(
            "<li>" + html.escape(str(warning)) + "</li>" for warning in report["warnings"]
        ) + "</ul></details>"
    markdown = f"# {heading}\n\n{progress}\n\n{table}\n\n{note}{warning_md}\n"
    html_text = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{heading}</title><style>"
        "body{font:14px system-ui,sans-serif;color:#17324d;margin:24px;background:#fff}"
        "h1{font-size:24px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;width:100%}"
        "th,td{border:1px solid #dce4ee;padding:8px;text-align:center}thead{background:#edf3fb}"
        "td:first-child{text-align:left}.pending{color:#8994a3}.errors{background:#fff2e8}"
        "tbody tr:nth-child(even){background-color:#f8fafc}.note{line-height:1.6;color:#52657a}"
        "</style></head><body>"
        f"<h1>{heading}</h1><p>{progress}</p><div class=\"scroll\">{table}</div>"
        f'<p class="note">{html.escape(note)}</p>{warning_html}</body></html>\n'
    )
    contents = {"csv": "\ufeff" + csv_text.getvalue(), "md": markdown, "html": html_text,
                "json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    if preserve_existing:
        _backup_tables(paths, contents)
    for suffix, content in contents.items():
        _atomic_write(paths[suffix], content)
    return {suffix: str(path) for suffix, path in paths.items()}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", nargs="+", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--preserve-existing", action="store_true",
                        help="Preserve completed cells and snapshot changed table files before updating")
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--prune-percents", nargs="+", type=float, default=DEFAULT_PRUNE_PERCENTS)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--exec-mode", default="hard18")
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.input_root, args.layers, args.prune_percents, args.method,
                              args.data, args.model, args.exec_mode, args.dtype)
        paths = write_report(report, args.output_prefix, preserve_existing=args.preserve_existing)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Summary: {report['completed_count']}/{report['expected_count']} completed")
    for suffix, path in paths.items():
        print(f"{suffix}: {path}")
    for warning in report["warnings"][:10]:
        print(f"Warning: {warning}", file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
