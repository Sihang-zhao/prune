"""Worker orchestration, resumable checkpoints and XLRS benchmark reports."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import time
from datetime import datetime

from .metrics import evaluate, load_jsonl, save_markdown


SCHEMA = "qwen3_perturbation_v1"
RESUME_KEYS = (
    "schema",
    "data",
    "model",
    "image_root",
    "method",
    "implementation",
    "layer",
    "prune_percent",
    "keep_tokens",
    "dtype",
    "split",
    "limit",
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
    "decode_hf_images",
    "temperature",
    "perturbation_lambda",
    "local_radius",
    "query_chunk_size",
    "group_chunk_size",
    "source_chunk_size",
    "max_attention_mb",
    "greedy_backend",
    "diagnostics",
    "seed",
    "verify_candidates",
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_resume_config(source, args):
    cfg = vars(args) if not isinstance(args, dict) else args
    path = Path(source).with_name("config.json")
    previous = json.loads(path.read_text(encoding="utf-8"))
    mismatches = [key for key in RESUME_KEYS if previous.get(key) != cfg.get(key)]
    if mismatches:
        raise ValueError("Resume configuration differs: " + ", ".join(mismatches))


def _select_resume_cases(cases, source):
    if source is None:
        return cases, []
    previous = load_jsonl(source)
    current = {int(case.index) for case in cases}
    successful = {
        int(row["index"]): row
        for row in previous
        if row.get("status") != "error" and int(row["index"]) in current
    }
    return (
        [case for case in cases if int(case.index) not in successful],
        [successful[index] for index in sorted(successful)],
    )


def _seed_checkpoint(output, rows):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint_", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_row(row, cfg):
    if row.get("status") == "error":
        return False
    if cfg["method"] == "base":
        if row["retained_tokens"] != row["original_image_tokens"]:
            raise ValueError("base did not retain all visual groups")
        return True
    counts = row["per_view_group_counts"]
    selected = row["selected_indices"]
    if len(selected) != len(counts):
        raise ValueError("selection/view count mismatch")
    for ids, count in zip(selected, counts):
        expected = (
            cfg["keep_tokens"]
            if cfg["keep_tokens"] is not None
            else max(1, round(count * (1 - cfg["prune_percent"] / 100)))
        )
        if len(ids) != expected or ids != sorted(set(ids)) or min(ids) < 0 or max(ids) >= count:
            raise ValueError(
                "selection violated the exact per-view group budget or original ID order"
            )
    trace = row["vision_layer_token_counts"]
    unit = row["vision_merge_unit"]
    full = row["original_image_tokens"] * unit
    compact = row["retained_tokens"] * unit
    boundary = cfg["layer"] + 1
    if len(trace) != cfg["vision_depth"] or trace != [full] * boundary + [compact] * (
        len(trace) - boundary
    ):
        raise ValueError("visual blocks did not use the expected full/compact token lengths")
    return True


def _devices(cfg):
    import torch

    if not torch.cuda.is_available():
        if cfg["gpus"] is not None:
            raise RuntimeError("CUDA GPUs were requested but CUDA is unavailable")
        return ["cpu"]
    ids = str(cfg["gpus"] or "0").split(",")
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not i.strip().isdigit() or int(i) >= torch.cuda.device_count() for i in ids)
    ):
        raise ValueError("invalid --gpus list")
    return [f"cuda:{int(i)}" for i in ids]


def _run_workers(cases, cfg, devices, log):
    if not cases:
        return []
    from .data import case_dict
    from .worker import worker
    from tqdm import tqdm

    context = mp.get_context("spawn")
    size = cfg["worker_cases"] or len(cases)
    compact_rows = []
    with Path(cfg["output"]).open("a", encoding="utf-8") as handle, tqdm(
        total=len(cases), desc="Testing", unit="case"
    ) as progress:
        for start in range(0, len(cases), size):
            batch = cases[start : start + size]
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=worker,
                    args=(
                        device,
                        [case_dict(c) for c in batch[i :: len(devices)]],
                        cfg,
                        result_queue,
                    ),
                )
                for i, device in enumerate(devices)
                if batch[i :: len(devices)]
            ]
            completed = 0
            indices = set()
            failures = []
            for process in processes:
                process.start()
            try:
                while completed < len(processes):
                    try:
                        kind, value = result_queue.get(timeout=1)
                    except queue.Empty:
                        if all(not process.is_alive() for process in processes):
                            try:
                                kind, value = result_queue.get(timeout=2)
                            except queue.Empty:
                                break
                        else:
                            continue
                    if kind == "row":
                        try:
                            _validate_row(value, cfg)
                        except Exception as error:
                            value.update(status="error", correct=False, error=str(error))
                        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                        handle.flush()
                        indices.add(int(value["index"]))
                        compact = dict(value)
                        compact.pop("pruning_diagnostics", None)
                        compact.pop("selected_indices", None)
                        compact_rows.append(compact)
                        progress.update(1)
                        progress.set_postfix(
                            errors=sum(row.get("status") == "error" for row in compact_rows)
                        )
                        if value.get("status") == "error":
                            log(f"case {value['index']} failed: {value.get('error')}")
                    elif kind == "worker_error":
                        failures.append(value)
                    elif kind == "done":
                        completed += 1
            finally:
                for process in processes:
                    if completed < len(processes) and process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                result_queue.close()
                result_queue.join_thread()
            exits = [p.exitcode for p in processes if p.exitcode != 0]
            if failures or exits or indices != {int(case.index) for case in batch}:
                raise RuntimeError(
                    f"worker batch incomplete; exit codes={exits}; details={failures}. Saved results can be resumed."
                )
    return compact_rows


def _resume_source(cfg):
    if cfg.get("resume_from"):
        path = Path(cfg["resume_from"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if not cfg["resume"]:
        return None
    paths = list(Path(cfg["output_dir"]).glob("*/results.jsonl"))
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, p.parent.name)) if paths else None


def run(args):
    from .data import load_cases

    cfg = dict(vars(args))
    cfg["schema"] = SCHEMA
    cfg["data"] = str(Path(cfg["data"]).expanduser().resolve())
    model = Path(cfg["model"]).expanduser()
    if model.is_dir():
        cfg["model"], cfg["offline"] = str(model.resolve()), True
    data = Path(cfg["data"])
    cfg["image_root"] = str(
        Path(cfg["image_root"] or (data if data.is_dir() else data.parent)).expanduser().resolve()
    )
    cfg["output_dir"] = str(Path(cfg["output_dir"]).expanduser().resolve())
    model_config = model / "config.json"
    cfg["vision_depth"] = (
        int(json.loads(model_config.read_text())["vision_config"]["depth"])
        if model_config.is_file()
        else 27
    )
    source = _resume_source(cfg)
    if source:
        _validate_resume_config(source, cfg)
    cases = load_cases(cfg["data"], cfg["limit"], cfg["split"])
    if not cases:
        raise ValueError("no evaluation cases were found")
    pending, reused = _select_resume_cases(cases, source)
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = root / stamp
    counter = 0
    while run_dir.exists():
        counter += 1
        run_dir = root / f"{stamp}_{counter:03d}"
    run_dir.mkdir()
    cfg.update(output=str(run_dir / "results.jsonl"), case_count=len(cases))
    devices = _devices(cfg)
    _write_json(run_dir / "config.json", {**cfg, "devices": devices})
    started = time.perf_counter()
    with (run_dir / "test.log").open("w", encoding="utf-8") as log_file:

        def log(message):
            print(message, flush=True)
            log_file.write(datetime.now().isoformat(timespec="seconds") + " " + message + "\n")
            log_file.flush()

        log(
            f"method={cfg['method']} implementation={cfg['implementation']}; cases={len(cases)} devices={devices}"
        )
        log(f"reused={len(reused)} pending={len(pending)}")
        _seed_checkpoint(cfg["output"], reused)
        generated = _run_workers(pending, cfg, devices, log)
        rows = sorted(
            {int(row["index"]): row for row in reused + generated}.values(),
            key=lambda row: int(row["index"]),
        )
        summary = evaluate(rows)
        if summary["total"] != len(cases):
            raise RuntimeError("case count mismatch; no complete summary will be published")
        valid = [row for row in rows if row.get("status") != "error"]
        summary["avg_retention_ratio"] = (
            sum(row["retention_ratio"] for row in valid) / len(valid) if valid else None
        )
        summary["total_wall_seconds"] = time.perf_counter() - started
        if valid and all("flops_breakdown" in row for row in valid):
            summary["avg_flops_breakdown"] = {
                key: sum(row["flops_breakdown"][key] for row in valid) / len(valid)
                for key in valid[0]["flops_breakdown"]
            }
        _write_json(run_dir / "summary.json", summary)
        display = dict(
            cfg,
            visual_pruning=cfg["method"] != "base",
            visual_keep_ratio=1 - cfg["prune_percent"] / 100,
        )
        save_markdown(run_dir / "results.md", summary, config=display)
        report = (run_dir / "results.md").read_text()
        budget = (
            f"K={cfg['keep_tokens']}"
            if cfg["keep_tokens"] is not None
            else f"removed={cfg['prune_percent']:g}%"
        )
        label = (
            "Qwen3-VL base"
            if cfg["method"] == "base"
            else f"{cfg['method']}; L={cfg['layer']}; {budget}"
        )
        start = report.index("<tbody>\n<tr><td>") + len("<tbody>\n<tr><td>")
        end = report.index("</td>", start)
        (run_dir / "results.md").write_text(report[:start] + label + report[end:])
        log(
            f"macro accuracy={summary['macro_acc']:.4f}; time={summary['avg_testtime']:.4f}s; TFLOPs={summary['avg_tflops']:.6f}; errors={summary['errors']}"
        )
        log(f"saved: {run_dir}")
        return summary
