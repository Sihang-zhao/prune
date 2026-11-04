#!/usr/bin/env python3
"""Choose skip/resume/run for one sweep cell without reading large JSONL files."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluator
import testn
from scripts.summarize_results import _profile


def select_run(args: Any) -> tuple[str, Path | None]:
    args.data = str(Path(args.data).expanduser().resolve())
    data_path = Path(args.data)
    args.image_root = str(Path(args.image_root or (data_path if data_path.is_dir() else data_path.parent)).expanduser().resolve())
    args.model = evaluator._resolve_model(args.model)
    if args.visual_prune_percent is not None:
        args.visual_keep_ratio = 1.0 - float(args.visual_prune_percent) / 100.0
    requested = vars(args)
    profile = _profile(requested, 0)
    completed, partial = [], []
    for config_path in Path(args.output_dir).expanduser().glob("*/config.json"):
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(previous, dict) or _profile(previous, 0) != profile:
                continue
            keys = ("prune_method", "pruning_layer", "visual_keep_tokens", "visual_pruning")
            if any(previous.get(key) != requested.get(key) for key in keys):
                continue
            if abs(float(previous.get("visual_keep_ratio", -1)) - args.visual_keep_ratio) > 1e-9:
                continue
            summary_path = config_path.with_name("summary.json")
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                count = previous.get("case_count")
                if isinstance(count, int) and count > 0 and summary.get("total") == count and summary.get("errors", 0) == 0:
                    completed.append((summary_path.stat().st_mtime_ns, str(summary_path), summary_path))
                    continue
            except (OSError, ValueError, AttributeError):
                pass
            checkpoint = config_path.with_name("results.jsonl")
            if checkpoint.is_file() and checkpoint.stat().st_size:
                # The evaluator must accept exactly this checkpoint; a changed
                # setting must never turn into an accidental cross-run resume.
                evaluator._validate_resume_config(checkpoint, args)
                partial.append((checkpoint.stat().st_mtime_ns, str(checkpoint), checkpoint))
        except (OSError, ValueError, TypeError):
            continue
    if completed:
        return "skip", max(completed)[2]
    if partial:
        return "resume", max(partial)[2]
    return "run", None


def main(argv: list[str] | None = None) -> None:
    action, source = select_run(testn.get_args(argv))
    print(action + ("\t" + str(source.resolve()) if source else ""))


if __name__ == "__main__":
    main()
