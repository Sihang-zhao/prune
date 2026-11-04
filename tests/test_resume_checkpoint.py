"""Interrupt real evaluator loops without loading model weights or a GPU."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluator
import testn
from utils.data import Case
from utils.metrics import load_jsonl, save_jsonl


class _Queue:
    def __init__(self, events):
        self.events = iter(events)

    def get(self, timeout=None):
        event = next(self.events)
        if isinstance(event, BaseException):
            raise event
        return event

    def close(self):
        pass

    def join_thread(self):
        pass


class _Process:
    exitcode = 0

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class _Context:
    def __init__(self, events):
        self.queue = _Queue(events)
        self.worker_indices = []

    def Queue(self):
        return self.queue

    def Process(self, *, target, args):
        self.worker_indices.extend(case["index"] for case in args[1])
        return _Process()


def _row(index):
    return {
        "index": index, "status": "ok", "correct": True, "gt": "A",
        "model_answer": "A", "taskL3": "OC", "testtime": 1.0, "flops": 10**12,
        "selection_mode": "middle_text_early", "selection_strategy": "topk",
        "selection_layer": 15, "original_image_tokens": 4, "retained_tokens": 2,
        "retention_ratio": 0.5, "vision_merge_unit": 4,
        "vision_layer_token_counts": [16] * 15 + [8] * 9,
        "selection_budgets": [[0, 2, 0]], "selected_indices": [[0, 2]],
        "selection_groups": [{"final": [], "middle": [0, 2], "shallow": []}],
    }


class ResumeCheckpointTests(unittest.TestCase):
    def test_equal_timestamps_choose_the_newest_numeric_run_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("1104_1200_100", "previous", "1104_1200_99", "1104_1200")
            stamp = 1_700_000_000_000_000_000
            for name in names:
                folder = root / name
                folder.mkdir()
                path = folder / "results.jsonl"
                save_jsonl(path, [_row(0)])
                os.utime(path, ns=(stamp, stamp))
                os.utime(folder, ns=(stamp, stamp))
            args = SimpleNamespace(resume=True, resume_from=None, output_dir=root)
            self.assertEqual(evaluator._resume_source(args), root / "1104_1200_100" / "results.jsonl")

    def test_repeated_interrupts_keep_all_old_and_new_successes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs"
            source = output / "previous" / "results.jsonl"
            source.parent.mkdir(parents=True)

            def arguments():
                return testn.get_args([
                    "--data", str(root / "dataset.jsonl"), "--model", "fake/model",
                    "--image-root", str(root), "--output-dir", str(output),
                    "--decision-layer-human", "15", "--visual-keep-tokens", "2",
                    "--dtype", "bf16", "--worker-cases", "0", "--resume",
                ])

            initial = _row(0)
            save_jsonl(source, [initial])
            original_bytes = source.read_bytes()
            source.with_name("config.json").write_text(json.dumps(vars(arguments())))
            cases = [Case(id=index, gt="A", question="Q", index=index) for index in range(4)]

            def run(events, *, interrupted=False):
                context = _Context(events)
                args = arguments()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    stack.enter_context(patch("utils.data.load_cases", return_value=cases))
                    stack.enter_context(patch.object(evaluator, "_gpu_ids", return_value=["cpu"]))
                    stack.enter_context(patch.object(evaluator.mp, "get_context", return_value=context))
                    stack.enter_context(patch.dict(sys.modules, {"compression.worker": SimpleNamespace(worker=object())}))
                    if interrupted:
                        with self.assertRaises(KeyboardInterrupt):
                            evaluator.test(args)
                        summary = None
                    else:
                        summary = evaluator.test(args)
                return args, context, summary

            first, context, _ = run([("row", _row(1)), KeyboardInterrupt()], interrupted=True)
            self.assertEqual(context.worker_indices, [1, 2, 3])
            self.assertEqual(load_jsonl(first.output), [_row(0), _row(1)])
            self.assertFalse(Path(first.output).with_name("summary.json").exists())

            second, context, _ = run([("row", _row(2)), KeyboardInterrupt()], interrupted=True)
            self.assertEqual(context.worker_indices, [2, 3])
            self.assertEqual(load_jsonl(second.output), [_row(0), _row(1), _row(2)])

            final, context, summary = run([("row", _row(3)), ("done", None)])
            self.assertEqual(context.worker_indices, [3])
            self.assertEqual(summary["total"], 4)
            self.assertEqual(summary["correct"], 4)
            self.assertEqual(summary["errors"], 0)
            self.assertEqual(load_jsonl(final.output), [_row(index) for index in range(4)])
            self.assertEqual(source.read_bytes(), original_bytes)

            # Resuming a completed checkpoint requires no new worker and still
            # publishes a complete full-metadata checkpoint for future resumes.
            reused, context, summary = run([])
            self.assertEqual(context.worker_indices, [])
            self.assertEqual(summary["total"], 4)
            self.assertEqual(load_jsonl(reused.output), [_row(index) for index in range(4)])

    def test_interrupt_during_seed_does_not_publish_a_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "old" / "results.jsonl"
            source.parent.mkdir()
            save_jsonl(source, [_row(0), _row(1)])
            destination = root / "new" / "results.jsonl"
            encode = json.dumps
            calls = 0

            def interrupted_encode(row, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return encode(row, **kwargs)

            with patch.object(evaluator.json, "dumps", side_effect=interrupted_encode):
                with self.assertRaises(KeyboardInterrupt):
                    evaluator._seed_resume_checkpoint(destination, [_row(0), _row(1)])
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.iterdir()), [])
            args = SimpleNamespace(resume=True, resume_from=None, output_dir=root)
            self.assertEqual(evaluator._resume_source(args), source)


if __name__ == "__main__":
    unittest.main()
