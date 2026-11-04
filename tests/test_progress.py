"""The visible accuracy covers all completed cases, including resumed runs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import runner
from evaluation.data import Case


class _Progress:
    instances = []

    def __init__(self, **kwargs):
        self.settings = kwargs
        self.n = kwargs.get("initial", 0)
        self.snapshots = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, count):
        self.n += count

    def set_postfix(self, ordered_dict=None, **kwargs):
        kwargs.pop("refresh", None)
        self.snapshots.append((self.n, dict(ordered_dict or {}, **kwargs)))


class _Queue:
    def __init__(self, events):
        self.events = iter(events)

    def get(self, timeout=None):
        return next(self.events)

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
    def __init__(self, batches):
        self.batches = iter(batches)
        self.indices = []

    def Queue(self):
        return _Queue(next(self.batches))

    def Process(self, *, target, args):
        self.indices.extend(row["index"] for row in args[1])
        return _Process()


def _row(index, correct, status=None):
    return {
        "index": index,
        "correct": correct,
        "status": status or ("correct" if correct else "incorrect"),
        "original_image_tokens": 1,
        "retained_tokens": 1,
    }


class ProgressAccuracyTests(unittest.TestCase):
    def _run(self, indices, batches, *, previous=(), worker_cases=0):
        _Progress.instances = []
        context = _Context(batches)
        cases = [Case(id=index, index=index, question="Q", gt="A") for index in indices]
        with tempfile.TemporaryDirectory() as temporary:
            cfg = {
                "output": str(Path(temporary) / "results.jsonl"),
                "method": "base",
                "worker_cases": worker_cases,
            }
            with patch.object(runner.mp, "get_context", return_value=context), patch(
                "tqdm.tqdm", _Progress
            ):
                rows = runner._run_workers(
                    cases, cfg, ["cpu"], lambda message: None, initial_rows=previous
                )
        self.assertEqual(len(_Progress.instances), 1)
        return rows, _Progress.instances[0], context

    def test_fresh_accuracy_includes_wrong_answers_and_error_cases(self):
        events = [
            ("row", _row(0, True)),
            ("row", _row(1, False)),
            ("row", _row(2, True, "error")),
            ("done", "cpu"),
        ]
        rows, bar, context = self._run([0, 1, 2], [events])
        self.assertEqual(bar.settings["total"], 3)
        self.assertEqual(bar.settings.get("initial", 0), 0)
        self.assertEqual(context.indices, [0, 1, 2])
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            bar.snapshots[-3:],
            [
                (1, {"acc": "1.0000", "errors": 0}),
                (2, {"acc": "0.5000", "errors": 0}),
                (3, {"acc": "0.3333", "errors": 1}),
            ],
        )

    def test_resumed_prefix_and_multiple_batches_share_one_running_accuracy(self):
        previous = [_row(0, True), _row(1, False)]
        batches = [
            [("row", _row(2, True)), ("row", _row(3, False)), ("done", "cpu")],
            [("row", _row(4, True)), ("row", _row(5, False, "error")), ("done", "cpu")],
        ]
        rows, bar, context = self._run([2, 3, 4, 5], batches, previous=previous, worker_cases=2)
        self.assertEqual(bar.settings["total"], 6)
        self.assertEqual(bar.settings["initial"], 2)
        self.assertEqual(bar.snapshots[0], (2, {"acc": "0.5000", "errors": 0}))
        self.assertEqual(
            bar.snapshots[-4:],
            [
                (3, {"acc": "0.6667", "errors": 0}),
                (4, {"acc": "0.5000", "errors": 0}),
                (5, {"acc": "0.6000", "errors": 0}),
                (6, {"acc": "0.5000", "errors": 1}),
            ],
        )
        self.assertEqual(context.indices, [2, 3, 4, 5])
        self.assertEqual(len(rows), 4)

    def test_fully_resumed_run_displays_final_accuracy_without_any_worker(self):
        previous = [_row(0, True), _row(1, False)]
        rows, bar, context = self._run([], [], previous=previous)
        self.assertEqual(rows, [])
        self.assertEqual(context.indices, [])
        self.assertEqual((bar.settings["total"], bar.settings["initial"], bar.n), (2, 2, 2))
        self.assertEqual(bar.snapshots[-1], (2, {"acc": "0.5000", "errors": 0}))


if __name__ == "__main__":
    unittest.main()
