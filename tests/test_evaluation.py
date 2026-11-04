"""Public evaluation contracts: budgets, scoring, and safe checkpoint reuse."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def entry():
    spec = importlib.util.spec_from_file_location(
        "_evaluation_entry_under_test", ROOT / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationContracts(unittest.TestCase):
    def args(self, *extra):
        return entry().get_args(["--data", "sample.jsonl", "--model", "fake/model", *extra])

    def test_help_needs_no_model_or_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "evaluate.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        for flag in ("--method", "--implementation", "--layer", "--prune-percent", "--resume"):
            self.assertIn(flag, result.stdout)
        self.assertNotIn("direct_residual", result.stdout)
        self.assertNotIn("residual_lite", result.stdout)

    def test_defaults_select_optimized_perturbation_and_base_is_unpruned(self):
        args = self.args()
        self.assertEqual(
            (args.method, args.implementation, args.layer, args.prune_percent),
            ("perturbation", "optimized", 18, 70),
        )
        self.assertEqual(
            (args.max_image_pixels, args.max_image_patches, args.max_new_tokens), (4194304, 1, 32)
        )
        self.assertEqual((args.dtype, args.greedy_backend), ("bf16", "reference"))
        base = self.args("--method", "base")
        self.assertEqual(base.prune_percent, 0)
        self.assertIsNone(base.keep_tokens)
        self.assertEqual(self.args("--method", "nocover").method, "nocover")

    def test_invalid_or_mixed_budgets_fail_before_model_loading(self):
        invalid = (
            ("--method", "direct_residual"),
            ("--prune-percent", "100"),
            ("--prune-percent", "-1"),
            ("--layer", "0"),
            ("--query-chunk-size", "0"),
            ("--temperature", "0"),
            ("--perturbation-lambda", "-1"),
            ("--method", "base", "--prune-percent", "50"),
        )
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((SystemExit, ValueError)):
                    self.args(*extra)

    def test_macro_accuracy_and_per_case_costs_do_not_use_micro_or_totals(self):
        from evaluation.metrics import evaluate
        from evaluation.tasks import ALL_TASKS

        rows = [
            {
                "index": i,
                "correct": True,
                "status": "correct",
                "taskL3": "OC",
                "testtime": 0.5,
                "flops": 10**12,
            }
            for i in range(88)
        ]
        rows.extend(
            {
                "index": 88 + i,
                "correct": False,
                "status": "incorrect",
                "taskL3": task,
                "testtime": 0.5,
                "flops": 10**12,
            }
            for i, task in enumerate(t for t in ALL_TASKS if t != "OC")
        )
        report = evaluate(rows)
        self.assertEqual(report["total"], 100)
        self.assertAlmostEqual(report["micro_acc"], 0.88)
        self.assertAlmostEqual(report["macro_acc"], 1 / 13)
        self.assertEqual(report["macro_task_count"], 13)
        self.assertEqual(report["avg_testtime"], 0.5)
        self.assertEqual(report["avg_tflops"], 1.0)
        self.assertEqual(report["total_tflops"], 100.0)

    def test_changed_method_budget_or_native_settings_cannot_resume(self):
        from evaluation.runner import _validate_resume_config

        args = self.args()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "results.jsonl"
            config = source.with_name("config.json")
            original = vars(args).copy()
            config.write_text(json.dumps(original), encoding="utf-8")
            _validate_resume_config(source, args)
            changes = (
                ("method", "base"),
                ("implementation", "reference"),
                ("layer", 15),
                ("prune_percent", 80),
                ("model", "another/model"),
                ("max_image_pixels", 65536),
            )
            for key, value in changes:
                config.write_text(json.dumps(dict(original, **{key: value})), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    _validate_resume_config(source, args)

    def test_resume_reuses_only_successes_and_keeps_original_metadata(self):
        from evaluation.data import Case
        from evaluation.runner import _select_resume_cases

        cases = [Case(id=i, index=i, question="Q", gt="A") for i in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "results.jsonl"
            successful = {"index": 0, "status": "correct", "selected_indices": [[1, 3]]}
            failed = {"index": 1, "status": "error", "error": "interrupted"}
            source.write_text(
                "\n".join(json.dumps(row) for row in (successful, failed)) + "\n", encoding="utf-8"
            )
            pending, reused = _select_resume_cases(cases, source)
            self.assertEqual([case.index for case in pending], [1, 2])
            self.assertEqual(reused, [successful])

    def test_base_costs_count_raw_attention_and_native_mergers_per_view(self):
        from evaluation.data import Case
        from evaluation.worker import make_record

        vision = SimpleNamespace(
            depth=3,
            hidden_size=4,
            intermediate_size=8,
            out_hidden_size=6,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0, 2],
        )
        text = SimpleNamespace(num_hidden_layers=2, hidden_size=8, intermediate_size=16)
        model = SimpleNamespace(
            config=SimpleNamespace(vision_config=vision, text_config=text),
            last_image_grid_thw=[[1, 2, 2], [1, 2, 4]],
        )
        output = SimpleNamespace(
            answer="A", seconds=0.2, input_tokens=10, output_tokens=2, image_tokens=3
        )
        case = Case(id="base", question="Q", gt="A", category="OC")
        record = make_record(output, case, vars(self.args("--method", "base")), model, "cpu")
        self.assertEqual(record["flops"], 48448)
        parts = record["flops_breakdown"]
        self.assertEqual(parts["vision_linear_flops"], 3 * 12 * 2 * (4 * 4**2 + 2 * 4 * 8))
        self.assertEqual(parts["vision_attention_flops"], 4 * 4 * 3 * (4**2 + 8**2))
        self.assertEqual(parts["score_merger_flops"], 0)
        self.assertEqual(parts["selector_flops"], 0)
        self.assertEqual(record["retention_ratio"], 1)
        self.assertIsNone(record["prefill_latency"])
        self.assertNotIn("selected_indices", record)
        self.assertNotIn("pruning_diagnostics", record)

    def test_summary_diagnostics_keep_ids_without_full_probability_maps(self):
        from evaluation.data import Case
        from evaluation.worker import make_record

        vision = SimpleNamespace(
            depth=3,
            hidden_size=4,
            intermediate_size=8,
            out_hidden_size=6,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0, 2],
        )
        text = SimpleNamespace(num_hidden_layers=2, hidden_size=8, intermediate_size=16)
        model = SimpleNamespace(config=SimpleNamespace(vision_config=vision, text_config=text))
        output = SimpleNamespace(
            answer="A",
            seconds=0.2,
            input_tokens=9,
            output_tokens=2,
            image_tokens=2,
            image_grid_thw=[[1, 2, 2], [1, 2, 4]],
            selected_indices=[[0], [1]],
            selector_flops=100,
            verification_flops=0,
            pruning_latency=0.1,
            prefill_latency=0.05,
            selection_budgets=[(0, 1, 0), (0, 1, 0)],
            vision_layer_token_counts=[12, 12, 12],
            q_flowcover_debug=[
                {
                    "N": 2,
                    "K": 1,
                    "support": [0.4, 0.6],
                    "query_group_count": 1,
                    "source_query_count": 4,
                }
            ],
        )
        case = Case(id="pruned", question="Q", gt="A", category="OC")
        cfg = vars(self.args("--layer", "2"))
        summary = make_record(output, case, cfg, model, "cpu")
        self.assertEqual(summary["selected_indices"], [[0], [1]])
        self.assertEqual(summary["pruning_diagnostics"][0]["query_group_count"], 1)
        self.assertNotIn("support", summary["pruning_diagnostics"][0])
        detailed = make_record(output, case, dict(cfg, diagnostics="full"), model, "cpu")
        self.assertEqual(detailed["pruning_diagnostics"][0]["support"], [0.4, 0.6])

    def test_repeated_interrupts_keep_the_union_of_old_and_new_successes(self):
        from evaluation import runner
        from evaluation.data import Case
        from evaluation.metrics import load_jsonl

        class Queue:
            def __init__(self, events):
                self.events = iter(events)

            def get(self, timeout=None):
                item = next(self.events)
                if isinstance(item, BaseException):
                    raise item
                return item

            def close(self):
                pass

            def join_thread(self):
                pass

        class Process:
            exitcode = 0

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        class Context:
            def __init__(self, events):
                self.queue, self.indices = Queue(events), []

            def Queue(self):
                return self.queue

            def Process(self, *, target, args):
                self.indices.extend(case["index"] for case in args[1])
                return Process()

        def row(index):
            return {
                "index": index,
                "status": "correct",
                "correct": True,
                "taskL3": "OC",
                "gt": "A",
                "model_answer": "A",
                "testtime": 1.0,
                "flops": 10**12,
                "original_image_tokens": 4,
                "retained_tokens": 2,
                "retention_ratio": 0.5,
                "selected_indices": [[0, 2]],
                "selection_budgets": [[0, 2, 0]],
                "per_view_group_counts": [4],
                "vision_merge_unit": 4,
                "vision_layer_token_counts": [16] * 19 + [8] * 8,
                "pruning_diagnostics": [{"support": [0.1, 0.2, 0.3, 0.4]}],
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs"
            source = output / "previous" / "results.jsonl"
            source.parent.mkdir(parents=True)

            def arguments():
                return entry().get_args(
                    [
                        "--data",
                        str(root / "cases.jsonl"),
                        "--model",
                        "fake/model",
                        "--keep-tokens",
                        "2",
                        "--dtype",
                        "fp32",
                        "--resume",
                        "--output-dir",
                        str(output),
                    ]
                )

            config = dict(vars(arguments()), schema=runner.SCHEMA, image_root=str(root))
            source.with_name("config.json").write_text(json.dumps(config), encoding="utf-8")
            source.write_text(json.dumps(row(0)) + "\n", encoding="utf-8")
            original = source.read_bytes()
            cases = [Case(id=i, index=i, question="Q", gt="A") for i in range(4)]

            def execute(events, interrupt):
                context = Context(events)
                with patch("evaluation.data.load_cases", return_value=cases), patch.object(
                    runner, "_devices", return_value=["cpu"]
                ), patch.object(
                    runner.mp, "get_context", return_value=context
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    if interrupt:
                        with self.assertRaises(KeyboardInterrupt):
                            runner.run(arguments())
                        report = None
                    else:
                        report = runner.run(arguments())
                checkpoint = runner._resume_source({"resume": True, "output_dir": str(output)})
                return context.indices, load_jsonl(checkpoint), report

            indices, saved, _ = execute([("row", row(1)), KeyboardInterrupt()], True)
            self.assertEqual(indices, [1, 2, 3])
            self.assertEqual(saved, [row(0), row(1)])
            indices, saved, _ = execute([("row", row(2)), KeyboardInterrupt()], True)
            self.assertEqual(indices, [2, 3])
            self.assertEqual(saved, [row(0), row(1), row(2)])
            indices, saved, report = execute([("row", row(3)), ("done", None)], False)
            self.assertEqual(indices, [3])
            self.assertEqual(saved, [row(i) for i in range(4)])
            self.assertEqual((report["total"], report["errors"]), (4, 0))
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
