"""Independent contracts for layer/ratio aggregation, using tiny JSON fixtures."""
from __future__ import annotations

import csv
from html.parser import HTMLParser
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "summarize_results.py"
TASKS = ("OC", "RC", "OLUC", "RLUC", "OCC", "OCL", "OMS", "OSR", "AD", "ECR", "RP", "RCCD", "CCR")


def _load_module():
    spec = importlib.util.spec_from_file_location("_qflow_summary_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Headers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headers = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        if tag == "th":
            self._current = [dict(attrs), ""]

    def handle_data(self, data):
        if self._current is not None:
            self._current[1] += data

    def handle_endtag(self, tag):
        if tag == "th" and self._current is not None:
            self.headers.append(self._current)
            self._current = None


class SummaryReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="qflow_summary_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "dataset_a"
        self.model = self.root / "model_a"
        self.data.mkdir()
        self.model.mkdir()

    def make_run(self, name="run", *, layer=18, percent=50, config_overrides=None,
                 summary_overrides=None, summary=True, mtime=1_000_000_000):
        run = self.root / "outputs" / name
        run.mkdir(parents=True)
        config = {
            "data": str(self.data), "image_root": str(self.data), "model": str(self.model),
            "split": "train", "dtype": "auto", "effective_dtypes": ["bf16"],
            "prune_method": "q_flowcover_direct_residual", "compression_mode": "hard18",
            "exec_mode": "hard18", "decision_layer_human": layer, "pruning_layer": layer,
            "decision_block_index": layer - 1, "vision_layers": f"6,{layer},24",
            "visual_pruning": True, "visual_prune_percent": percent,
            "visual_keep_ratio": 1.0 - percent / 100.0, "visual_keep_tokens": None,
            "case_count": 100, "limit": None, "max_image_pixels": 4194304,
            "max_image_patches": 1, "max_new_tokens": 32, "enable_thinking": False,
            "q_flowcover_query_chunk": 256, "q_flowcover_candidate_chunk": 64,
            "q_flowcover_local_radius": 1, "q_flowcover_radius": 1,
            "q_flowcover_query_mass": 0.9, "q_flowcover_adaptive_radius_max": 3,
            "q_flowcover_affinity_mode": "raw", "text_source": "baseline",
            "patch_selection": "uniform", "include_global_view": True,
            "tile_single_images": True, "patch_size": 448, "patch_stride": 336,
            "global_view_size": 448, "decode_hf_images": False,
        }
        config.update(config_overrides or {})
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if summary:
            # 88 successes in one large task, zero in twelve one-case tasks.
            # Sample-weighted accuracy=.88, task-balanced accuracy=1/13.
            tasks = {task: {"correct": 0, "total": 1, "acc": 0.0} for task in TASKS}
            tasks["OC"] = {"correct": 88, "total": 88, "acc": 1.0}
            values = {
                "total": 100, "correct": 88, "errors": 0, "valid_total": 100,
                "acc": 0.88, "micro_acc": 0.88, "macro_acc": 1.0 / 13.0,
                "macro_task_count": 13, "taskL3": tasks,
                "avg_testtime": 0.125, "total_test_time": 9876.0,
                "avg_tflops": 1.25, "total_tflops": 125.0, "avg_tflops_per_sec": 99.0,
            }
            values.update(summary_overrides or {})
            path = run / "summary.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            os.utime(path, ns=(mtime, mtime))
        return run

    def report(self, **kwargs):
        return self.module.build_report([self.root / "outputs"], **kwargs)

    @staticmethod
    def completed_row(report):
        rows = [row for row in report["rows"] if row["status"] != "pending"]
        if len(rows) != 1:
            raise AssertionError(f"Expected one completed row, found {len(rows)}")
        return rows[0]

    def test_macro_accuracy_and_per_case_time_flops_use_the_right_fields(self):
        self.make_run()
        report = self.report()
        row = self.completed_row(report)
        self.assertEqual(report["expected_count"], 32)
        self.assertEqual(report["completed_count"], 1)
        self.assertAlmostEqual(row["metrics"]["avg"], 100.0 / 13.0)
        self.assertEqual(row["metrics"]["time"], 0.125)
        self.assertEqual(row["metrics"]["tflops"], 1.25)
        self.assertEqual(row["task_accuracies"]["OC"], 100.0)
        self.assertEqual(row["task_accuracies"]["RC"], 0.0)
        self.assertEqual(set(row["task_accuracies"]), set(TASKS))

    def test_empty_grid_has_32_pending_cells_and_no_invented_zeroes(self):
        (self.root / "outputs").mkdir()
        report = self.report()
        self.assertEqual(report["completed_count"], 0)
        self.assertEqual(len(report["rows"]), 32)
        self.assertEqual({(r["layer"], r["prune_percent"]) for r in report["rows"]},
                         {(l, p) for l in (15, 16, 17, 18) for p in (50, 60, 70, 80, 85, 90, 95, 98)})
        for row in report["rows"]:
            self.assertEqual(row["status"], "pending")
            self.assertTrue(all(v is None for v in row["metrics"].values()))
            self.assertTrue(all(v is None for v in row["task_accuracies"].values()))

    def test_latest_completed_wins_but_a_newer_config_only_run_does_not(self):
        self.make_run("z_old_complete", summary_overrides={"avg_testtime": 9.0}, mtime=1_000_000_000)
        winner = self.make_run("a_new_complete", summary_overrides={"avg_testtime": 2.0}, mtime=2_000_000_000)
        self.make_run("zz_newest_running", summary=False, mtime=3_000_000_000)
        report = self.report()
        row = self.completed_row(report)
        self.assertEqual(row["metrics"]["time"], 2.0)
        self.assertIn(winner.name, str(row["source"]))

    def test_malformed_or_incomplete_summary_is_skipped_with_a_warning(self):
        self.make_run("valid", mtime=1_000_000_000)
        broken = self.make_run("bad_json", mtime=3_000_000_000)
        (broken / "summary.json").write_text("{unfinished", encoding="utf-8")
        broken_config = self.make_run("bad_config", mtime=3_000_000_000)
        (broken_config / "config.json").write_text("{unfinished", encoding="utf-8")
        self.make_run("wrong_count", summary_overrides={"total": 99}, mtime=4_000_000_000)
        report = self.report()
        self.assertEqual(report["completed_count"], 1)
        self.assertIn("valid", str(self.completed_row(report)["source"]))
        self.assertGreaterEqual(len(report["warnings"]), 3)

    def test_layer_15_and_98_percent_are_identified_from_config(self):
        self.make_run("arbitrary_directory_name", layer=15, percent=98)
        row = self.completed_row(self.report())
        self.assertEqual((row["layer"], row["prune_percent"]), (15, 98))

    def test_missing_task_is_none_not_a_zero_accuracy(self):
        run = self.make_run()
        path = run / "summary.json"
        summary = json.loads(path.read_text())
        del summary["taskL3"]["RC"]
        summary.update(macro_acc=1.0 / 12.0, macro_task_count=12)
        path.write_text(json.dumps(summary))
        report = self.report()
        row = self.completed_row(report)
        self.assertIsNone(row["task_accuracies"]["RC"])
        self.assertIsNone(row["metrics"]["avg"])
        self.assertEqual(row["task_accuracies"]["OC"], 100.0)
        self.assertTrue(report["warnings"])

    def test_mixed_dataset_profiles_require_an_explicit_filter(self):
        self.make_run("dataset_a", layer=15)
        other = self.root / "dataset_b"
        self.make_run("dataset_b", layer=16, config_overrides={"data": str(other), "image_root": str(other)})
        with self.assertRaisesRegex(ValueError, "different experiment profiles"):
            self.report()
        report = self.report(data=str(self.data))
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(self.completed_row(report)["layer"], 15)

    def test_effective_auto_and_explicit_bf16_are_the_same_profile(self):
        self.make_run("auto_bf16", layer=15)
        self.make_run("explicit_bf16", layer=16, config_overrides={"dtype": "bf16"})
        report = self.report(dtype="bf16")
        self.assertEqual(report["completed_count"], 2)
        self.assertEqual({r["dtype"] for r in report["rows"] if r["status"] != "pending"}, {"bf16"})

    def test_mixed_models_require_an_explicit_filter(self):
        self.make_run("model_a", layer=15)
        self.make_run("model_b", layer=16, config_overrides={"model": str(self.root / "model_b")})
        with self.assertRaisesRegex(ValueError, "different experiment profiles"):
            self.report()
        report = self.report(model=str(self.model))
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(self.completed_row(report)["layer"], 15)

    def test_mixed_dtype_profiles_are_rejected_or_filtered(self):
        self.make_run("bf16", layer=15)
        self.make_run("fp16", layer=16, config_overrides={"dtype": "fp16", "effective_dtypes": ["fp16"]})
        with self.assertRaisesRegex(ValueError, "different experiment profiles"):
            self.report()
        report = self.report(dtype="bf16")
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(self.completed_row(report)["layer"], 15)

    def test_mixed_local_radius_is_not_silently_combined(self):
        self.make_run("radius1", layer=15)
        self.make_run("radius2", layer=16, config_overrides={"q_flowcover_local_radius": 2})
        with self.assertRaisesRegex(ValueError, "different experiment profiles"):
            self.report()

    def test_other_methods_and_execution_modes_do_not_enter_the_table(self):
        self.make_run("target")
        self.make_run("other_method", layer=15, config_overrides={"prune_method": "direct_residual"})
        self.make_run("other_mode", layer=16,
                      config_overrides={"compression_mode": "full19_gather", "exec_mode": "full19_gather"})
        self.assertEqual(self.report()["completed_count"], 1)

    def test_completed_errors_are_visible_in_status_and_metadata(self):
        self.make_run(summary_overrides={"errors": 2, "valid_total": 98})
        report = self.report()
        row = self.completed_row(report)
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(row["status"], "completed_with_errors")
        self.assertEqual(row["errors"], 2)

    def test_outputs_have_two_level_headers_and_machine_readable_data(self):
        self.make_run(layer=15, percent=98)
        report = self.report()
        paths = self.module.write_report(report, self.root / "report" / "all_results")
        self.assertEqual(set(paths), {"csv", "md", "html", "json"})
        for path in paths.values():
            self.assertTrue(Path(path).is_file())
        parsed = _Headers()
        parsed.feed(Path(paths["html"]).read_text())
        self.assertTrue(any(attrs.get("colspan") == "8" and text.strip() == "Perception"
                            for attrs, text in parsed.headers))
        self.assertTrue(any(attrs.get("colspan") == "5" and text.strip() == "Reasoning"
                            for attrs, text in parsed.headers))
        header_text = [text.strip() for _, text in parsed.headers]
        self.assertEqual(tuple(t for t in header_text if t in TASKS), TASKS)
        with Path(paths["csv"]).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(task in rows[0] for task in TASKS))
        self.assertTrue(any(float(row["OC"]) == 100.0 for row in rows
                            if row["OC"] not in ("", "--", "—")))
        self.assertTrue(any(row["OC"] in ("", "--", "—") for row in rows))
        loaded = json.loads(Path(paths["json"]).read_text())
        self.assertEqual((loaded["completed_count"], loaded["expected_count"]), (1, 32))
        for name in ("md", "html"):
            text = Path(paths[name]).read_text()
            self.assertIn("OC", text)
            self.assertIn("TFLOPs", text)

    def test_cli_help_uses_only_the_standard_library(self):
        run = subprocess.run([sys.executable, "-S", str(SCRIPT_PATH), "--help"],
                             capture_output=True, text=True, check=True)
        self.assertIn("--output", run.stdout)
        self.assertIn("--data", run.stdout)
        self.assertIn("--dtype", run.stdout)

    def test_report_cannot_overwrite_original_run_files(self):
        run = self.make_run()
        report = self.report()
        originals = {name: (run / name).read_bytes() for name in ("config.json", "summary.json")}
        for prefix in ("summary", "config", "results"):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "overwrite an original run"):
                self.module.write_report(report, run / prefix)
        for name, contents in originals.items():
            self.assertEqual((run / name).read_bytes(), contents)
        self.assertFalse((run / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
