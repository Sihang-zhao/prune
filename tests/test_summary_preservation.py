"""Protect completed benchmark tables during interrupted or partial sweeps."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_summary_preservation_under_test", ROOT / "scripts/summarize_results.py")
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)
TASKS = ("OC", "RC", "OLUC", "RLUC", "OCC", "OCL", "OMS", "OSR", "AD", "ECR", "RP", "RCCD", "CCR")


class SummaryPreservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qflow_preserve_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefix = self.root / "tables" / "summary"
        self.data = self.root / "data"
        self.model = self.root / "model"

    def make_run(self, name, layer, cases=26):
        root = self.root / name
        run = root / "run"
        run.mkdir(parents=True)
        config = {
            "data": str(self.data), "image_root": str(self.data), "model": str(self.model),
            "dtype": "bf16", "effective_dtypes": ["bf16"], "case_count": cases,
            "prune_method": "q_flowcover_direct_residual", "compression_mode": "hard18",
            "decision_layer_human": layer, "pruning_layer": layer,
            "visual_prune_percent": 50, "vision_layers": f"6,{layer},24",
        }
        summary = {
            "total": cases, "errors": 0, "macro_acc": 0.5, "macro_task_count": 13,
            "avg_testtime": layer / 100, "avg_tflops": layer,
            "taskL3": {task: {"total": 2, "acc": 0.5} for task in TASKS},
        }
        (run / "config.json").write_text(json.dumps(config))
        (run / "summary.json").write_text(json.dumps(summary))
        return root

    def saved_bytes(self):
        return {suffix: Path(str(self.prefix) + "." + suffix).read_bytes() for suffix in ("csv", "md", "html", "json")}

    def write(self, report):
        SUMMARY.write_report(report, self.prefix, preserve_existing=True)

    def test_completed_layer18_survives_scan_of_only_new_layer15(self):
        old = self.make_run("old", 18)
        new = self.make_run("new", 15)
        self.write(SUMMARY.build_report(old))
        original = self.saved_bytes()
        report = SUMMARY.build_report(new)
        self.write(report)
        self.assertEqual(report["completed_count"], 2)
        by_layer = {r["layer"]: r for r in report["rows"] if r["status"] == "completed"}
        self.assertEqual(set(by_layer), {15, 18})
        self.assertEqual(by_layer[18]["metrics"]["time"], 0.18)
        self.assertIn(str(old), by_layer[18]["source"])
        snapshots = list((self.prefix.parent / "summary_history").iterdir())
        self.assertEqual(len(snapshots), 1)
        for suffix, content in original.items():
            self.assertEqual((snapshots[0] / ("summary." + suffix)).read_bytes(), content)
        # Rebuilding the same partial report does not accumulate duplicate backups.
        self.write(SUMMARY.build_report(new))
        self.assertEqual(len(list((self.prefix.parent / "summary_history").iterdir())), 1)

    def test_new_incomplete_run_does_not_replace_completed_cell(self):
        old = self.make_run("old", 18)
        pending = self.make_run("pending", 18)
        (pending / "run" / "summary.json").unlink()
        self.write(SUMMARY.build_report(old))
        report = SUMMARY.build_report(pending, data=self.data, model=self.model, dtype="bf16")
        self.write(report)
        self.assertEqual(report["completed_count"], 1)
        complete = next(row for row in report["rows"] if row["status"] == "completed")
        self.assertEqual(complete["layer"], 18)
        self.assertEqual(complete["cases"], 26)

    def test_different_case_count_cannot_overwrite_previous_tables(self):
        self.write(SUMMARY.build_report(self.make_run("old", 18, cases=260)))
        original = self.saved_bytes()
        newer = SUMMARY.build_report(self.make_run("new", 15, cases=3080))
        with self.assertRaisesRegex(ValueError, "different experiment profile.*case_count"):
            self.write(newer)
        self.assertEqual(self.saved_bytes(), original)
        self.assertFalse((self.prefix.parent / "summary_history").exists())

    def test_different_method_cannot_overwrite_previous_tables(self):
        self.write(SUMMARY.build_report(self.make_run("old", 18)))
        original = self.saved_bytes()
        report = SUMMARY.build_report(self.root / "not_yet_run", method="q_flowcover")
        with self.assertRaisesRegex(ValueError, "different experiment profile.*method"):
            self.write(report)
        self.assertEqual(self.saved_bytes(), original)

    def test_empty_new_profile_still_checks_explicit_data_filter(self):
        self.write(SUMMARY.build_report(self.make_run("old", 18)))
        original = self.saved_bytes()
        report = SUMMARY.build_report(self.root / "not_yet_run", data=self.root / "different_dataset")
        with self.assertRaisesRegex(ValueError, "different experiment profile.*data"):
            self.write(report)
        self.assertEqual(self.saved_bytes(), original)

    def test_old_completed_cells_outside_requested_grid_are_preserved(self):
        self.write(SUMMARY.build_report(self.make_run("old", 18)))
        report = SUMMARY.build_report(self.make_run("new", 15), layers=(15,))
        self.write(report)
        complete = [row for row in report["rows"] if row["status"] == "completed"]
        self.assertEqual({row["layer"] for row in complete}, {15, 18})
        self.assertEqual(report["layers"], [15, 18])

    def test_identical_report_update_needs_no_backup(self):
        root = self.make_run("old", 18)
        self.write(SUMMARY.build_report(root))
        self.write(SUMMARY.build_report(root))
        self.assertFalse((self.prefix.parent / "summary_history").exists())

    def test_corrupt_previous_json_is_left_unchanged(self):
        root = self.make_run("old", 18)
        self.write(SUMMARY.build_report(root))
        self.prefix.with_suffix(".json").write_text("{unfinished")
        original = self.saved_bytes()
        with self.assertRaises(ValueError):
            self.write(SUMMARY.build_report(root))
        self.assertEqual(self.saved_bytes(), original)


if __name__ == "__main__":
    unittest.main()
