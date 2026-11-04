"""Completed runs share a table only when their evaluation inputs agree."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize.py"


def module():
    spec = importlib.util.spec_from_file_location("_summary_under_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="perturbation summary ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "runs"
        self.prefix = self.root / "table" / "summary"
        self.subject = module()

    def fixture(
        self,
        name,
        *,
        method="perturbation",
        config=None,
        summary=None,
        complete=True,
        mtime=1_000_000_000,
    ):
        from evaluation.tasks import ALL_TASKS

        run = self.input / name
        run.mkdir(parents=True)
        cfg = dict(
            schema="qwen3_perturbation_v1",
            data=str(self.root / "data"),
            model=str(self.root / "model"),
            image_root=str(self.root / "data"),
            method=method,
            implementation="optimized",
            layer=18,
            prune_percent=0 if method == "base" else 70,
            keep_tokens=None,
            dtype="bf16",
            case_count=100,
            split="train",
        )
        cfg.update(config or {})
        (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        if complete:
            tasks = {task: {"total": 1, "correct": 0, "acc": 0.0} for task in ALL_TASKS}
            tasks["OC"] = {"total": 88, "correct": 88, "acc": 1.0}
            values = dict(
                total=100,
                errors=0,
                micro_acc=0.88,
                macro_acc=1 / 13,
                macro_task_count=13,
                avg_testtime=0.125,
                avg_tflops=2.5,
                total_test_time=1000,
                total_tflops=250,
                taskL3=tasks,
            )
            values.update(summary or {})
            target = run / "summary.json"
            target.write_text(json.dumps(values), encoding="utf-8")
            os.utime(target, ns=(mtime, mtime))
        return run

    def test_base_and_perturbation_use_task_macro_and_per_case_units(self):
        self.fixture("base", method="base")
        self.fixture("pruned")
        rows = self.subject.summarize(self.input, self.prefix)
        self.assertEqual(len(rows), 2)
        with Path(str(self.prefix) + ".csv").open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(
            {row["Method"] for row in records}, {"base/optimized", "perturbation/optimized"}
        )
        for row in records:
            self.assertEqual(row["Avg."], "7.69")
            self.assertEqual(row["time (s)"], "0.1250")
            self.assertEqual(row["TFLOPs"], "2.500000")
            self.assertEqual(row["OC"], "100.00")
            self.assertEqual(row["RC"], "0.00")
            self.assertEqual(row["Cases"], "100")
        self.assertEqual(
            next(row for row in records if row["Method"].startswith("base"))["Layer"], "—"
        )
        for extension in ("md", "html"):
            text = Path(str(self.prefix) + "." + extension).read_text()
            self.assertIn('<th colspan="8">Perception</th>', text)
            self.assertIn('<th colspan="5">Reasoning</th>', text)

    def test_only_completed_new_schema_runs_enter_latest_matching_cell(self):
        self.fixture("old_complete", mtime=1_000_000_000)
        self.fixture("new_complete", summary={"avg_testtime": 0.5}, mtime=2_000_000_000)
        self.fixture("newest_running", complete=False)
        self.fixture("wrong_count", summary={"total": 99}, mtime=3_000_000_000)
        self.fixture(
            "old_schema",
            config={"schema": "old", "data": "another dataset"},
            summary={"avg_testtime": 100},
            mtime=4_000_000_000,
        )
        self.subject.summarize(self.input, self.prefix)
        with Path(str(self.prefix) + ".csv").open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["time (s)"], "0.5000")

    def test_mixed_profiles_reject_before_overwriting_existing_reports(self):
        self.fixture("valid")
        self.subject.summarize(self.input, self.prefix)
        originals = {
            suffix: Path(str(self.prefix) + "." + suffix).read_bytes()
            for suffix in ("csv", "md", "html")
        }
        self.fixture(
            "different_data", config={"data": str(self.root / "other_data"), "prune_percent": 80}
        )
        with self.assertRaisesRegex(ValueError, "different datasets"):
            self.subject.summarize(self.input, self.prefix)
        for suffix, expected in originals.items():
            self.assertEqual(Path(str(self.prefix) + "." + suffix).read_bytes(), expected)

    def test_help_is_standalone_and_run_summary_cannot_be_overwritten(self):
        help_result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--input-root", help_result.stdout)
        run = self.fixture("valid")
        original = (run / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "individual run"):
            self.subject.summarize(self.input, run / "summary")
        self.assertEqual((run / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
