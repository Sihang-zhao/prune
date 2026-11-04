"""Exercise sweep shells and the real summary CLI without loading a model."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCENTS = (50, 60, 70, 80, 85, 90, 95, 98)

# Act as the sweep's Python interpreter. Evaluation produces only tiny fixture
# files; summary invocations execute the actual implementation with site
# packages disabled, so these tests have no torch/CUDA/model dependency.
FAKE_INTERPRETER = r'''
import json
import os
from pathlib import Path
import subprocess
import sys

script, *arguments = sys.argv[1:]
if Path(script).name in {"summarize_results.py", "sweep_state.py"}:
    raise SystemExit(subprocess.call([sys.executable, "-S", script, *arguments]))
if Path(script).name != "testn.py":
    raise SystemExit("Unexpected sweep command: " + script)

def value(flag):
    return arguments[arguments.index(flag) + 1]

layer = int(value("--decision-layer-human"))
percent = int(value("--visual-prune-percent"))
output = Path(value("--output-dir")) / "fake_run"
output.mkdir(parents=True, exist_ok=False)
tasks = ("OC", "RC", "OLUC", "RLUC", "OCC", "OCL", "OMS", "OSR",
         "AD", "ECR", "RP", "RCCD", "CCR")
config = {
    "data": value("--data"), "model": value("--model"),
    "image_root": value("--data"), "prune_method": value("--prune-method"),
    "decision_layer_human": layer, "pruning_layer": layer,
    "visual_prune_percent": percent, "visual_keep_ratio": 1.0 - percent / 100.0,
    "compression_mode": value("--exec-mode"), "dtype": value("--dtype"),
    "effective_dtypes": [value("--dtype")], "case_count": 26,
    "vision_layers": f"6,{layer},24",
    "visual_pruning": True,
}
summary = {
    "total": 26, "correct": 13, "errors": 0, "macro_acc": 0.5,
    "micro_acc": 0.5, "macro_task_count": 13,
    "avg_testtime": layer / 100.0, "avg_tflops": (100 - percent) / 10.0,
    "taskL3": {task: {"total": 2, "correct": 1, "acc": 0.5} for task in tasks},
}
(output / "config.json").write_text(json.dumps(config), encoding="utf-8")
(output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
with open(os.environ["QFLOW_TEST_CALLS"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"layer": layer, "percent": percent, "output": str(output)}) + "\n")
'''


class SweepSummaryIntegrationTests(unittest.TestCase):
    def _run_sweep(self, name: str, layer: int | None = None, skip_layers: str = ""):
        temporary = tempfile.TemporaryDirectory(prefix="qflow sweep summary ")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        dataset = root / "data with spaces"
        model = root / "model with spaces"
        dataset.mkdir()
        model.mkdir()
        output = root / "results with spaces"
        interpreter = root / "fake python"
        interpreter.write_text(f"#!{sys.executable}\n" + FAKE_INTERPRETER, encoding="utf-8")
        interpreter.chmod(0o755)
        calls = root / "evaluation calls.jsonl"
        environment = dict(os.environ)
        for key in ("QFLOW_SUMMARY_ROOT", "QFLOW_SUMMARY_LAYERS"):
            environment.pop(key, None)
        environment.update(QFLOW_PYTHON=str(interpreter), QFLOW_TEST_CALLS=str(calls), QFLOW_SKIP_LAYERS=skip_layers)
        command = [
            "bash", str(PROJECT_ROOT / "scripts" / name),
            str(dataset), str(model), "0,1", str(output),
        ]
        if layer is not None:
            command.append(str(layer))
        completed = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertNotIn("Summary update failed", completed.stderr)
        # Rerunning the exact same command must skip completed experiments.
        repeated = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
        self.assertNotIn("Summary update failed", repeated.stderr)
        self.assertIn("[SKIP]", repeated.stdout)
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        evaluations = [json.loads(row) for row in calls.read_text(encoding="utf-8").splitlines()]
        # The matrix must maintain one shared table set, not four independent
        # layer tables. Leaf summary.json files are the fake benchmark inputs.
        for suffix in ("json", "csv", "md", "html"):
            tables = [path for path in output.rglob(f"summary.{suffix}")
                      if path.parent.name != "fake_run" and "summary_history" not in path.parts]
            self.assertEqual(tables, [output / f"summary.{suffix}"])
        self.assertEqual(report["warnings"], [])
        return output, report, csv_rows, evaluations

    def _assert_cells(self, report, csv_rows, evaluations, layers):
        cells = {(layer, percent) for layer in layers for percent in PERCENTS}
        self.assertEqual(report["completed_count"], len(cells))
        self.assertEqual(report["expected_count"], len(cells))
        self.assertEqual(len(report["rows"]), len(cells))
        self.assertEqual({(row["layer"], row["prune_percent"]) for row in report["rows"]}, cells)
        self.assertTrue(all(row["status"] == "completed" and row["cases"] == 26 for row in report["rows"]))
        self.assertTrue(all(row["metrics"]["avg"] == 50.0 and row["task_count"] == 13 for row in report["rows"]))
        self.assertEqual(len(csv_rows), len(cells))
        self.assertEqual({(int(row["Layer"]), int(row["Prune%"])) for row in csv_rows}, cells)
        self.assertEqual(len(evaluations), len(cells))
        self.assertEqual({(row["layer"], row["percent"]) for row in evaluations}, cells)
        self.assertEqual(len({row["output"] for row in evaluations}), len(cells))

    def test_single_layer_sweep_finishes_all_eight_summary_rows(self):
        output, report, csv_rows, evaluations = self._run_sweep("run_pruning_sweep.sh", layer=17)
        self._assert_cells(report, csv_rows, evaluations, (17,))
        for row in evaluations:
            self.assertEqual(Path(row["output"]), output / f"prune_{row['percent']}" / "fake_run")

    def test_four_layer_sweep_maintains_one_complete_32_row_table(self):
        output, report, csv_rows, evaluations = self._run_sweep("run_layer_pruning_sweep.sh")
        self._assert_cells(report, csv_rows, evaluations, (15, 16, 17, 18))
        for row in evaluations:
            self.assertEqual(
                Path(row["output"]),
                output / f"layer_{row['layer']}" / f"prune_{row['percent']}" / "fake_run",
            )

    def test_skip_layer_18_retains_32_row_grid_without_running_layer_18(self):
        output, report, csv_rows, evaluations = self._run_sweep("run_layer_pruning_sweep.sh", skip_layers="18")
        self.assertEqual(report["expected_count"], 32)
        self.assertEqual(report["completed_count"], 24)
        self.assertEqual(len(evaluations), 24)
        self.assertEqual({row["layer"] for row in evaluations}, {15, 16, 17})
        self.assertTrue(all(row["status"] == "pending" for row in report["rows"] if row["layer"] == 18))


if __name__ == "__main__":
    unittest.main()
