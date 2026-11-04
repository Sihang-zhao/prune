"""Sweep decisions must honor complete cells and matching checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import testn
from scripts.sweep_state import select_run


class SweepStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sweep state ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.model = self.root / "model"
        self.data.mkdir()
        self.model.mkdir()
        self.output = self.root / "output"

    def args(self):
        return testn.get_args([
            "--data", str(self.data), "--model", str(self.model), "--dtype", "bf16",
            "--decision-layer-human", "15", "--visual-prune-percent", "98",
            "--output-dir", str(self.output),
        ])

    def run_fixture(self, name, *, complete=False, errors=0, overrides=None, mtime=1_000_000_000):
        run = self.output / name
        run.mkdir(parents=True)
        args = self.args()
        config = dict(vars(args), data=str(self.data), image_root=str(self.data),
                      model=str(self.model), visual_keep_ratio=0.020000000000000018,
                      case_count=3080, effective_dtypes=["bf16"])
        config.update(overrides or {})
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        result = run / "results.jsonl"
        result.write_text('{"index": 0}\n', encoding="utf-8")
        os.utime(result, ns=(mtime, mtime))
        if complete:
            path = run / "summary.json"
            path.write_text(json.dumps({"total": 3080, "errors": errors}), encoding="utf-8")
            os.utime(path, ns=(mtime, mtime))
        return run

    def test_empty_cell_starts_new_run(self):
        self.assertEqual(select_run(self.args()), ("run", None))

    def test_complete_cell_wins_over_newer_interrupted_run(self):
        complete = self.run_fixture("complete", complete=True)
        self.run_fixture("interrupted", mtime=2_000_000_000)
        self.assertEqual(select_run(self.args()), ("skip", complete / "summary.json"))

    def test_partial_cell_uses_latest_matching_checkpoint(self):
        self.run_fixture("old")
        newest = self.run_fixture("new", mtime=2_000_000_000)
        self.run_fixture("wrong_dataset", overrides={"data": str(self.root / "subset")}, mtime=3_000_000_000)
        self.assertEqual(select_run(self.args()), ("resume", newest / "results.jsonl"))

    def test_other_method_layer_and_budget_are_never_reused(self):
        for key, value in (("prune_method", "q_flowcover"), ("pruning_layer", 18), ("visual_keep_ratio", 0.5)):
            self.run_fixture(key, complete=True, overrides={key: value})
        self.assertEqual(select_run(self.args()), ("run", None))

    def test_completed_run_with_errors_is_resumed_for_retry(self):
        run = self.run_fixture("with_errors", complete=True, errors=2)
        self.assertEqual(select_run(self.args()), ("resume", run / "results.jsonl"))

    def test_incomplete_summary_is_resumed_not_skipped(self):
        run = self.run_fixture("missing_cases", complete=True)
        (run / "summary.json").write_text('{"total":1681,"errors":0}', encoding="utf-8")
        self.assertEqual(select_run(self.args()), ("resume", run / "results.jsonl"))


if __name__ == "__main__":
    unittest.main()
