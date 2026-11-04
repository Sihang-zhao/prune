"""Do not infer persistence from absent or mismatched scalar diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.summarize_perturbation import build_report, write_report


class PerturbationSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def snapshot(self, name, layer, scores, *, actual_id="real-7", grid=(2, 3), image_color="white", query=0):
        sample = self.root / name / "diagnostics" / "sample_000000"
        view = sample / "view_000"
        view.mkdir(parents=True)
        count = len(scores)
        ids = np.arange(count)
        query_mask = np.zeros(count, dtype=bool)
        query_mask[query] = True
        np.savez_compressed(view / "arrays.npz", group_id=ids, group_row=ids // grid[1],
                            group_col=ids % grid[1], deletion_scores=np.asarray(scores), query_mask=query_mask)
        Image.new("RGB", (12, 10), image_color).save(view / "input.png")
        (sample / "summary.json").write_text(json.dumps({
            "sample_id": "000000", "question": "Which building borders the river?", "options": [], "ground_truth": "A",
            "extra": {"sample_id": actual_id, "score_layer": layer, "perturbation_layer": layer + 1},
            "views": [{"N": count, "K": 1, "grid": list(grid), "algorithm_id": "q_flowcover_perturbation", "statistics_computed": True}],
        }))
        (self.root / name / "config.json").write_text(json.dumps({"model": "same/model", "data": "same/data", "seed": 0}))
        return sample

    def test_one_measured_layer_does_not_claim_persistence(self):
        self.snapshot("one", 12, [0, 1, 2, 3, 4, 5])
        report = build_report([self.root])
        self.assertFalse(report["cross_layer_persistence_measured"])
        self.assertEqual(report["pairs"], [])
        write_report(report, self.root / "summary")
        self.assertTrue((self.root / "summary" / "summary.json").is_file())

    def test_two_layers_compare_original_ids_and_recompute_ranks(self):
        self.snapshot("one", 12, [0, 1, 2, 3, 4, 5])
        self.snapshot("two", 18, [0, 5, 4, 3, 2, 1])
        report = build_report([self.root])
        self.assertTrue(report["cross_layer_persistence_measured"])
        pair = report["pairs"][0]
        self.assertEqual(pair["sample_id"], "real-7")
        self.assertEqual(pair["common_eligible"], 5)
        self.assertAlmostEqual(pair["spearman"], -1.0)
        self.assertEqual(pair["top_overlap_coefficient"], 0.)
        self.assertEqual({row["group_id"] for row in report["rankings"]}, {1, 2, 3, 4, 5})
        self.assertTrue(all(row["perturbation_layer"] == row["score_layer"] + 1 for row in report["rankings"]))

    def test_equal_directory_indices_do_not_join_different_actual_ids(self):
        self.snapshot("one", 12, [0, 1, 2, 3, 4, 5], actual_id=7)
        self.snapshot("two", 18, [0, 1, 2, 3, 4, 5], actual_id="7")
        report = build_report([self.root])
        self.assertEqual(report["snapshot_count"], 2)
        self.assertFalse(report["cross_layer_persistence_measured"])

    def test_mismatched_input_or_grid_is_rejected_not_resized(self):
        self.snapshot("one", 12, [0, 1, 2, 3, 4, 5])
        self.snapshot("two", 18, [0, 1, 2, 3, 4, 5], image_color="black")
        report = build_report([self.root])
        self.assertFalse(report["cross_layer_persistence_measured"])
        self.assertTrue(any("identity/mapping mismatch" in warning for warning in report["warnings"]))
        # Even identical image bytes cannot align different grids by N alone.
        self.snapshot("three", 14, [0, 1, 2, 3, 4, 5], grid=(3, 2))
        report = build_report([self.root / "one", self.root / "three"])
        self.assertFalse(report["cross_layer_persistence_measured"])

    def test_constant_scores_and_conflicting_repeats_do_not_invent_stability(self):
        self.snapshot("one", 12, [0, 0, 0, 0, 0, 0])
        self.snapshot("two", 18, [0, 0, 0, 0, 0, 0])
        report = build_report([self.root])
        self.assertFalse(report["cross_layer_persistence_measured"])
        self.assertIsNone(report["pairs"][0]["spearman"])
        self.assertIsNone(report["pairs"][0]["top_overlap_coefficient"])
        self.snapshot("repeat", 18, [0, 1, 2, 3, 4, 5])
        report = build_report([self.root])
        self.assertEqual(report["pairs"], [])
        self.assertTrue(any("conflicting repeated layer" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
