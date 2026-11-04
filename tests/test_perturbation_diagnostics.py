"""ID/rank/record checks for optional post-inference CPU diagnostics."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from compression.perturbation_diagnostics import (
    analyze_view, save_perturbation_diagnostics, spearman_correlation,
)


class PerturbationDiagnosticsTests(unittest.TestCase):
    def fixture(self):
        return dict(
            N=6, K=2, grid=[2, 3], merge_unit=4,
            text_scores=[.5, .2, .1, -.1, -.2, -.3],
            deletion_scores=[.02, .01, .5, .4, .2, .1],
            support=[.1, .2, .3, .15, .15, .1],
            query_group_index=[0, 1], text_topk=[0, 1],
            flow_group_index=[0, 2], final_group_index=[0, 3],
        )

    def test_native_ids_and_counterfactual_differences(self):
        summary, arrays = analyze_view(self.fixture())
        self.assertEqual(summary["text_vs_final"], dict(intersection=[0], text_only=[1], final_only=[3]))
        self.assertEqual(summary["flow_vs_final"], dict(intersection=[0], flow_only=[2], final_only=[3]))
        np.testing.assert_array_equal(arrays["group_row"], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(arrays["group_col"], [0, 1, 2, 0, 1, 2])
        self.assertEqual(summary["perturbation_top10"][0]["group_id"], 2)

    def test_spearman_uses_tie_midranks_and_undefined_is_null(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 2, 4], [1, 2, 2, 4]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 2, 4], [4, 2, 2, 1]), -1.0)
        self.assertIsNone(spearman_correlation([1, 1], [1, 2]))

    def test_invalid_mapping_budget_indices_and_nonfinite_fail(self):
        for replacement in (
            dict(grid=[3, 3]), dict(final_group_index=[0, 0]),
            dict(final_group_index=[0, 6]), dict(final_group_index=[0, .5]),
            dict(K=3), dict(support=[float("nan")] * 6),
            dict(group_order="sorted_by_score"),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    analyze_view(dict(self.fixture(), **replacement))

    def test_zero_scores_are_not_invented_high_sensitivity(self):
        summary, arrays = analyze_view(dict(self.fixture(), deletion_scores=[0.0] * 6))
        self.assertFalse(summary["deletion_scores_informative"])
        self.assertEqual(summary["high_perturbation_low_text"]["count"], 0)
        self.assertIsNone(summary["spearman_deletion_vs_support"])
        np.testing.assert_array_equal(arrays["deletion_scores_normalized"], np.zeros(6))

    def test_missing_flow_is_explicitly_unavailable(self):
        record = self.fixture()
        record.pop("flow_group_index")
        summary, arrays = analyze_view(record)
        self.assertFalse(summary["flow_comparison_available"])
        self.assertIsNone(summary["flow_vs_final"])
        self.assertNotIn("flow_keep_mask", arrays)

    def test_full_budget_unmeasured_scores_are_na_not_zero(self):
        record = dict(self.fixture(), K=6, query_group_index=list(range(6)),
                      text_topk=list(range(6)), final_group_index=list(range(6)),
                      flow_group_index=list(range(6)), deletion_scores=[], support=[],
                      statistics_computed=False)
        summary, arrays = analyze_view(record)
        self.assertFalse(summary["statistics_computed"])
        self.assertIsNone(summary["deletion_scores_informative"])
        self.assertIsNone(summary["high_perturbation_low_text"]["count"])
        self.assertEqual(summary["perturbation_top10"], [])
        self.assertTrue(np.isnan(arrays["deletion_scores"]).all())
        self.assertTrue(np.isnan(arrays["support"]).all())
        with tempfile.TemporaryDirectory() as folder:
            result = save_perturbation_diagnostics(folder, "full", [Image.new("RGB", (40, 30))], [record], question="Full budget?")
            content = (Path(result["directory"]) / "view_000" / "groups.csv").read_text()
            self.assertIn("NA", content)
            self.assertNotIn("nan", content)

    def test_render_saves_panels_before_composition_and_exact_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            result = save_perturbation_diagnostics(
                folder, "case/7", [Image.new("RGB", (240, 160), "#bed1b0")],
                [self.fixture()], question="Which houses border the river?",
                ground_truth="A", answer="A",
            )
            root = Path(result["directory"]) / "view_000"
            self.assertTrue((root / "summary.json").is_file())
            self.assertEqual(len(list((root / "subfig").glob("*.png"))), 12)
            with Image.open(root / "overview.png") as image:
                self.assertEqual(image.size, (2880, 2160))
            with Image.open(root / "input.png") as image:
                self.assertEqual(image.size, (240, 160))
            with np.load(root / "arrays.npz") as arrays:
                np.testing.assert_array_equal(arrays["deletion_scores"], self.fixture()["deletion_scores"])
                np.testing.assert_array_equal(arrays["group_id"], np.arange(6))


if __name__ == "__main__":
    unittest.main()
