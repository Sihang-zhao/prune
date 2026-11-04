"""Whole-selector equivalence, explicit dispatch and native gather checks."""

import unittest

import torch

from pruning.reference import select_q_flowcover_perturbation_views as reference
from pruning.selector import select_q_flowcover_perturbation_optimized_views as optimized


class OptimizedSelectorTests(unittest.TestCase):
    def test_same_objective_scores_indices_and_group_hashes(self):
        for seed in range(8):
            torch.manual_seed(seed)
            n, unit, h, d = 24, 4, 2, 8
            projected, text = torch.randn(n, 12), torch.randn(5, 12)
            q, k, v = [torch.randn(n * unit, h, d) for _ in range(3)]
            hidden = torch.randn(n * unit, 8)
            for method in ("q_flowcover_perturbation", "q_flowcover_perturbation_nocover"):
                for keep in (1, n // 2, n):
                    with self.subTest(seed=seed, method=method, keep=keep):
                        args = (
                            [projected],
                            text,
                            [q],
                            [k],
                            [v],
                            [hidden],
                            [keep],
                            [n * unit],
                            unit,
                            method,
                            [(4, 6)],
                        )
                        kwargs = dict(
                            scale=d**-0.5,
                            query_chunk_size=7,
                            group_chunk_size=3,
                            source_chunk_size=13,
                            compare_flow=True,
                        )
                        expected, er = reference(*args, **kwargs)
                        actual, ar = optimized(*args, **kwargs)
                        self.assertTrue(torch.equal(expected[0].indices, actual[0].indices))
                        for key in (
                            "text_scores",
                            "text_mass",
                            "pi",
                            "support",
                            "deletion_scores",
                            "omega",
                            "greedy_gains",
                        ):
                            torch.testing.assert_close(
                                torch.tensor(ar[0][key]),
                                torch.tensor(er[0][key]),
                                atol=3e-7,
                                rtol=3e-5,
                            )
                        self.assertEqual(ar[0]["raw_keep_idx_hash"], er[0]["raw_keep_idx_hash"])
                        self.assertEqual(ar[0]["flow_group_index"], er[0]["flow_group_index"])

    def test_explicit_native_branch_and_reference_indices_match(self):
        from test_native_model import PerturbationRuntimeTests

        fixture = PerturbationRuntimeTests()
        fixture.setUpClass()
        try:
            fixture.setUp()
            expected = fixture.run_native(fixture.runner())
            actual = fixture.runner()
            actual.perturbation_selector_impl = "optimized"
            actual.perturbation_max_attention_mb = 16.0
            actual.perturbation_greedy_backend = "scripted_local_refresh"
            fixture.run_native(actual)
            self.assertEqual(
                expected._last_compression_metadata["raw_keep_idx_hash"],
                actual._last_compression_metadata["raw_keep_idx_hash"],
            )
            self.assertEqual(actual._last_compression_metadata["selector_impl"], "optimized")
            self.assertIn("selector.py", actual._last_compression_metadata["selector_file"])
            for a, b in zip(
                expected._last_native_final_features, actual._last_native_final_features
            ):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            self.assertEqual(
                actual._last_compression_metadata["native_verification"]["status"], "passed"
            )
        finally:
            fixture.tearDownClass()
