"""Compare every exact greedy marginal, including real R=0 and boundary ties."""
from __future__ import annotations

import unittest

import torch

from compression.perturbation_greedy_optimized import BACKENDS, greedy_direct_cover_optimized
from compression.q_flowcover_perturbation import greedy_direct_cover, sparse_local_affinity


class PerturbationOptimizedGreedyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def case(self, height, width, radius, seed):
        generator = torch.Generator().manual_seed(seed)
        count = height * width
        features = torch.randn(count, 13, generator=generator)
        affinity, table, valid = sparse_local_affinity(features, height, width, radius)
        text = torch.softmax(torch.randn(count, generator=generator), dim=0)
        residual = torch.rand(count, generator=generator)
        return text, residual, affinity, table, valid

    def compare(self, case, keep, **kw):
        text, residual, affinity, table, valid = case
        expected = greedy_direct_cover(text, residual, keep, affinity, table, valid)
        for backend in BACKENDS:
            actual = greedy_direct_cover_optimized(text, residual, keep, affinity, table, valid, backend=backend, **kw)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_all_round_gains_and_indices_match_reference_multiple_grids(self):
        for seed in (1, 29, 71):
            for height, width in ((1, 1), (1, 9), (3, 5), (8, 9)):
                for radius in (0, 1, 2):
                    case = self.case(height, width, radius, seed)
                    for keep in sorted({1, max(1, height * width // 2), height * width}):
                        with self.subTest(seed=seed, grid=(height, width), radius=radius, keep=keep):
                            self.compare(case, keep)
                            self.compare(case, keep, grid_shape=(height, width), local_radius=radius)

    def test_zero_residual_and_exact_ties_choose_lowest_original_ids(self):
        case = list(self.case(4, 7, 1, 18))
        case[0] = torch.ones(28)
        case[1] = torch.zeros(28)
        self.compare(case, 19)
        for backend in BACKENDS:
            ids, gains = greedy_direct_cover_optimized(case[0], case[1], 19, *case[2:], backend=backend)
            torch.testing.assert_close(ids, torch.arange(19), atol=0, rtol=0)
            torch.testing.assert_close(gains, torch.ones(19), atol=0, rtol=0)

    def test_asymmetric_affinity_on_symmetric_graph_and_padding(self):
        case = list(self.case(5, 6, 1, 20))
        generator = torch.Generator().manual_seed(51)
        affinity, table, valid = case[2:]
        affinity = torch.rand(affinity.shape, generator=generator).masked_fill(~valid, 0)
        affinity.masked_fill_(valid & (table == torch.arange(30)[:, None]), 1.)
        case[2] = affinity
        self.compare(case, 15)

    def test_r0_is_exact_modular_topk_not_general_static_topk(self):
        text, residual, affinity, table, valid = self.case(3, 8, 0, 5)
        ids, gains, info = greedy_direct_cover_optimized(text, residual, 10, affinity, table, valid, return_diagnostics=True)
        order = torch.argsort(text + residual, stable=True, descending=True)[:10]
        torch.testing.assert_close(ids, order.sort().values, atol=0, rtol=0)
        torch.testing.assert_close(gains, (text + residual)[order], atol=0, rtol=0)
        self.assertTrue(info["greedy_modular_self_only"])
        self.assertEqual(info["greedy_sequential_rounds"], 0)
        case = self.case(3, 8, 1, 5)
        _, _, info = greedy_direct_cover_optimized(case[0], case[1], 10, *case[2:], return_diagnostics=True)
        self.assertFalse(info["greedy_modular_self_only"])
        self.assertEqual(info["greedy_sequential_rounds"], 10)

    def test_grid_structure_uses_25_affected_slots_and_reuses_no_values(self):
        case = self.case(8, 8, 1, 3)
        _, _, info = greedy_direct_cover_optimized(
            case[0], case[1], 32, *case[2:],
            grid_shape=(8, 8), local_radius=1, return_diagnostics=True,
        )
        self.assertEqual(info["greedy_affected_candidate_slots"], 25)
        self.assertEqual(info["greedy_cached_grid_shape"], [8, 8])
        self.assertFalse(info["greedy_cross_sample_tensor_cache"])
        self.compare(case, 32, grid_shape=(8, 8), local_radius=1)

    def test_reference_backend_explicitly_keeps_original_full_refresh_path(self):
        case = self.case(8, 8, 1, 11)
        ids, gains, info = greedy_direct_cover_optimized(
            case[0], case[1], 32, *case[2:], backend="reference",
            grid_shape=(8, 8), local_radius=1, return_diagnostics=True,
        )
        expected = greedy_direct_cover(case[0], case[1], 32, *case[2:])
        torch.testing.assert_close(ids, expected[0], atol=0, rtol=0)
        torch.testing.assert_close(gains, expected[1], atol=0, rtol=0)
        self.assertEqual(info["greedy_backend"], "reference")
        self.assertEqual(info["greedy_affected_candidate_slots"], 0)
        self.assertIsNone(info["greedy_cached_grid_shape"])
        self.assertEqual(info["greedy_optimized_candidate_edge_visits_bound"], 32 * 64 * 9)
        self.assertEqual(info["estimated_flops"], 5 * 32 * 64 * 9)
        self.assertEqual(info["estimated_argmax_comparisons"], 32 * 63)
        self.assertEqual(info["greedy_sequential_rounds"], 32)

    def test_no_cross_call_state_or_input_mutation(self):
        first = self.case(5, 7, 1, 2)
        saved = [x.clone() for x in first]
        second = self.case(5, 7, 1, 3)
        for backend in BACKENDS:
            a = greedy_direct_cover_optimized(first[0], first[1], 15, *first[2:], backend=backend)
            greedy_direct_cover_optimized(second[0], second[1], 15, *second[2:], backend=backend)
            b = greedy_direct_cover_optimized(first[0], first[1], 15, *first[2:], backend=backend)
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        for a, b in zip(first, saved):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_float64_preserves_dtype_and_exact_reduction(self):
        case = list(self.case(3, 9, 1, 27))
        for index in (0, 1, 2):
            case[index] = case[index].double()
        # The original stores marginal gains as FP32 irrespective of inputs;
        # the optimized interface preserves this diagnostic dtype too.
        expected = greedy_direct_cover(case[0], case[1], 13, *case[2:])
        for backend in BACKENDS:
            actual = greedy_direct_cover_optimized(case[0], case[1], 13, *case[2:], backend=backend)
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)

    def test_invalid_backend_shapes_types_and_budget_raise(self):
        text, residual, affinity, table, valid = self.case(2, 3, 1, 3)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            greedy_direct_cover_optimized(text, residual, 3, affinity, table, valid, backend="silent_fallback")
        for keep in (0, 7):
            with self.assertRaises(ValueError):
                greedy_direct_cover_optimized(text, residual, keep, affinity, table, valid)
        with self.assertRaises(ValueError):
            greedy_direct_cover_optimized(text, residual[:-1], 3, affinity, table, valid)
        with self.assertRaises(TypeError):
            greedy_direct_cover_optimized(text, residual, 3, affinity, table.int(), valid)
        with self.assertRaises(TypeError):
            greedy_direct_cover_optimized(text, residual, 3, affinity, table, valid.float())


if __name__ == "__main__":
    unittest.main()
