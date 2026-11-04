"""Exact-reference checks for the rectangular/Value-Gram statistics path."""

from __future__ import annotations

import unittest

import torch

from pruning.reference import group_deletion_statistics as reference
from pruning.reference import greedy_direct_cover, sparse_local_affinity
from pruning.statistics import group_deletion_statistics_optimized as optimized


def explicit(q, k, v, query, pi, unit, *, scale=0.4, mask=None, dtype=torch.float32):
    raw, heads = q.shape[:2]
    count = pi.numel()
    repeats = heads // int(k.shape[1])
    raw_queries = (query[:, None] * unit + torch.arange(unit)).flatten()
    weights = pi[query].to(dtype).repeat_interleave(unit) / unit
    query = q[raw_queries].transpose(0, 1).to(dtype)
    key = k.transpose(0, 1).to(dtype).repeat_interleave(repeats, 0)
    value = v.transpose(0, 1).to(dtype).repeat_interleave(repeats, 0)
    logits = query @ key.transpose(-1, -2) * scale
    if mask is not None:
        allowed = (
            mask[None, None]
            if mask.ndim == 1
            else mask[raw_queries][None] if mask.ndim == 2 else mask[:, raw_queries]
        )
        logits.masked_fill_(~allowed, -float("inf"))
    attention = logits.softmax(-1)
    full = attention @ value
    support = (attention.mean(0).reshape(-1, count, unit).sum(-1) * weights[:, None]).sum(0)
    scores = []
    for group in range(count):
        deleted_logits = logits.clone()
        deleted_logits[:, :, group * unit : (group + 1) * unit] = -float("inf")
        deleted = deleted_logits.softmax(-1) @ value
        score = (full - deleted).square().sum(-1).mean(0)
        scores.append((score * weights).sum())
    return torch.stack(scores), support


class OptimizedStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, seed=1702, *, groups=12, unit=4, heads=4, kv_heads=2, dim=8):
        rng = torch.Generator().manual_seed(seed)
        q = torch.randn(groups * unit, heads, dim, generator=rng)
        k = torch.randn(groups * unit, kv_heads, dim, generator=rng)
        v = torch.randn(groups * unit, kv_heads, dim, generator=rng)
        query = torch.arange(0, groups, 2)
        pi = torch.zeros(groups)
        pi[query] = torch.linspace(-1, 1, query.numel()).softmax(0)
        return q, k, v, query, pi, unit

    def test_matches_explicit_and_reference_for_multiple_chunks_and_seeds(self):
        for seed in range(4):
            for heads, kv_heads in ((4, 4), (4, 2), (4, 1)):
                data = self.fixture(seed, heads=heads, kv_heads=kv_heads)
                expected, expected_support = explicit(*data)
                baseline, baseline_support, _ = reference(
                    *data, scale=0.4, query_chunk_size=7, group_chunk_size=3, source_chunk_size=13
                )
                for chunk in (1, 7, 256):
                    actual, support, details = optimized(*data, scale=0.4, query_chunk_size=chunk)
                    torch.testing.assert_close(actual, expected, atol=7e-7, rtol=7e-6)
                    torch.testing.assert_close(actual, baseline, atol=7e-7, rtol=7e-6)
                    torch.testing.assert_close(support, expected_support, atol=2e-7, rtol=3e-6)
                    torch.testing.assert_close(support, baseline_support, atol=2e-7, rtol=3e-6)
                    self.assertEqual(details["ordinary_qk_source_scans"], 1)
                    self.assertFalse(details["full_attention_matrix_saved"])
                    self.assertLess(details["max_attention_chunk_shape"][1], data[0].shape[0])

    def test_half_inputs_keep_fp32_statistics_and_native_inputs_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16):
            q, k, v, query, pi, unit = self.fixture()
            data = q.to(dtype), k.to(dtype), v.to(dtype), query, pi, unit
            snapshots = [t.clone() for t in data[:5]]
            actual, support, _ = optimized(*data, scale=0.4, query_chunk_size=7)
            expected, expected_support = explicit(*data)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(support, expected_support, atol=3e-7, rtol=3e-6)
            for before, after in zip(snapshots, data):
                self.assertTrue(torch.equal(before, after))

    def test_group_size_is_not_hardcoded_to_four(self):
        for unit in (1, 2, 4, 9):
            data = self.fixture(groups=6, unit=unit)
            actual, support, _ = optimized(*data, scale=0.4, query_chunk_size=5)
            expected, expected_support = explicit(*data)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(support, expected_support, atol=3e-7, rtol=3e-6)

    def test_bool_masks_preserve_full_legal_source_denominator(self):
        data = self.fixture()
        raw = data[0].shape[0]
        sources = torch.ones(raw, dtype=torch.bool)
        sources[3::7] = False
        pairwise = sources[None].expand(raw, -1).clone()
        pairwise[::3, 1::8] = False
        for mask in (sources, pairwise, pairwise[None], pairwise[None].expand(4, -1, -1).clone()):
            actual, support, _ = optimized(*data, scale=0.4, allowed_mask=mask, query_chunk_size=7)
            expected, expected_support = explicit(*data, mask=mask)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(support, expected_support, atol=3e-7, rtol=3e-6)

    def test_near_unit_mass_recomputes_real_deleted_sources(self):
        q = torch.ones((8, 1, 1))
        k = torch.tensor([60.0] * 4 + [-60.0] * 4).reshape(8, 1, 1)
        v = torch.tensor([2.0] * 4 + [-3.0] * 4).reshape(8, 1, 1)
        query, pi = torch.tensor([0]), torch.tensor([1.0, 0.0])
        data = q, k, v, query, pi, 4
        actual, support, details = optimized(*data, scale=1, source_chunk_size=2)
        expected, _ = explicit(*data, scale=1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(details["stable_recomputed_group_ids"], [0])
        self.assertAlmostEqual(float(actual[0]), 25.0, places=5)
        self.assertAlmostEqual(float(support.sum()), 1.0, places=5)

    def test_constant_and_large_common_value_offsets_are_numerically_safe(self):
        q, k, v, query, pi, unit = self.fixture()
        constant = torch.full_like(v, 12345.0)
        actual, _, info = optimized(q, k, constant, query, pi, unit, scale=0.4)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
        self.assertEqual(info["cancellation_recomputed_group_ids"], [])
        shifted = 10000 + 0.1 * v
        actual, _, _ = optimized(q, k, shifted, query, pi, unit, scale=0.4)
        expected, _ = explicit(q, k, shifted, query, pi, unit, dtype=torch.float64)
        torch.testing.assert_close(actual.double(), expected, atol=2e-8, rtol=3e-5)

    def test_cancellation_recovery_uses_direct_vector_error(self):
        data = self.fixture()
        # Raising the numerical-conditioning threshold deliberately exercises
        # the same exact direct-vector recovery on ordinary random inputs.
        actual, support, details = optimized(
            *data, scale=0.4, cancellation_threshold=0.99, group_chunk_size=2, query_chunk_size=7
        )
        expected, expected_support = explicit(*data)
        self.assertGreater(len(details["cancellation_recomputed_group_ids"]), 0)
        torch.testing.assert_close(actual, expected, atol=7e-7, rtol=7e-6)
        torch.testing.assert_close(support, expected_support, atol=2e-7, rtol=3e-6)

    def test_memory_budget_limits_queries_without_changing_scores(self):
        data = self.fixture()
        expected, _ = explicit(*data)
        actual, _, info = optimized(*data, scale=0.4, max_attention_mb=0.002)
        self.assertEqual(info["query_chunk_effective"], 2)
        torch.testing.assert_close(actual, expected, atol=7e-7, rtol=7e-6)
        with self.assertRaisesRegex(ValueError, "even one"):
            optimized(*data, scale=0.4, max_attention_mb=0.00001)

    def test_rejects_invalid_masks_and_chunk_configuration(self):
        data = self.fixture()
        raw = data[0].shape[0]
        with self.assertRaisesRegex(TypeError, "bool"):
            optimized(*data, scale=0.4, allowed_mask=torch.zeros(raw))
        with self.assertRaisesRegex(ValueError, "fully masked"):
            optimized(*data, scale=0.4, allowed_mask=torch.zeros(raw, dtype=torch.bool))
        source = torch.arange(raw) < 4
        with self.assertRaisesRegex(ValueError, "every legal source"):
            optimized(*data, scale=0.4, allowed_mask=source)
        for option in (
            dict(query_chunk_size=0),
            dict(group_chunk_size=0),
            dict(source_chunk_size=0),
            dict(max_attention_mb=0),
            dict(cancellation_threshold=0),
            dict(near_one_threshold=1),
        ):
            with self.assertRaises(ValueError):
                optimized(*data, scale=0.4, **option)

    def test_statistics_preserve_final_greedy_selection(self):
        for seed in range(10):
            data = self.fixture(seed, groups=25)
            rng = torch.Generator().manual_seed(seed + 99)
            text = torch.randn(25, generator=rng).softmax(0)
            query = text.topk(13).indices.sort().values
            pi = torch.zeros_like(text)
            pi[query] = text[query] / text[query].sum()
            data = (*data[:3], query, pi, data[-1])
            expected, _, _ = reference(*data, scale=0.4, query_chunk_size=7)
            actual, _, _ = optimized(*data, scale=0.4, query_chunk_size=7)
            features = torch.randn(25, 17, generator=rng)
            affinity, table, valid = sparse_local_affinity(features, 5, 5, 1)
            expected.index_fill_(0, data[3], 0.0)
            actual.index_fill_(0, data[3], 0.0)
            ref_idx, ref_gain = greedy_direct_cover(text, expected, 13, affinity, table, valid)
            opt_idx, opt_gain = greedy_direct_cover(text, actual, 13, affinity, table, valid)
            self.assertTrue(torch.equal(ref_idx, opt_idx))
            torch.testing.assert_close(ref_gain, opt_gain, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
