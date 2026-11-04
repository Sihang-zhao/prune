"""Inline numerical recovery preserves fixed-query deletion scores."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from pruning import statistics
from pruning.reference import greedy_direct_cover, sparse_local_affinity


def _fixture(seed=71, *, groups=12, unit=4, heads=4, kv_heads=2, dim=8):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(groups * unit, heads, dim, generator=generator)
    k = torch.randn(groups * unit, kv_heads, dim, generator=generator)
    v = torch.randn(groups * unit, kv_heads, dim, generator=generator)
    query = torch.arange(0, groups, 2)
    pi = torch.zeros(groups)
    pi[query] = torch.linspace(-1, 1, query.numel()).softmax(0)
    return q, k, v, query, pi, unit


def _explicit(data, *, mask=None, dtype=torch.float32):
    q, k, v, query, pi, unit = data
    heads, groups = q.shape[1], pi.numel()
    ids = (query[:, None] * unit + torch.arange(unit)).flatten()
    weights = pi[query].to(dtype).repeat_interleave(unit) / unit
    keys = k.transpose(0, 1).to(dtype).repeat_interleave(heads // k.shape[1], 0)
    values = v.transpose(0, 1).to(dtype).repeat_interleave(heads // v.shape[1], 0)
    logits = q[ids].transpose(0, 1).to(dtype) @ keys.transpose(-1, -2) * 0.4
    if mask is not None:
        allowed = mask[None, None] if mask.ndim == 1 else mask[ids][None]
        logits.masked_fill_(~allowed, -float("inf"))
    attention = logits.softmax(-1)
    full = attention @ values
    support = (attention.mean(0).reshape(-1, groups, unit).sum(-1) * weights[:, None]).sum(0)
    scores = []
    for group in range(groups):
        remaining = logits.clone()
        remaining[:, :, group * unit : (group + 1) * unit] = -float("inf")
        deleted = remaining.softmax(-1) @ values
        scores.append(((full - deleted).square().sum(-1).mean(0) * weights).sum())
    return torch.stack(scores), support


class InlineStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_forced_inline_cancellation_does_not_repeat_qk_or_full_av(self):
        data = _fixture()
        original = statistics._rectangle
        with patch.object(statistics, "_rectangle", wraps=original) as calls:
            inline, support, info = statistics.group_deletion_statistics_optimized(
                *data,
                scale=0.4,
                query_chunk_size=7,
                cancellation_threshold=0.99,
                cancellation_recovery="inline",
            )
        chunks = math.ceil(data[3].numel() * data[-1] / 7)
        self.assertEqual(calls.call_count, chunks)
        self.assertEqual(info["cancellation_qk_source_scans"], 0)
        self.assertGreater(info["inline_recovered_tuple_count"], 0)
        self.assertTrue(info["cancellation_recomputed_group_ids"])
        with patch.object(statistics, "_rectangle", wraps=original) as calls:
            deferred, old_support, previous = statistics.group_deletion_statistics_optimized(
                *data,
                scale=0.4,
                query_chunk_size=7,
                cancellation_threshold=0.99,
            )
        self.assertEqual(calls.call_count, 2 * chunks)
        self.assertEqual(previous["cancellation_recovery"], "deferred")
        self.assertEqual(previous["cancellation_qk_source_scans"], 1)
        self.assertEqual(info["source_query_count"], previous["source_query_count"])
        self.assertLess(info["attention_flops_estimated"], previous["attention_flops_estimated"])
        torch.testing.assert_close(inline, deferred, atol=7e-7, rtol=7e-6)
        self.assertTrue(torch.equal(support, old_support))

    def test_inline_matches_independent_deletion_for_gqa_masks_and_half_inputs(self):
        for unit, kv_heads, dtype in (
            (1, 4, torch.float32),
            (2, 2, torch.float32),
            (4, 1, torch.float16),
            (4, 2, torch.bfloat16),
            (9, 2, torch.float32),
        ):
            data = _fixture(groups=6, unit=unit, kv_heads=kv_heads)
            data = (*(tensor.to(dtype) for tensor in data[:3]), *data[3:])
            snapshots = [tensor.clone() for tensor in data[:5]]
            raw = data[0].shape[0]
            source_mask = torch.ones(raw, dtype=torch.bool)
            source_mask[3::7] = False
            pair_mask = source_mask[None].expand(raw, -1).clone()
            pair_mask[::3, 1::8] = False
            for mask in (None, source_mask, pair_mask):
                with self.subTest(
                    unit=unit, kv_heads=kv_heads, dtype=dtype, masked=mask is not None
                ):
                    actual, support, _ = statistics.group_deletion_statistics_optimized(
                        *data,
                        scale=0.4,
                        query_chunk_size=5,
                        allowed_mask=mask,
                        cancellation_threshold=0.99,
                        cancellation_recovery="inline",
                    )
                    expected, expected_support = _explicit(data, mask=mask)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(support, expected_support, atol=3e-7, rtol=3e-6)
            for original, after in zip(snapshots, data):
                self.assertTrue(torch.equal(original, after))

    def test_near_unit_mass_still_uses_independent_stable_deletion(self):
        q = torch.ones(8, 1, 1)
        k = torch.tensor([60.0] * 4 + [-60.0] * 4).reshape(8, 1, 1)
        v = torch.tensor([2.0] * 4 + [-3.0] * 4).reshape(8, 1, 1)
        data = q, k, v, torch.tensor([0]), torch.tensor([1.0, 0.0]), 4
        actual, support, info = statistics.group_deletion_statistics_optimized(
            *data,
            scale=1,
            cancellation_recovery="inline",
            source_chunk_size=2,
        )
        self.assertEqual(info["stable_recomputed_group_ids"], [0])
        self.assertAlmostEqual(actual[0].item(), 25.0, places=5)
        self.assertAlmostEqual(support.sum().item(), 1.0, places=5)

    def test_centered_constant_and_large_offset_values_are_safe(self):
        q, k, v, query, pi, unit = _fixture()
        constant = torch.full_like(v, 12345.0)
        actual, _, info = statistics.group_deletion_statistics_optimized(
            q,
            k,
            constant,
            query,
            pi,
            unit,
            scale=0.4,
            cancellation_recovery="inline",
        )
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
        self.assertEqual(info["inline_recovered_tuple_count"], 0)
        shifted = (q, k, 10000 + 0.1 * v, query, pi, unit)
        actual, _, _ = statistics.group_deletion_statistics_optimized(
            *shifted,
            scale=0.4,
            cancellation_threshold=0.99,
            cancellation_recovery="inline",
        )
        expected, _ = _explicit(shifted, dtype=torch.float64)
        torch.testing.assert_close(actual.double(), expected, atol=2e-8, rtol=3e-5)

    def test_recovery_preserves_exact_greedy_groups_and_marginal_scores(self):
        for seed in range(5):
            data = _fixture(seed, groups=25)
            generator = torch.Generator().manual_seed(seed + 99)
            text = torch.randn(25, generator=generator).softmax(0)
            query = text.topk(13).indices.sort().values
            pi = torch.zeros_like(text)
            pi[query] = text[query] / text[query].sum()
            data = (*data[:3], query, pi, data[-1])
            deferred, _, _ = statistics.group_deletion_statistics_optimized(
                *data,
                scale=0.4,
                query_chunk_size=7,
                cancellation_threshold=0.99,
            )
            inline, _, _ = statistics.group_deletion_statistics_optimized(
                *data,
                scale=0.4,
                query_chunk_size=7,
                cancellation_threshold=0.99,
                cancellation_recovery="inline",
            )
            affinity, table, valid = sparse_local_affinity(
                torch.randn(25, 17, generator=generator), 5, 5, 1
            )
            expected = greedy_direct_cover(
                text, deferred.index_fill(0, query, 0), 13, affinity, table, valid
            )
            actual = greedy_direct_cover(
                text, inline.index_fill(0, query, 0), 13, affinity, table, valid
            )
            self.assertTrue(torch.equal(actual[0], expected[0]))
            torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-5)

    def test_unknown_recovery_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cancellation_recovery"):
            statistics.group_deletion_statistics_optimized(
                *_fixture(), scale=0.4, cancellation_recovery="bad"
            )


if __name__ == "__main__":
    unittest.main()
