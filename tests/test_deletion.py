"""Small explicit-deletion references for the independent new selector.

Only these tiny references materialize dense attention/affinity.  The online
selector uses source/query/group chunks and sparse local neighborhoods.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from pruning import reference as subject


class PerturbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(507)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.generator)

    def fixture(self, groups=6, unit=4, heads=4, kv_heads=2, dim=5):
        q = self.randn(groups * unit, heads, dim)
        k = self.randn(groups * unit, kv_heads, dim)
        v = self.randn(groups * unit, kv_heads, dim)
        query = torch.tensor([0, 2, groups - 1])
        pi = torch.zeros(groups)
        pi[query] = torch.tensor([0.2, 0.3, 0.5])
        return q, k, v, query, pi, unit

    def explicit(self, q, k, v, query, pi, unit, *, scale=0.4, mask=None):
        heads = int(q.shape[1])
        repeats = heads // int(k.shape[1])
        raw, groups = int(q.shape[0]), int(pi.numel())
        selected_raw = (query[:, None] * unit + torch.arange(unit)).flatten()
        weights = pi[query].repeat_interleave(unit) / unit
        query_tensor = q.float()[selected_raw].transpose(0, 1)
        keys = k.float().transpose(0, 1).repeat_interleave(repeats, dim=0)
        values = v.float().transpose(0, 1).repeat_interleave(repeats, dim=0)
        logits = torch.bmm(query_tensor, keys.transpose(1, 2)) * scale
        if mask is not None:
            if mask.ndim == 1:
                allowed = mask[None, None]
            elif mask.ndim == 2:
                allowed = mask[selected_raw][None]
            else:
                allowed = mask[:, selected_raw]
            logits.masked_fill_(~allowed, -float("inf"))
        attention = logits.softmax(-1)
        output = attention @ values
        support = (attention.mean(0).reshape(-1, groups, unit).sum(-1) * weights[:, None]).sum(0)
        scores = []
        for group in range(groups):
            deleted_logits = logits.clone()
            deleted_logits[:, :, group * unit : (group + 1) * unit] = -float("inf")
            deleted_output = deleted_logits.softmax(-1) @ values
            squared = (output - deleted_output).square().sum(-1).mean(0)
            scores.append((squared * weights).sum())
        return torch.stack(scores), support

    def test_raw_query_closed_form_matches_real_group_deletion(self):
        data = self.fixture()
        expected, support = self.explicit(*data)
        max_errors = []
        for query_chunk, group_chunk, source_chunk in (
            (1, 1, 1),
            (3, 2, 5),
            (7, 4, 13),
            (256, 32, 256),
        ):
            actual, actual_support, debug = subject.group_deletion_statistics(
                *data,
                scale=0.4,
                query_chunk_size=query_chunk,
                group_chunk_size=group_chunk,
                source_chunk_size=source_chunk,
            )
            torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-6)
            torch.testing.assert_close(actual_support, support, atol=2e-7, rtol=3e-6)
            self.assertEqual(debug["source_query_count"], 12)
            self.assertEqual(debug["kv_repeat_count"], 2)
            self.assertFalse(debug["full_attention_matrix_saved"])
            max_errors.append(float((actual - expected).abs().max()))
        self.assertLess(max(max_errors), 5e-7)

    def test_query_averaging_before_renormalization_is_not_the_deletion_error(self):
        # This catches the tempting but incorrect ratio of group-averaged
        # m/b/o.  A denominator applies to each raw query independently.
        q, k, v, query, pi, unit = self.fixture()
        expected, _ = self.explicit(q, k, v, query, pi, unit)
        raw_query = (query[:, None] * unit + torch.arange(unit)).flatten()
        key = k.transpose(0, 1).repeat_interleave(2, dim=0)
        value = v.transpose(0, 1).repeat_interleave(2, dim=0)
        att = (q[raw_query].transpose(0, 1) @ key.transpose(1, 2) * 0.4).softmax(-1)
        output = (att @ value).reshape(4, 3, unit, 5).mean(2)
        bad = []
        for g in range(6):
            local = att[:, :, g * unit : (g + 1) * unit]
            mass = local.sum(-1).reshape(4, 3, unit).mean(2)
            source = (local @ value[:, g * unit : (g + 1) * unit]).reshape(4, 3, unit, 5).mean(2)
            error = (source - mass[..., None] * output) / (1 - mass)[..., None]
            bad.append((error.square().sum(-1).mean(0) * pi[query]).sum())
        self.assertGreater(float((torch.stack(bad) - expected).abs().max()), 1e-3)

    def test_mask_scale_gqa_and_half_precision_match_explicit_fp32_statistics(self):
        q, k, v, query, pi, unit = self.fixture()
        mask = torch.ones(q.shape[0], dtype=torch.bool)
        mask[3::7] = False
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for scale in (0.2, 0.8):
                local = q.to(dtype), k.to(dtype), v.to(dtype), query, pi, unit
                actual, support, _ = subject.group_deletion_statistics(
                    *local,
                    scale=scale,
                    allowed_mask=mask,
                    query_chunk_size=3,
                    group_chunk_size=2,
                    source_chunk_size=5,
                )
                expected, expected_support = self.explicit(*local, scale=scale, mask=mask)
                torch.testing.assert_close(actual, expected, atol=8e-7, rtol=1e-5)
                torch.testing.assert_close(support, expected_support, atol=3e-7, rtol=3e-6)

    def test_pairwise_and_headwise_bool_masks(self):
        data = self.fixture()
        raw = data[0].shape[0]
        pairwise = torch.ones((raw, raw), dtype=torch.bool)
        pairwise[:, 2::8] = False
        pairwise[::3, 1::8] = False
        for mask in (pairwise, pairwise[None], pairwise[None].expand(4, -1, -1).clone()):
            actual, support, _ = subject.group_deletion_statistics(
                *data, scale=0.4, allowed_mask=mask, source_chunk_size=3
            )
            expected, expected_support = self.explicit(*data, mask=mask)
            torch.testing.assert_close(actual, expected, atol=7e-7, rtol=7e-6)
            torch.testing.assert_close(support, expected_support, atol=3e-7, rtol=3e-6)

    def test_near_unit_mass_uses_true_masked_recompute_without_epsilon_inflation(self):
        q = torch.ones((8, 1, 1))
        k = torch.tensor([60.0] * 4 + [-60.0] * 4).reshape(8, 1, 1)
        v = torch.tensor([2.0] * 4 + [-3.0] * 4).reshape(8, 1, 1)
        query, pi = torch.tensor([0]), torch.tensor([1.0, 0.0])
        actual, support, debug = subject.group_deletion_statistics(
            q, k, v, query, pi, 4, scale=1, source_chunk_size=2, group_chunk_size=1
        )
        expected, _ = self.explicit(q, k, v, query, pi, 4, scale=1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(debug["stable_recomputed_group_ids"], [0])
        self.assertAlmostEqual(float(actual[0]), 25.0, places=5)
        self.assertAlmostEqual(float(support.sum()), 1.0, places=5)

    def test_rejects_float_masks_fully_masked_queries_and_invalid_inputs(self):
        data = self.fixture()
        raw = data[0].shape[0]
        with self.assertRaisesRegex(TypeError, "bool"):
            subject.group_deletion_statistics(*data, scale=0.4, allowed_mask=torch.zeros(raw))
        with self.assertRaisesRegex(ValueError, "fully masked"):
            subject.group_deletion_statistics(
                *data, scale=0.4, allowed_mask=torch.zeros(raw, dtype=torch.bool)
            )
        # A mask allowing exactly one group has no legal deletion for it.
        source = torch.arange(raw) < 4
        with self.assertRaisesRegex(ValueError, "every legal source"):
            subject.group_deletion_statistics(*data, scale=0.4, allowed_mask=source)
        for kwargs in (
            {"query_chunk_size": 0},
            {"group_chunk_size": 0},
            {"source_chunk_size": 0},
            {"scale": -1},
        ):
            with self.assertRaises(ValueError):
                subject.group_deletion_statistics(*data, **{"scale": 0.4, **kwargs})
        bad = list(data)
        bad[0] = bad[0].clone()
        bad[0][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            subject.group_deletion_statistics(*bad, scale=0.4)

    def dense_affinity(self, features, height, width, radius):
        z = F.normalize(features.float(), dim=-1)
        similarity = (z @ z.T).clamp_min(0)
        for a in range(height * width):
            for b in range(height * width):
                if max(abs(a // width - b // width), abs(a % width - b % width)) > radius:
                    similarity[a, b] = 0
        similarity.fill_diagonal_(1)
        return similarity

    def dense_greedy(self, p, omega, keep, similarity):
        selected = []
        coverage = torch.zeros_like(p)
        gains = []
        for _ in range(keep):
            gain = p + (omega[:, None] * (similarity - coverage[:, None]).clamp_min(0)).sum(0)
            gain[selected] = -float("inf")
            best = int(gain.argmax())
            selected.append(best)
            gains.append(gain[best])
            coverage = torch.maximum(coverage, similarity[:, best])
        return torch.tensor(sorted(selected)), torch.stack(gains)

    def test_sparse_greedy_matches_dense_every_round_and_ties(self):
        for height, width, radius in ((1, 1, 0), (1, 5, 0), (2, 3, 1), (3, 3, 2)):
            count = height * width
            z = self.randn(count, 13)
            p = self.randn(count).softmax(0)
            residual = self.randn(count).square()
            affinity, table, valid = subject.sparse_local_affinity(z, height, width, radius)
            dense = self.dense_affinity(z, height, width, radius)
            for keep in (1, count):
                actual, gain = subject.greedy_direct_cover(
                    p, residual, keep, affinity, table, valid
                )
                expected, expected_gain = self.dense_greedy(p, residual, keep, dense)
                self.assertTrue(torch.equal(actual, expected))
                torch.testing.assert_close(gain, expected_gain, atol=2e-6, rtol=2e-6)
        p = torch.ones(6)
        affinity, table, valid = subject.sparse_local_affinity(torch.ones(6, 3), 2, 3, 0)
        actual, _ = subject.greedy_direct_cover(p, torch.zeros_like(p), 3, affinity, table, valid)
        self.assertEqual(actual.tolist(), [0, 1, 2])

    def view_data(self):
        first, second = self.fixture(), self.fixture(groups=8)
        return {
            "projected_views": [self.randn(6, 12), self.randn(8, 12)],
            "text_tokens": self.randn(5, 12),
            "q_views": [first[0], second[0]],
            "k_views": [first[1], second[1]],
            "v_views": [first[2], second[2]],
            "hidden_views": [self.randn(24, 7), self.randn(32, 7)],
            "keep_counts": [3, 4],
            "raw_counts": [24, 32],
            "merge_unit": 4,
            "grid_shapes": [(2, 3), (2, 4)],
        }

    def test_public_selector_dense_objective_budget_diagnostics_and_nocover(self):
        data = self.view_data()
        for method, radius in (
            ("q_flowcover_perturbation", 1),
            ("q_flowcover_perturbation_nocover", 0),
        ):
            results, records = subject.select_q_flowcover_perturbation_views(
                **data, method=method, scale=0.4, compare_flow=True
            )
            for view, (result, record) in enumerate(zip(results, records)):
                p = torch.tensor(record["text_mass"])
                query = torch.tensor(record["query_group_index"])
                pi = torch.tensor(record["pi"])
                score, support = self.explicit(
                    data["q_views"][view],
                    data["k_views"][view],
                    data["v_views"][view],
                    query,
                    pi,
                    4,
                )
                torch.testing.assert_close(
                    torch.tensor(record["deletion_scores"]), score, atol=7e-7, rtol=7e-6
                )
                torch.testing.assert_close(
                    torch.tensor(record["support"]), support, atol=3e-7, rtol=3e-6
                )
                residual = score.index_fill(0, query, 0)
                self.assertEqual(
                    torch.tensor(record["omega"])[query].tolist(), [0.0] * query.numel()
                )
                features = data["hidden_views"][view].reshape(p.numel(), -1)
                dense = self.dense_affinity(features, *data["grid_shapes"][view], radius)
                expected, gain = self.dense_greedy(p, residual, data["keep_counts"][view], dense)
                self.assertTrue(torch.equal(result.indices, expected))
                torch.testing.assert_close(
                    torch.tensor(record["greedy_gains"]), gain, atol=2e-6, rtol=5e-6
                )
                self.assertEqual(result.indices.numel(), record["K"])
                self.assertEqual(result.indices.unique().numel(), record["K"])
                self.assertEqual(record["final_group_index"], expected.tolist())
                self.assertIn("flow_group_index", record)
                self.assertGreater(record["selector_seconds"], 0)
                if radius == 0:
                    self.assertTrue(
                        torch.equal(
                            expected, (p + residual).topk(record["K"]).indices.sort().values
                        )
                    )

    def test_lambda_zero_full_budget_and_per_view_isolation(self):
        data = self.view_data()
        results, records = subject.select_q_flowcover_perturbation_views(
            **data, method="q_flowcover_perturbation", scale=0.4, perturbation_lambda=0
        )
        for result, record in zip(results, records):
            self.assertEqual(result.indices.tolist(), record["text_topk"])
        combined, _ = subject.select_q_flowcover_perturbation_views(
            **data, method="q_flowcover_perturbation", scale=0.4
        )
        for view in (0, 1):
            local = {
                key: value if key in ("text_tokens", "merge_unit") else [value[view]]
                for key, value in data.items()
            }
            single, _ = subject.select_q_flowcover_perturbation_views(
                **local, method="q_flowcover_perturbation", scale=0.4
            )
            self.assertTrue(torch.equal(single[0].indices, combined[view].indices))
        data["keep_counts"] = [6, 8]
        full, full_records = subject.select_q_flowcover_perturbation_views(
            **data, method="q_flowcover_perturbation", scale=0.4
        )
        for result, record, count in zip(full, full_records, (6, 8)):
            self.assertEqual(result.indices.tolist(), list(range(count)))
            self.assertEqual(record["source_query_count"], 0)
            self.assertIn("statistics_skipped_reason", record)
            self.assertFalse(record["statistics_computed"])
            self.assertEqual(record["support"], [])
            self.assertEqual(record["deletion_scores"], [])
            self.assertEqual(len(record["keep_idx_hash"]), 64)
        data["q_views"] = data["k_views"] = data["v_views"] = [None, None]
        skipped, _ = subject.select_q_flowcover_perturbation_views(
            **data, method="q_flowcover_perturbation", scale=0.4
        )
        self.assertEqual(skipped[0].indices.tolist(), list(range(6)))

    def test_invalid_method_and_view_collections_raise_without_fallback(self):
        data = self.view_data()
        with self.assertRaisesRegex(ValueError, "no fallback"):
            subject.select_q_flowcover_perturbation_views(**data, method="q_flowcover", scale=0.4)
        for key in ("v_views", "raw_counts", "hidden_views", "grid_shapes"):
            invalid = {**data, key: data[key][:1]}
            with self.assertRaisesRegex(ValueError, "identical"):
                subject.select_q_flowcover_perturbation_views(
                    **invalid, method="q_flowcover_perturbation", scale=0.4
                )


if __name__ == "__main__":
    unittest.main()
