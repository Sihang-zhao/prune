"""Dense CPU references for the historical q_flowcover_direct_residual path.

The references deliberately materialize tiny attention and affinity matrices;
production inference keeps attention chunks and a sparse neighborhood graph.
No checkpoint, dataset or historical source directory is required.
"""
from __future__ import annotations

import unittest


class DirectResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"PyTorch unavailable: {exc}") from exc
        from compression import q_flowcover, q_flowcover_variants

        cls.torch = torch
        cls.shared = q_flowcover
        cls.selector = q_flowcover_variants
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'previous_threads'):
            cls.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.generator = self.torch.Generator().manual_seed(109)

    def _randn(self, *shape):
        return self.torch.randn(*shape, generator=self.generator)

    def _fixture(self):
        return {
            'projected': [self._randn(6, 12), self._randn(8, 12)],
            'text': self._randn(5, 12),
            'q': [self._randn(24, 2, 4), self._randn(32, 2, 4)],
            'k': [self._randn(24, 2, 4), self._randn(32, 2, 4)],
            'hidden': [self._randn(24, 5), self._randn(32, 5)],
            'keep': [3, 5], 'raw': [24, 32], 'grid': [(2, 3), (2, 4)],
        }

    def _select(self, data, **kwargs):
        return self.selector.select_q_flowcover_variant_views(
            data['projected'], data['text'], data['q'], data['k'], data['hidden'],
            data['keep'], data['raw'], 4, 'q_flowcover_direct_residual', data['grid'],
            scale=0.5, **kwargs,
        )

    def _dense_support(self, q, k, query_groups, pi, unit=4):
        """Native attention dtype, mean and group aggregation contract."""
        torch = self.torch
        raw_count, heads, _ = q.shape
        groups = raw_count // unit
        logits = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2)) * 0.5
        attention = logits.float().softmax(-1).to(q.dtype).mean(0)
        grouped_keys = attention.reshape(raw_count, groups, unit).sum(-1)
        raw_queries = (query_groups[:, None] * unit + torch.arange(unit)).flatten()
        raw_weights = pi[query_groups].repeat_interleave(unit) / unit
        support = (grouped_keys[raw_queries] * raw_weights[:, None]).sum(0)
        if q.dtype != torch.float32:
            support = support / support.sum().clamp_min(torch.finfo(support.dtype).tiny)
        return support

    def _dense_affinity(self, features, height, width, radius, mode='raw'):
        """Rows are source evidence; columns are candidate kept groups."""
        torch = self.torch
        groups = height * width
        cosine = (features[:, None] * features[None, :]).sum(-1)
        neighbors = torch.zeros((groups, groups), dtype=torch.bool)
        for source in range(groups):
            sy, sx = divmod(source, width)
            for candidate in range(groups):
                cy, cx = divmod(candidate, width)
                neighbors[source, candidate] = max(abs(sy - cy), abs(sx - cx)) <= radius
        if mode == 'raw':
            return cosine.clamp_min(0).masked_fill(~neighbors, 0)
        thresholds = torch.zeros(groups)
        for source in range(groups):
            nonself = neighbors[source].clone()
            nonself[source] = False
            if bool(nonself.any()):
                thresholds[source] = cosine[source, nonself].median()
        affinity = ((cosine - thresholds[:, None]) / (1 - thresholds[:, None] + 1e-6)).clamp_min(0)
        affinity.fill_diagonal_(1)
        return affinity.masked_fill(~neighbors, 0)

    def _dense_greedy(self, affinity, text_mass, residual, keep):
        torch = self.torch
        coverage = torch.zeros_like(residual)
        chosen = []
        for _ in range(keep):
            gains = text_mass + (residual[:, None] * (affinity - coverage[:, None]).clamp_min(0)).sum(0)
            gains[chosen] = -float('inf')
            candidate = int(gains.argmax())
            chosen.append(candidate)
            coverage = torch.maximum(coverage, affinity[:, candidate])
        return torch.tensor(sorted(chosen))

    def _reference(self, data, view, radius=1, mode='raw'):
        torch = self.torch
        groups = data['projected'][view].shape[0]
        unit = 4
        question = torch.nn.functional.normalize(data['text'].float().mean(0), dim=0)
        scores = torch.nn.functional.cosine_similarity(data['projected'][view].float(), question[None], dim=-1)
        text_mass = scores.softmax(0)
        query = torch.topk(scores, data['keep'][view], sorted=False).indices.sort().values
        pi = torch.zeros_like(text_mass)
        pi[query] = text_mass[query] / text_mass[query].sum().clamp_min(1e-12)
        support = self._dense_support(data['q'][view], data['k'][view], query, pi)
        residual = support.clone()
        residual[query] = 0
        features = torch.nn.functional.normalize(data['hidden'][view].reshape(groups, -1).float(), dim=-1)
        affinity = self._dense_affinity(features, *data['grid'][view], radius, mode)
        selected = self._dense_greedy(affinity, text_mass, residual, data['keep'][view])
        return selected, query, text_mass, pi, support, residual

    def test_public_selector_matches_dense_attention_and_direct_cover_reference(self):
        torch = self.torch
        data = self._fixture()
        for mode in ('raw', 'local_centered'):
            for chunk in (1, 3, 64):
                with self.subTest(mode=mode, chunk=chunk):
                    result, debug = self._select(data, query_chunk_size=chunk, affinity_mode=mode)
                    for view, (selected, record) in enumerate(zip(result, debug)):
                        expected, query, text, pi, support, residual = self._reference(data, view, mode=mode)
                        self.assertTrue(torch.equal(selected.indices, expected))
                        self.assertEqual(record['query_group_index'], query.tolist())
                        for key, reference in [('text_mass', text), ('pi', pi), ('support', support), ('residual_support', residual)]:
                            torch.testing.assert_close(torch.tensor(record[key]), reference, atol=2e-7, rtol=2e-6)

    def test_query_and_residual_probability_contract(self):
        torch = self.torch
        data = self._fixture()
        results, debug = self._select(data)
        for result, record, keep, count in zip(results, debug, data['keep'], (6, 8)):
            query = torch.tensor(record['query_group_index'])
            self.assertEqual(query.numel(), keep)
            self.assertEqual(record['text_topk'], query.tolist())
            self.assertEqual(result.indices.tolist(), sorted(set(result.indices.tolist())))
            self.assertEqual(result.indices.numel(), keep)
            self.assertEqual(result.budget, (0, keep, 0))
            self.assertTrue(torch.equal(result.middle_indices, result.indices))
            self.assertEqual(result.final_indices.numel(), 0)
            self.assertEqual(result.shallow_indices.numel(), 0)
            for key in ('text_mass', 'pi', 'support'):
                self.assertAlmostEqual(sum(record[key]), 1.0, places=6)
            outside = torch.ones(count, dtype=torch.bool)
            outside[query] = False
            pi, residual, support = map(torch.tensor, (record['pi'], record['residual_support'], record['support']))
            self.assertTrue(bool((pi[outside] == 0).all()))
            self.assertTrue(bool((residual[query] == 0).all()))
            torch.testing.assert_close(residual[outside], support[outside])
            self.assertEqual(record['omega'], record['residual_support'])
            self.assertEqual(record['source_query_count'], keep * 4)

    def test_sparse_direct_greedy_matches_dense_reference_at_padded_boundaries(self):
        torch = self.torch
        for height, width, radius in ((1, 4, 1), (2, 3, 1), (3, 3, 1), (2, 4, 2), (1, 1, 0)):
            groups = height * width
            features = torch.nn.functional.normalize(self._randn(groups, 9), dim=-1)
            text_mass = self._randn(groups).softmax(0)
            residual = self._randn(groups).softmax(0)
            for mode in ('raw', 'local_centered'):
                affinity = self._dense_affinity(features, height, width, radius, mode)
                for keep in range(1, groups + 1):
                    with self.subTest(grid=(height, width), radius=radius, mode=mode, keep=keep):
                        expected = self._dense_greedy(affinity, text_mass, residual, keep)
                        actual, _ = self.selector._direct_cover(features, text_mass, residual, keep, height, width, radius, mode)
                        self.assertTrue(torch.equal(actual, expected))

    def test_no_residual_keeps_direct_text_topk_instead_of_spreading_text_by_coverage(self):
        torch = self.torch
        features = torch.nn.functional.normalize(self._randn(9, 7), dim=-1)
        text_mass = torch.tensor([0.31, 0.05, 0.21, 0.01, 0.16, 0.08, 0.06, 0.07, 0.05])
        for mode in ('raw', 'local_centered'):
            actual, _ = self.selector._direct_cover(features, text_mass, torch.zeros(9), 3, 3, 3, 1, mode)
            self.assertEqual(actual.tolist(), [0, 2, 4])

    def test_radius_zero_is_direct_text_plus_residual_topk(self):
        torch = self.torch
        data = self._fixture()
        results, debug = self._select(data, local_radius=0)
        for selected, record, keep in zip(results, debug, data['keep']):
            weights = torch.tensor(record['text_mass']) + torch.tensor(record['residual_support'])
            expected = weights.topk(keep, sorted=False).indices.sort().values
            self.assertTrue(torch.equal(selected.indices, expected))

    def test_keep_all_returns_all_middle_groups_and_zeros_residual(self):
        torch = self.torch
        data = self._fixture()
        data['keep'] = [6, 8]
        result, debug = self._select(data)
        for selected, record, count in zip(result, debug, data['keep']):
            self.assertTrue(torch.equal(selected.indices, torch.arange(count)))
            self.assertEqual(selected.budget, (0, count, 0))
            self.assertEqual(record['source_query_count'], 4 * count)
            self.assertEqual(record['residual_support'], [0.0] * count)
            torch.testing.assert_close(torch.tensor(record['pi']), torch.tensor(record['text_mass']))

    def test_query_mass_and_adaptive_cap_do_not_change_this_method(self):
        torch = self.torch
        data = self._fixture()
        base, records = self._select(data, query_mass=0.9, adaptive_radius_max=3)
        for mass, cap in ((0.01, 1), (1.0, 9)):
            actual, changed = self._select(data, query_mass=mass, adaptive_radius_max=cap)
            for first, second, old, new in zip(base, actual, records, changed):
                self.assertTrue(torch.equal(first.indices, second.indices))
                self.assertEqual(old['query_group_index'], new['query_group_index'])
                self.assertEqual(old['local_radius'], new['local_radius'])
                self.assertEqual(old['support'], new['support'])

    def test_image_views_are_independent(self):
        torch = self.torch
        data = self._fixture()
        combined, _ = self._select(data)
        for view in (0, 1):
            single = {key: value if key == 'text' else [value[view]] for key, value in data.items()}
            actual, _ = self._select(single)
            self.assertTrue(torch.equal(combined[view].indices, actual[0].indices))
        changed = dict(data)
        for key in ('projected', 'q', 'k', 'hidden'):
            changed[key] = [data[key][0], self._randn(*data[key][1].shape)]
        altered, _ = self._select(changed)
        self.assertTrue(torch.equal(combined[0].indices, altered[0].indices))

    def test_half_precision_support_preserves_native_probability_cast_and_renormalization(self):
        torch = self.torch
        raw_q = self._randn(24, 2, 4) * 2
        raw_k = self._randn(24, 2, 4) * 2
        query = torch.tensor([0, 2, 5])
        pi = torch.tensor([0.2, 0, 0.3, 0, 0, 0.5])
        for dtype in (torch.float16, torch.bfloat16):
            q, k = raw_q.to(dtype), raw_k.to(dtype)
            reference = self._dense_support(q, k, query, pi)
            float_reference = self._dense_support(q.float(), k.float(), query, pi)
            with self.subTest(dtype=dtype):
                self.assertGreater(float((reference - float_reference).abs().max()), 1e-5)
                for chunk in (1, 3, 64):
                    actual, count = self.shared._group_source_support(q, k, query, pi, 4, scale=0.5, query_chunk_size=chunk)
                    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
                    self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)
                    self.assertEqual(count, 12)

    def test_invalid_budget_grid_radius_or_affinity_is_rejected(self):
        data = self._fixture()
        for key, value in (('keep', [0, 5]), ('keep', [7, 5]), ('grid', [(3, 3), (2, 4)]), ('q', [data['q'][0]])):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self._select({**data, key: value})
        for kwargs in ({'local_radius': -1}, {'affinity_mode': 'bad'}, {'query_mass': 0}, {'adaptive_radius_max': 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self._select(data, **kwargs)


if __name__ == '__main__':
    unittest.main()
