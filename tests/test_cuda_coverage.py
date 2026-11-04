"""The optional fused backend retains every sequential greedy decision."""

from __future__ import annotations

import importlib.util
import unittest

import torch

from pruning.cuda_coverage import greedy_direct_cover_cuda
from pruning.reference import greedy_direct_cover, sparse_local_affinity


class CudaCoverageAvailabilityTests(unittest.TestCase):
    def test_cpu_inputs_fail_explicitly_instead_of_silently_changing_backend(self):
        p = torch.ones(4, dtype=torch.float32)
        table = torch.arange(4).reshape(4, 1)
        with self.assertRaises((ValueError, RuntimeError)):
            greedy_direct_cover_cuda(
                p, p, 2, p[:, None], table, torch.ones_like(table, dtype=torch.bool)
            )


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton"), "CUDA and Triton required"
)
class CudaCoverageTests(unittest.TestCase):
    def fixture(self, shape, radius, seed):
        torch.manual_seed(seed)
        n = shape[0] * shape[1]
        p = torch.randn(n, device="cuda").softmax(0)
        r = torch.rand(n, device="cuda") / n
        z = torch.randn(n, 24, device="cuda")
        affinity, table, valid = sparse_local_affinity(z, *shape, radius)
        return p, r, affinity, table, valid

    def compare(self, data, keep, shape=None, radius=1):
        p, r, affinity, table, valid = data
        originals = [value.clone() for value in data]
        expected, expected_gains = greedy_direct_cover(p, r, keep, affinity, table, valid)
        actual, gains = greedy_direct_cover_cuda(
            p, r, keep, affinity, table, valid, grid_shape=shape, local_radius=radius
        )
        self.assertTrue(torch.equal(actual, expected))
        torch.testing.assert_close(gains, expected_gains, atol=2e-7, rtol=2e-6)
        for value, original in zip(data, originals):
            self.assertTrue(torch.equal(value, original))
        return actual, gains

    def test_matches_reference_for_boundaries_radii_budgets_and_fresh_inputs(self):
        for shape, radius in (((1, 7), 0), ((3, 5), 1), ((8, 8), 2), ((16, 16), 1), ((32, 32), 1)):
            for seed in (0, 17):
                data = self.fixture(shape, radius, seed)
                n = shape[0] * shape[1]
                for keep in (1, max(1, n // 2), n if n <= 64 else 3):
                    with self.subTest(shape=shape, radius=radius, seed=seed, keep=keep):
                        self.compare(data, keep, shape, radius)
        # Reuse a compiled shape with different sample tensors, then restore
        # the first input. No previous coverage/selected state may leak.
        first = self.fixture((8, 8), 1, 2)
        a, gains = self.compare(first, 23, (8, 8), 1)
        self.compare(self.fixture((8, 8), 1, 5), 23, (8, 8), 1)
        b, repeated = self.compare(first, 23, (8, 8), 1)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(gains, repeated))

    def test_ties_use_smallest_original_group_id(self):
        data = self.fixture((4, 5), 1, 1)
        p = torch.full_like(data[0], 0.05)
        r = torch.zeros_like(p)
        for shape in (None, (4, 5)):
            actual, gains = self.compare((p, r, *data[2:]), 17, shape, 1)
            self.assertTrue(torch.equal(actual, torch.arange(17, device="cuda")))
            self.assertTrue(torch.equal(gains, p[:17]))

    def test_explicit_unsupported_shapes_and_dtypes_raise(self):
        p, r, affinity, table, valid = self.fixture((4, 4), 1, 0)
        with self.assertRaises(ValueError):
            greedy_direct_cover_cuda(p, r, 17, affinity, table, valid)
        with self.assertRaises(ValueError):
            greedy_direct_cover_cuda(p, r, 4, affinity, table, valid, grid_shape=(2, 2))
        with self.assertRaises(TypeError):
            greedy_direct_cover_cuda(p.half(), r, 4, affinity, table, valid)


if __name__ == "__main__":
    unittest.main()
