"""Acceleration is explicit, resumable settings stay isolated, native math stays aligned."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import evaluate
from evaluation.runner import _validate_resume_config


class AccelerationIntegrationTests(unittest.TestCase):
    def args(self, *extra):
        return evaluate.get_args(["--data", "cases.jsonl", "--model", "fake/model", *extra])

    def test_acceleration_is_opt_in_and_rejects_unrelated_paths(self):
        ordinary = self.args()
        self.assertEqual(
            (ordinary.greedy_backend, ordinary.cancellation_recovery), ("reference", "deferred")
        )
        fast = self.args("--greedy-backend", "cuda_fused", "--cancellation-recovery", "inline")
        self.assertEqual(
            (fast.greedy_backend, fast.cancellation_recovery), ("cuda_fused", "inline")
        )
        for extra in (
            ("--implementation", "reference", "--cancellation-recovery", "inline"),
            ("--implementation", "reference", "--greedy-backend", "cuda_fused"),
            ("--method", "base", "--cancellation-recovery", "inline"),
            ("--greedy-backend", "cuda_fused", "--local-radius", "3"),
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_old_checkpoints_resume_defaults_but_not_new_timing_modes(self):
        ordinary = self.args()
        previous = vars(ordinary).copy()
        previous.pop("cancellation_recovery")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "results.jsonl"
            source.with_name("config.json").write_text(json.dumps(previous))
            _validate_resume_config(source, ordinary)
            for option in (
                ("--cancellation-recovery", "inline"),
                ("--greedy-backend", "cuda_fused"),
            ):
                with self.subTest(option=option), self.assertRaisesRegex(
                    ValueError, option[0][2:].replace("-", "_")
                ):
                    _validate_resume_config(source, self.args(*option))

    def test_inline_native_model_keeps_original_groups_and_all_queries(self):
        import torch
        from test_native_model import PerturbationRuntimeTests

        fixture = PerturbationRuntimeTests()
        fixture.setUpClass()
        try:
            fixture.setUp()
            expected = fixture.runner()
            expected.perturbation_selector_impl = "optimized"
            expected.perturbation_greedy_backend = "reference"
            expected.perturbation_max_attention_mb = 16.0
            fixture.run_native(expected)
            actual = fixture.runner()
            actual.perturbation_selector_impl = "optimized"
            actual.perturbation_greedy_backend = "reference"
            actual.perturbation_max_attention_mb = 16.0
            actual.perturbation_cancellation_recovery = "inline"
            fixture.run_native(actual)
            self.assertEqual(
                expected._last_compression_metadata["raw_keep_idx_hash"],
                actual._last_compression_metadata["raw_keep_idx_hash"],
            )
            for view in actual._last_q_flowcover_debug:
                self.assertEqual(view["query_group_count"], view["K"])
                self.assertEqual(view["source_query_count"], view["K"] * view["merge_unit"])
                self.assertEqual(view["cancellation_recovery"], "inline")
            for a, b in zip(
                expected._last_native_final_features, actual._last_native_final_features
            ):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        finally:
            fixture.tearDownClass()


if __name__ == "__main__":
    unittest.main()
