"""Offline native Qwen3-VL checks for the independent perturbation adapter."""

from types import SimpleNamespace
import unittest


class PerturbationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            from transformers.models.qwen3_vl import modeling_qwen3_vl
            from models.qwen3 import Qwen3VLPerturbationRunner
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.torch = torch
        cls.config_cls = Qwen3VLVisionConfig
        cls.vision_cls = Qwen3VLVisionModel
        cls.runner_cls = Qwen3VLPerturbationRunner
        cls.native = modeling_qwen3_vl
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "old_threads"):
            cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch = self.torch
        torch.manual_seed(47)
        self.cfg = self.config_cls(
            depth=6,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=24,
            num_heads=2,
            spatial_merge_size=2,
            patch_size=2,
            temporal_patch_size=2,
            num_position_embeddings=64,
            deepstack_visual_indexes=[0, 2, 3, 5],
            _attn_implementation="eager",
        )
        self.vit = self.vision_cls(self.cfg).eval()
        self.grid = torch.tensor([[1, 4, 4], [1, 4, 8]])
        self.pixels = torch.randn(48, 24)
        self.text = torch.randn(6, 24)

    def runner(self, *, layer=3, keep=2, method="q_flowcover_perturbation"):
        torch = self.torch
        runner = object.__new__(self.runner_cls)
        runner.model = SimpleNamespace(config=SimpleNamespace(vision_config=self.cfg))
        runner.vit = self.vit
        runner.projection = self.vit.merger
        runner.device = torch.device("cpu")
        runner.pruning_layer = runner.decision_layer_human = layer
        runner.decision_block_index = layer - 1
        runner.bridge_block_index = layer
        runner.vision_layers = (1, layer, 6)
        runner.visual_keep_ratio = 1.0 if keep is None else 0.5
        runner.visual_keep_tokens = keep
        runner.visual_pruning = True
        runner.pruning_mode = "middle_text_early"
        runner.pruning_strategy = "topk"
        runner.compression_mode = "full19_gather"
        runner.prune_method = method
        runner.perturbation_lambda = 1.0
        runner.perturbation_temperature = 1.0
        runner.perturbation_query_chunk = 3
        runner.perturbation_group_chunk = 2
        runner.perturbation_key_chunk = 5
        runner.perturbation_local_radius = 1
        runner.perturbation_verify_candidates = 2
        runner.perturbation_compare_flow = True
        runner.perturbation_selector_impl = "reference"
        runner.perturbation_max_attention_mb = 16.0
        runner.perturbation_greedy_backend = "reference"
        return runner

    def run_native(self, runner, pixels=None, grid=None):
        helper = self.native.apply_rotary_pos_emb_vision
        with self.torch.no_grad():
            runner._vision_forward_middle_early(
                self.pixels if pixels is None else pixels,
                self.grid if grid is None else grid,
                self.text,
            )
        self.assertIs(helper, self.native.apply_rotary_pos_emb_vision)
        return runner

    def test_generic_layers_actual_native_qkv_once_and_compact_next_block(self):
        for layer in (2, 3, 4):
            with self.subTest(layer=layer):
                calls = [0]

                def count(_module, _args, _output):
                    calls[0] += 1

                handle = self.vit.blocks[layer].attn.qkv.register_forward_hook(count)
                try:
                    runner = self.run_native(self.runner(layer=layer))
                finally:
                    handle.remove()
                self.assertEqual(calls[0], 1)
                self.assertEqual(
                    runner._last_vision_layer_token_counts, [48] * (layer + 1) + [16] * (5 - layer)
                )
                metadata = runner._last_compression_metadata
                self.assertEqual(metadata["score_layer"], layer)
                self.assertEqual(metadata["perturbation_layer"], layer + 1)
                self.assertEqual(metadata["actual_prune_after_layer"], layer + 1)
                self.assertEqual(metadata["qkv_capture_count"], 1)
                self.assertEqual(metadata["qkv_projection_replay_count"], 0)
                self.assertEqual(metadata["actual_gather_count"], 1)
                self.assertEqual([v.shape[0] for v in runner._last_native_final_features], [2, 2])
                for branch in runner._last_native_deepstack:
                    self.assertEqual([v.shape[0] for v in branch], [2, 2])
                check = runner._last_q_flowcover_debug[0]["numerical_verification"]
                self.assertEqual(check["status"], "passed")
                self.assertEqual(check["candidate_count"], 2)
                for result in runner._last_q_flowcover_debug:
                    self.assertEqual(result["query_group_count"], 2)
                    self.assertEqual(result["support"].__len__(), result["N"])
                self.assertIsNotNone(runner._last_stage_timings["perturbation"])
                self.assertIsNotNone(runner._last_stage_timings["greedy"])

    def test_full_budget_is_exact_native_output_and_no_capture(self):
        torch = self.torch
        with torch.no_grad():
            reference, reference_deep = self.vit(self.pixels, grid_thw=self.grid)
        runner = self.run_native(self.runner(keep=None))
        torch.testing.assert_close(
            torch.cat(runner._last_native_final_features), reference, atol=1e-6, rtol=1e-5
        )
        for actual, expected in zip(runner._last_native_deepstack, reference_deep):
            torch.testing.assert_close(torch.cat(actual), expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(runner._last_compression_metadata["qkv_capture_count"], 0)
        self.assertTrue(runner._last_compression_metadata["raw_keep_idx_hash"])
        self.assertEqual(runner._last_vision_layer_token_counts, [48] * 6)
        self.assertTrue(
            all(record["statistics_skipped"] for record in runner._last_q_flowcover_debug)
        )

    def test_captured_rope_qkv_reproduce_native_attention_readout(self):
        """Check against the real attention's projection input, not our formula twice."""
        from models.qwen3 import decode_native_qkv

        torch = self.torch
        attn = self.vit.blocks[3].attn
        seen = {}

        def block_inputs(_module, _args, kwargs):
            seen["positions"] = kwargs["position_embeddings"]

        def qkv_output(_module, _args, output):
            seen["qkv"] = output.detach().clone()

        def projected_input(_module, args):
            seen["native_readout"] = args[0].detach().clone()

        handles = [
            self.vit.blocks[3].register_forward_pre_hook(block_inputs, with_kwargs=True),
            attn.qkv.register_forward_hook(qkv_output),
            attn.proj.register_forward_pre_hook(projected_input),
        ]
        try:
            self.run_native(self.runner())
        finally:
            for handle in handles:
                handle.remove()
        q, k, v = decode_native_qkv(attn, seen["qkv"], seen["positions"])
        contexts = []
        for start, end in ((0, 16), (16, 48)):
            logits = q[start:end].transpose(0, 1).float() @ k[start:end].transpose(
                0, 1
            ).float().transpose(-1, -2)
            context = (
                torch.softmax(logits * attn.scaling, -1) @ v[start:end].transpose(0, 1).float()
            )
            contexts.append(context.transpose(0, 1).reshape(end - start, -1))
        torch.testing.assert_close(
            torch.cat(contexts), seen["native_readout"], atol=2e-6, rtol=2e-5
        )

    def test_view_isolation_and_determinism(self):
        torch = self.torch
        combined = self.run_native(self.runner())
        again = self.run_native(self.runner())
        self.assertEqual(
            combined._last_compression_metadata["raw_keep_idx_hash"],
            again._last_compression_metadata["raw_keep_idx_hash"],
        )
        for view, (start, end) in enumerate(((0, 16), (16, 48))):
            independent = self.run_native(
                self.runner(), self.pixels[start:end], self.grid[view : view + 1]
            )
            torch.testing.assert_close(
                combined._last_native_final_features[view],
                independent._last_native_final_features[0],
                atol=2e-6,
                rtol=1e-5,
            )
            self.assertEqual(
                combined._last_q_flowcover_debug[view]["final_group_index"],
                independent._last_q_flowcover_debug[0]["final_group_index"],
            )
            torch.testing.assert_close(
                torch.tensor(combined._last_q_flowcover_debug[view]["deletion_scores"]),
                torch.tensor(independent._last_q_flowcover_debug[0]["deletion_scores"]),
                atol=2e-7,
                rtol=2e-5,
            )

    def test_verification_skips_an_unmeasured_full_budget_first_view(self):
        grid = self.torch.tensor([[1, 2, 4], [1, 4, 8]])
        pixels = self.torch.randn(40, 24)
        runner = self.run_native(self.runner(keep=2), pixels, grid)
        self.assertFalse(runner._last_q_flowcover_debug[0]["statistics_computed"])
        self.assertEqual(
            runner._last_q_flowcover_debug[1]["numerical_verification"]["status"], "passed"
        )

    def test_rope_indices_and_complete_groups_at_actual_gather(self):
        torch = self.torch
        traces = {}

        def record(index):
            def capture(_module, _args, kwargs):
                traces[index] = (
                    kwargs["cu_seqlens"].clone(),
                    tuple(value.clone() for value in kwargs["position_embeddings"]),
                )

            return capture

        handles = [
            self.vit.blocks[i].register_forward_pre_hook(record(i), with_kwargs=True)
            for i in (0, 4)
        ]
        try:
            runner = self.run_native(self.runner())
        finally:
            for handle in handles:
                handle.remove()
        keep = torch.tensor(runner._last_compression_metadata["raw_keep_idx"])
        for original, gathered in zip(traces[0][1], traces[4][1]):
            torch.testing.assert_close(gathered, original[keep])
        self.assertEqual(traces[4][0].tolist(), [0, 8, 16])
        self.assertEqual(keep.unique().numel(), keep.numel())
        self.assertTrue(bool((keep[1:] > keep[:-1]).all()))
        for group in keep.reshape(-1, 4):
            self.assertEqual(group.tolist(), list(range(int(group[0]), int(group[0]) + 4)))

    def test_no_cover_mode_is_explicit_and_uses_radius_zero(self):
        runner = self.run_native(self.runner(method="q_flowcover_perturbation_nocover"))
        self.assertTrue(
            all(record["local_radius"] == 0 for record in runner._last_q_flowcover_debug)
        )
        runner.prune_method = "unknown"
        with self.assertRaisesRegex(RuntimeError, "explicit new selector"):
            self.run_native(runner)

    def test_native_hooks_are_removed_when_block_fails(self):
        attn_qkv = self.vit.blocks[3].attn.qkv
        before = (len(attn_qkv._forward_pre_hooks), len(attn_qkv._forward_hooks))

        def fail(_module, _args, _output):
            raise RuntimeError("injected native block failure")

        handle = self.vit.blocks[3].attn.proj.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.run_native(self.runner())
        finally:
            handle.remove()
        self.assertEqual(before, (len(attn_qkv._forward_pre_hooks), len(attn_qkv._forward_hooks)))


if __name__ == "__main__":
    unittest.main()
