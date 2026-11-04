"""Native baseline contracts without model downloads or GPU inference."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import evaluator
from compression import base_worker
from model.qwen_vl import QwenVL
from utils.data import Case, case_dict


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_native_baseline_entry_under_test", ROOT / "test_base.py")
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)


def config():
    return {
        "vision_config": {"depth": 3, "hidden_size": 4, "intermediate_size": 8,
                          "out_hidden_size": 6, "spatial_merge_size": 2,
                          "deepstack_visual_indexes": [0, 2]},
        "text_config": {"num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 16},
    }


class BaseEntryTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "test_base.py"), "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--warmup-cases", result.stdout)
        self.assertIn("unpruned Qwen3-VL", result.stdout)

    def test_native_baseline_defaults_match_current_evaluation_budget(self):
        args = _entry.get_args(["--data", "sample.jsonl"])
        self.assertEqual(args.evaluation_variant, "base")
        self.assertEqual(args.prune_method, "qwen3_base")
        self.assertFalse(args.visual_pruning)
        self.assertEqual(args.compression_mode, "off")
        self.assertEqual(args.visual_keep_ratio, 1.0)
        self.assertEqual(args.visual_prune_percent, 0)
        self.assertIsNone(args.visual_keep_tokens)
        self.assertEqual(args.max_image_pixels, 4194304)
        self.assertEqual(args.max_image_patches, 1)
        self.assertEqual(args.max_new_tokens, 32)
        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.worker_cases, 0)
        self.assertEqual(args.warmup_cases, 1)
        self.assertIsNone(args.limit)
        with patch.object(evaluator, "test", return_value={"total": 3080}) as run:
            self.assertEqual(_entry.main(["--data", "sample.jsonl"]), {"total": 3080})
        self.assertFalse(run.call_args.args[0].visual_pruning)

    def test_baseline_rejects_nontrivial_pruning_or_negative_warmup(self):
        for extra in (("--visual-prune-percent", "50"), ("--visual-keep-ratio", "0.5"),
                      ("--visual-keep-tokens", "2"), ("--warmup-cases", "-1")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    _entry.get_args(["--data", "sample.jsonl", *extra])
                self.assertEqual(error.exception.code, 2)

    def test_error_cases_make_the_entry_exit_unsuccessfully(self):
        with patch.object(evaluator, "test", return_value={"total": 3080, "errors": 1}):
            with self.assertRaisesRegex(RuntimeError, "1 error cases"):
                _entry.main(["--data", "sample.jsonl"])


class BaseAccountingTests(unittest.TestCase):
    def test_raw_patch_attention_and_native_mergers_use_current_convention(self):
        values = base_worker.current_flops_breakdown(config(), 10, 2, [[1, 2, 2], [1, 2, 4]])
        # Two independent image sequences have 4 and 8 raw patch rows.
        # This independent arithmetic must not use 3 merged groups or mix the
        # two views into one attention sequence of length 12.
        self.assertEqual(values["vision_linear_flops"], 3 * 12 * 2 * (4 * 4**2 + 2 * 4 * 8))
        self.assertEqual(values["vision_attention_flops"], 4 * 4 * 3 * (4**2 + 8**2))
        merger_row = 2 * 16 * (16 + 6)
        self.assertEqual(values["final_merger_flops"], 3 * merger_row)
        self.assertEqual(values["deepstack_mergers_flops"], 2 * 3 * merger_row)
        self.assertEqual(values["score_merger_flops"], 0)
        self.assertEqual(values["selector_flops"], 0)
        self.assertEqual(values["verification_flops"], 0)
        self.assertEqual(values["llm_prefill_flops"], 23680)
        self.assertEqual(values["llm_decode_flops"], 5376)
        self.assertEqual(values["estimated_total_flops"], 48448)

    def test_grid_capture_only_wraps_plain_native_processor_inputs(self):
        self.assertIs(base_worker.BaseQwenVL.__bases__[0], QwenVL)
        runner = object.__new__(base_worker.BaseQwenVL)
        inputs = {"image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]), "input_ids": torch.tensor([[1, 2]])}
        with patch.object(QwenVL, "_inputs", return_value=inputs) as prepare:
            result = runner._inputs("image", "prompt")
        self.assertIs(result, inputs)
        self.assertEqual(runner.last_image_grid_thw, [[1, 2, 2], [1, 2, 4]])
        prepare.assert_called_once_with("image", "prompt")
        self.assertIs(base_worker.BaseQwenVL.answer, QwenVL.answer)
        self.assertIs(base_worker.BaseQwenVL._pruned_inputs, QwenVL._pruned_inputs)


class BaseWorkerTests(unittest.TestCase):
    def test_persistent_native_model_warmup_and_rows_without_selector(self):
        calls = []
        loaded = []

        class Runner:
            def __init__(self, **kwargs):
                loaded.append(kwargs)
                self.config = config()
                self.last_image_grid_thw = [[1, 2, 2], [1, 2, 4]]

            def answer(self, images, prompt):
                # No scoring_text/selector argument is accepted or used.
                calls.append((images, prompt))
                return SimpleNamespace(answer="A", seconds=0.2, input_tokens=10, output_tokens=2,
                                       image_tokens=3, original_image_tokens=3, flops=1234,
                                       generation_seconds=0.19, decode_seconds=0.01)

            def stages(self):
                return ("ViT", "Projection", "LLM", "Answer")

        events = []
        queue = SimpleNamespace(put=events.append)
        cases = [Case(id=index, gt="A", question="Which object?", category="OC", index=index)
                 for index in range(2)]
        args = _entry.get_args(["--data", "sample.jsonl", "--model", "fake/model"])
        with patch.object(base_worker, "BaseQwenVL", Runner), patch.object(base_worker, "_load", return_value=["a", "b"]), contextlib.redirect_stderr(io.StringIO()):
            base_worker.worker("cpu", [case_dict(case) for case in cases], vars(args), queue)
        self.assertEqual(len(loaded), 1)
        self.assertNotIn("visual_pruning", loaded[0])
        self.assertEqual(len(calls), 3)  # One unscored warmup, two measured cases.
        self.assertEqual([kind for kind, _ in events], ["row", "row", "done"])
        for kind, row in events[:-1]:
            self.assertEqual(row["status"], "correct")
            self.assertTrue(row["correct"])
            self.assertFalse(row["visual_pruning"])
            self.assertEqual(row["flops"], 48448)
            self.assertEqual(row["legacy_estimated_flops"], 1234)
            self.assertEqual(row["flops_breakdown"]["score_merger_flops"], 0)
            self.assertEqual(row["selector_flops"], 0)
            self.assertEqual(row["vision_layer_token_counts"], [12] * 3)
            self.assertEqual(row["retention_ratio"], 1.0)
            self.assertEqual(row["retained_tokens"], row["original_image_tokens"])
            self.assertEqual(row["testtime"], 0.2)
            self.assertNotIn("selected_indices", row)
            self.assertNotIn("q_flowcover", row)
            self.assertNotIn("VisualTokenPruning", row["stages"])
        self.assertEqual(events[-1], ("done", "cpu"))


if __name__ == "__main__":
    unittest.main()
