"""Independent CLI/worker/resume contracts, without loading a checkpoint."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluator
import testn
from utils.data import Case

# unittest discovery prepends tests/, where the mathematical test module has
# the same basename as the independent entry point. Load the intended file
# under a distinct test-only name rather than importing the wrong module.
SPEC = importlib.util.spec_from_file_location(
    "_perturbation_entry_under_test", Path(__file__).resolve().parents[1] / "test_perturbation.py"
)
assert SPEC is not None and SPEC.loader is not None
test_perturbation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(test_perturbation)


class PerturbationEntryTests(unittest.TestCase):
    def args(self, *extra):
        return test_perturbation.get_args(["--data", "sample.jsonl", "--model", "unloaded/model", *extra])

    def test_help_and_explicit_evaluator_dispatch_do_not_load_weights(self):
        result = subprocess.run([sys.executable, "-S", test_perturbation.__file__, "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("q_flowcover_perturbation_nocover", result.stdout)
        self.assertIn("--query-chunk-size", result.stdout)
        with patch.object(evaluator, "test", return_value={"total": 1}) as run:
            self.assertEqual(test_perturbation.main(["--data", "sample.jsonl"]), {"total": 1})
        self.assertEqual(run.call_args.args[0].evaluation_variant, "perturbation")

    def test_new_defaults_and_old_defaults_remain_distinct(self):
        new = self.args()
        old = testn.get_args(["--data", "sample.jsonl", "--model", "unloaded/model"])
        self.assertEqual((new.prune_method, new.exec_mode, new.worker_cases),
                         ("q_flowcover_perturbation", "full19_gather", 0))
        self.assertEqual((new.temperature, new.perturbation_lambda, new.local_radius), (1., 1., 1))
        self.assertEqual((old.prune_method, old.exec_mode, old.worker_cases),
                         ("q_flowcover_direct_residual", "hard18", 128))
        self.assertFalse(hasattr(old, "evaluation_variant"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            testn.get_args(["--data", "sample.jsonl", "--prune-method", "q_flowcover_perturbation"])

    def test_relative_next_layer_and_checkpoint_depth_are_not_hardcoded(self):
        for layer in (2, 12, 15, 18, 20, 23):
            with self.subTest(layer=layer):
                args = self.args("--decision-layer-human", str(layer))
                self.assertEqual((args.score_layer, args.perturbation_layer, args.actual_prune_after_layer),
                                 (layer, layer + 1, layer + 1))
                self.assertEqual((args.decision_block_index, args.bridge_block_index), (layer - 1, layer))
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_text(json.dumps({"vision_config": {"depth": 6}}))
            args = self.args("--model", folder, "--decision-layer-human", "4")
            self.assertTrue(args.config_verified)
            self.assertEqual(args.vision_layers, "3,4,6")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("--model", folder, "--decision-layer-human", "6")

    def test_unknown_modes_invalid_parameters_and_old_control_changes_fail(self):
        invalid = [
            ("--prune-method", "unregistered"), ("--exec-mode", "hard18"),
            ("--decision-layer-human", "1"), ("--decision-layer-human", "24"),
            ("--temperature", "0"), ("--temperature", "nan"),
            ("--perturbation-lambda", "-1"), ("--perturbation-lambda", "inf"),
            ("--query-chunk-size", "0"), ("--group-chunk-size", "0"),
            ("--source-chunk-size", "0"), ("--local-radius", "-1"),
            ("--prune-fraction", "1"),
            ("--prune-fraction", ".5", "--visual-prune-percent", "50"),
            ("--prune-method", "text_cos", "--temperature", ".5"),
            ("--prune-method", "q_flowcover_direct_residual", "--perturbation-lambda", "2"),
        ]
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.args(*extra)
            self.assertEqual(error.exception.code, 2)
        for method in ("q_flowcover_perturbation", "q_flowcover_perturbation_nocover"):
            self.assertEqual(self.args("--prune-method", method, "--temperature", ".5", "--perturbation-lambda", "2").temperature, .5)

    def test_fraction_dry_run_and_diagnostics_are_explicit(self):
        args = self.args("--prune-fraction", ".9", "--dry-run", "--verify-candidates", "0", "--no-save-diagnostics")
        self.assertEqual(args.visual_prune_percent, 90.)
        self.assertEqual(args.limit, 1)
        self.assertEqual(args.verify_candidates, 2)
        self.assertFalse(args.save_diagnostics)

    def test_worker_factory_uses_new_runner_only_for_new_methods(self):
        from compression.perturbation_worker import build_runner
        from compression import qwen3_vl_modes as old_module
        from compression import qwen3_vl_perturbation as new_module
        # Import the derived class before mocking its base constructor. If the
        # adapter module is first imported while its base is MagicMock, Python
        # creates a mocked "class" and poisons later native-runtime tests.
        old_type, new_type = old_module.Qwen3VLCompressionRunner, new_module.Qwen3VLPerturbationRunner
        self.assertIsInstance(new_type, type)
        for method in test_perturbation.METHODS:
            cfg = vars(self.args("--prune-method", method, "--no-save-diagnostics"))
            old, new = object(), object()
            with patch.object(old_module, "Qwen3VLCompressionRunner", return_value=old) as old_ctor, \
                 patch.object(new_module, "Qwen3VLPerturbationRunner", return_value=new) as new_ctor:
                result = build_runner("cpu", cfg)
            if method.startswith("q_flowcover_perturbation"):
                self.assertIs(result, new)
                old_ctor.assert_not_called()
                self.assertFalse(new_ctor.call_args.kwargs["perturbation_compare_flow"])
            else:
                self.assertIs(result, old)
                new_ctor.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            build_runner("cpu", dict(vars(self.args()), prune_method="unknown"))
        self.assertIs(old_module.Qwen3VLCompressionRunner, old_type)
        self.assertIs(new_module.Qwen3VLPerturbationRunner, new_type)

    def test_parallel_evaluator_imports_the_independent_worker(self):
        from compression import perturbation_worker
        class Dispatched(Exception):
            pass
        chosen = []
        marker = object()
        class Context:
            def Queue(self):
                return object()
            def Process(self, *, target, args):
                chosen.append(target)
                raise Dispatched()
        with tempfile.TemporaryDirectory() as folder:
            args = self.args()
            args.output = str(Path(folder) / "results.jsonl")
            with patch.object(evaluator.mp, "get_context", return_value=Context()), \
                 patch.object(perturbation_worker, "worker", marker), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(Dispatched):
                evaluator._run_parallel([Case(id="native", gt="A", question="Q")], args,
                                        ["cpu"], SimpleNamespace(write=lambda *args, **kwargs: None))
        self.assertEqual(chosen, [marker])

    def test_resume_rejects_old_runs_and_changed_new_algorithm_settings(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "results.jsonl"
            config = Path(folder) / "config.json"
            config.write_text(json.dumps(vars(args)))
            evaluator._validate_resume_config(source, args)
            changes = {
                "evaluation_variant": None, "temperature": .5, "perturbation_lambda": 2.,
                "local_radius": 0, "query_chunk_size": 7, "group_chunk_size": 3,
                "source_chunk_size": 12, "seed": 3, "save_diagnostics": False,
                "verify_candidates": 0, "warmup_cases": 0,
            }
            for key, value in changes.items():
                config.write_text(json.dumps({**vars(args), key: value}))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    evaluator._validate_resume_config(source, args)
            # Isolation is symmetric even for shared A/B method names and the
            # same full19_gather boundary: their worker/timing paths differ.
            old = testn.get_args(["--data", "sample.jsonl", "--prune-method", "text_cos", "--exec-mode", "full19_gather"])
            new = self.args("--prune-method", "text_cos")
            config.write_text(json.dumps(vars(new)))
            with self.assertRaisesRegex(ValueError, "evaluation_variant"):
                evaluator._validate_resume_config(source, old)
            config.write_text(json.dumps(vars(old)))
            with self.assertRaisesRegex(ValueError, "evaluation_variant"):
                evaluator._validate_resume_config(source, new)


if __name__ == "__main__":
    unittest.main()
