"""CLI and evaluator contracts that do not require model weights."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluator
import testn


class CLITests(unittest.TestCase):
    def test_help_works_without_site_packages_and_lists_shared_options(self):
        result = subprocess.run(
            [sys.executable, '-S', str(Path(testn.__file__)), '--help'],
            capture_output=True, text=True, check=True,
        )
        for flag in ('--data', '--model', '--gpus', '--prune-method', '--exec-mode'):
            self.assertIn(flag, result.stdout)

    def test_fraction_and_aliases_reach_the_evaluator(self):
        with patch.object(evaluator, 'test', return_value={'total': 1}) as run:
            self.assertEqual(testn.main(['--data', 'sample.jsonl', '--prune-fraction', '0.7',
                                         '--local-radius', '2']), {'total': 1})
        args = run.call_args.args[0]
        self.assertEqual(args.visual_prune_percent, 70.0)
        self.assertEqual(args.q_flowcover_local_radius, 2)
        self.assertEqual((args.pruning_layer, args.decision_block_index), (18, 17))
        self.assertEqual(args.prune_method, 'q_flowcover_direct_residual')

    def test_invalid_or_conflicting_options_fail_before_loading_a_model(self):
        cases = [
            ['--decision-layer-human', '14'],
            ['--decision-layer-human', '19'],
            ['--qflow-query-chunk', '0'],
            ['--local-radius', '-1'],
            ['--affinity-mode', 'bad'],
            ['--prune-fraction', '1'],
            ['--prune-fraction', '0.5', '--visual-prune-percent', '50'],
            ['--att-stats', 'summary'],
            ['--save-att-stats'],
            ['--prune-method', 'q_flowcover_seeded_all'],
        ]
        for extra in cases:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    testn.get_args(['--data', 'sample.jsonl', *extra])
                self.assertEqual(exc.exception.code, 2)

    def test_each_supported_layer_maps_to_the_actual_pruning_boundary(self):
        for layer in (15, 16, 17, 18):
            with self.subTest(layer=layer):
                args = testn.get_args([
                    '--data', 'sample.jsonl', '--decision-layer-human', str(layer),
                ])
                self.assertEqual(args.decision_layer_human, layer)
                self.assertEqual(args.pruning_layer, layer)
                self.assertEqual(args.decision_block_index, layer - 1)
                self.assertEqual(args.vision_layers, f'6,{layer},24')

    def test_ninety_eight_percent_is_forwarded_without_changing_the_layer(self):
        for budget_args in (['--visual-prune-percent', '98'], ['--prune-fraction', '0.98']):
            with self.subTest(budget_args=budget_args):
                args = testn.get_args(['--data', 'sample.jsonl', *budget_args])
                self.assertEqual(args.visual_prune_percent, 98.0)
                self.assertEqual(args.pruning_layer, 18)

    def test_local_affinity_and_chunk_aliases_are_forwarded(self):
        for mode in ('raw', 'local_centered'):
            with self.subTest(mode=mode):
                args = testn.get_args([
                    '--data', 'sample.jsonl', '--local_radius', '0',
                    '--affinity-mode', mode, '--qflow-query-chunk', '7',
                ])
                self.assertEqual(args.q_flowcover_local_radius, 0)
                self.assertEqual(args.q_flowcover_affinity_mode, mode)
                self.assertEqual(args.q_flowcover_query_chunk, 7)
                self.assertEqual(args.text_source, 'baseline')

    def test_local_checkpoint_layer_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps({'vision_config': {'depth': 24, 'spatial_merge_size': 2}}))
            args = testn.get_args(['--data', 'sample.jsonl', '--model', tmp])
            self.assertTrue(args.config_verified)
            self.assertEqual(args.vision_depth, 24)
            config.write_text(json.dumps({'vision_config': {'depth': 18}}))
            with self.assertRaisesRegex(ValueError, 'depth=18'):
                testn.get_args(['--data', 'sample.jsonl', '--model', tmp])

    def test_local_checkpoint_validation_uses_the_requested_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps({'vision_config': {'depth': 18, 'spatial_merge_size': 2}}))
            for layer in (15, 16, 17):
                with self.subTest(layer=layer):
                    args = testn.get_args([
                        '--data', 'sample.jsonl', '--model', tmp,
                        '--decision-layer-human', str(layer),
                    ])
                    self.assertTrue(args.config_verified)
                    self.assertEqual(args.decision_block_index, layer - 1)
            with self.assertRaisesRegex(ValueError, 'depth=18'):
                testn.get_args([
                    '--data', 'sample.jsonl', '--model', tmp,
                    '--decision-layer-human', '18',
                ])


class ExecutionTraceTests(unittest.TestCase):
    def row(self, full_blocks, layer=18):
        return {
            'index': 0, 'selection_mode': 'middle_text_early', 'selection_strategy': 'topk',
            'selection_layer': layer, 'original_image_tokens': 4, 'retained_tokens': 2,
            'vision_merge_unit': 4, 'vision_layer_token_counts': [16] * full_blocks + [8] * (24 - full_blocks),
            'selection_budgets': [(0, 2, 0)], 'selected_indices': [[0, 2]],
            'selection_groups': [{'final': [], 'middle': [0, 2], 'shallow': []}],
        }

    def test_each_mode_accepts_its_own_token_trace(self):
        for layer in (15, 16, 17, 18):
            for mode in ('hard18', 'full19_gather', 'att_bridge19_fullkv'):
                with self.subTest(layer=layer, mode=mode):
                    args = testn.get_args([
                        '--data', 'sample.jsonl', '--exec-mode', mode,
                        '--decision-layer-human', str(layer), '--visual-keep-tokens', '2',
                    ])
                    full_blocks = layer if mode == 'hard18' else layer + 1
                    self.assertTrue(evaluator._validate_pruning_row(self.row(full_blocks, layer), args))

    def test_wrong_trace_and_wrong_fixed_budget_are_rejected(self):
        args = testn.get_args(['--data', 'sample.jsonl', '--visual-keep-tokens', '2'])
        with self.assertRaisesRegex(RuntimeError, 'execution trace'):
            evaluator._validate_pruning_row(self.row(19), args)
        row = self.row(18)
        row['selected_indices'] = [[0]]
        row['selection_budgets'] = [(0, 1, 0)]
        with self.assertRaisesRegex(RuntimeError, 'override K=2'):
            evaluator._validate_pruning_row(row, args)

    def test_trace_from_another_layer_is_rejected(self):
        args = testn.get_args([
            '--data', 'sample.jsonl', '--decision-layer-human', '15', '--visual-keep-tokens', '2',
        ])
        with self.assertRaisesRegex(RuntimeError, 'selection layer mismatch'):
            evaluator._validate_pruning_row(self.row(18, layer=18), args)
        with self.assertRaisesRegex(RuntimeError, 'execution trace'):
            evaluator._validate_pruning_row(self.row(18, layer=15), args)


class ResumeTests(unittest.TestCase):
    def test_changed_selector_or_budget_cannot_reuse_old_results(self):
        args = testn.get_args(['--data', 'sample.jsonl'])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'results.jsonl'
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps(vars(args)))
            evaluator._validate_resume_config(source, args)
            for key, value in [('prune_method', 'text_cos'), ('visual_keep_ratio', 0.3),
                               ('pruning_layer', 15),
                               ('q_flowcover_local_radius', 2), ('q_flowcover_affinity_mode', 'local_centered'),
                               ('text_source', 'full_prompt')]:
                previous = vars(args).copy()
                previous[key] = value
                config.write_text(json.dumps(previous))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    evaluator._validate_resume_config(source, args)

    def test_resume_requires_original_configuration(self):
        args = testn.get_args(['--data', 'sample.jsonl'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'missing'):
                evaluator._validate_resume_config(Path(tmp) / 'results.jsonl', args)


if __name__ == '__main__':
    unittest.main()
