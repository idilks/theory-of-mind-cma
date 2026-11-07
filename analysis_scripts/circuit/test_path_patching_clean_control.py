"""Actual-hook controls and coverage/provenance regressions; no model download."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, ClassVar

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import path_patching as core
import path_patching_clean_control as control

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/circuit/path_patching/union41_v2'


class ControlTests(unittest.TestCase):
    model: ClassVar[Any]
    tokens: ClassVar[torch.Tensor]
    pair: ClassVar[dict]
    rows: ClassVar[list]
    chunks: ClassVar[list]
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = HookedTransformer(HookedTransformerConfig(n_layers=3, d_model=16,
            n_heads=4, n_key_value_heads=2, d_head=4, d_mlp=32, d_vocab=23,
            n_ctx=12, act_fn='relu', normalization_type='RMS',
            positional_embedding_type='rotary', rotary_dim=4, device='cpu', seed=42)).eval()
        cls.tokens = torch.tensor([[1, 2, 3, 4]])
        cls.pair = {'causal_token_id': 5, 'original_token_id': 6}
        cls.rows = core.run_pair(cls.model, cls.tokens, cls.tokens, 5, 6,
            [[(0, 0)], [(1, 1)]], [[(2, 0)], [(2, 1)], [(2, 3)]],
            ['q', 'k', 'v'], True, batch_size=3)
        cls.chunks = control.layouts(cls.rows, 3, ['L0H0', 'L1H1'])

    def test_actual_hooks_zero_with_full_and_partial_batches(self):
        result = control.measure(core, self.model, self.tokens, self.pair, self.chunks, 1e-4)
        self.assertEqual({r['batch_width'] for r in result['rows']}, {4, 2})
        self.assertTrue(all(r['passes_tolerance'] for r in result['rows']))
        self.assertTrue(result['clean_and_frozen_receiver_caches_identical'])
        self.assertEqual(len(result['rows']), 7)  # 3 Q + 2 K groups + 2 V groups
        mapped = [m['row_index'] for r in result['rows'] for m in r['source_rows']]
        self.assertEqual(sorted(mapped), list(range(14)))
        print('Clean actual-hook maximum error:', max(r['max_logit_error'] for r in result['rows']))

    def test_detects_batch_dependent_activations_despite_identical_single_caches(self):
        def batch_shift(value, hook):
            if value.shape[0] > 1:
                changed = value.clone()
                changed[:, -1, 0, 0] += 1
                return changed
            return value
        self.model.add_hook('blocks.2.attn.hook_v', batch_shift, is_permanent=True)
        try:
            result = control.measure(core, self.model, self.tokens, self.pair, self.chunks, 1e-4)
        finally:
            self.model.reset_hooks(including_permanent=True)
        self.assertTrue(result['clean_and_frozen_receiver_caches_identical'])
        self.assertTrue(any(not r['passes_tolerance'] for r in result['rows']))
        self.assertEqual(len(result['rows']), 7)  # failures do not abort the grid
        self.assertTrue(all(r['batch_row_max_logit_error'] < 1e-4 for r in result['rows']))
        self.assertTrue(any(max(r['single_to_batch_receiver_delta_l2']) > .9 for r in result['rows']))
        print('Injected batch discrepancy detected:', max(r['max_logit_error'] for r in result['rows']))

    def test_distinct_row_orders_are_not_deduplicated(self):
        rows = self.rows[:7] + list(reversed(self.rows[7:]))
        chunks = control.layouts(rows, 3, ['L0H0', 'L1H1'])
        self.assertEqual(len(chunks), 6)
        with self.assertRaisesRegex(ValueError, 'expected senders'):
            control.layouts(self.rows[:7], 3, ['L0H0', 'L1H1'])

    def test_frozen_source_discrepancy_separated_from_clean_cache_control(self):
        capture = core.capture
        def changed_frozen_cache(model, tokens, names, hooks=None):
            logits, cache = capture(model, tokens, names, hooks)
            if hooks:
                cache = {k: v.clone() for k, v in cache.items()}
                cache['blocks.2.attn.hook_v'][:, -1, 0, 0] += 1
            return logits, cache
        with patch.object(core, 'capture', side_effect=changed_frozen_cache):
            result = control.measure(core, self.model, self.tokens, self.pair, self.chunks, 1e-4)
        self.assertFalse(result['clean_and_frozen_receiver_caches_identical'])
        self.assertTrue(any(r['max_logit_error'] > 1e-3 for r in result['rows']))
        self.assertTrue(all(r['clean_cache_max_logit_error'] < 1e-4 for r in result['rows']))

    def test_resume_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            control.freeze(core, path, {'input_sha': 'a'})
            control.freeze(core, path, {'input_sha': 'a'})
            with self.assertRaisesRegex(ValueError, 'Changed inputs'):
                control.freeze(core, path, {'input_sha': 'b'})

    def test_execute_saves_and_resumes_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            source.mkdir()
            study = {'versions': core.provenance()['versions'], 'plan': {}}
            core.atomic_json(source/'study.json', study)
            pair = {**self.pair, 'recipient_prompt': 'fixed', 'causal_ans': 'causal', 'exp_ans': 'original'}
            file = source/'pair_000000.json'
            core.atomic_json(file, {'rows': self.rows})
            cfg = json.loads(json.dumps(self.model.cfg.to_dict(), default=str))
            job = dict(file=file, scope='tiny', index=0, pair=pair, chunks=self.chunks,
                       arguments={'device': 'cpu'}, original_rows=14, model_config=cfg)
            task = {'task': {'cell': {'id': 'tiny_cell'}, 'groups': [{'sender_pairs': [{}, {}]}]}}
            inspection = (study, task, [job], {file.name: control.sha(file)})
            args = argparse.Namespace(source_root=source, output_root=output, task_id=0,
                                      preflight=False, tolerance=1e-4)
            tokenizer = MagicMock(return_value=argparse.Namespace(input_ids=self.tokens),
                                  padding_side='right', pad_token_id=0, bos_token_id=0)
            with patch.object(control, 'load_core', return_value=core), \
                 patch.object(control, 'inspect_task', return_value=inspection), \
                 patch.object(core, 'prepare_model', return_value=self.model) as prepare, \
                 patch.object(torch.cuda, 'get_device_name', return_value='CPU test'), \
                 patch.object(self.model, 'tokenizer', tokenizer), \
                 patch.object(self.model, 'to_tokens', side_effect=lambda s, **k: torch.tensor([[5 if s=='causal' else 6]])):
                control.execute(args)
                self.assertEqual(prepare.call_count, 1)
                control.execute(args)
                self.assertEqual(prepare.call_count, 1)  # resume did not load
            saved = control.read(output/'tiny_cell/tiny/pair_000000.json')
            self.assertEqual(saved['source_sha256'], control.sha(file))
            self.assertEqual(len(saved['rows']), 7)
            self.assertEqual(control.read(output/'tiny_cell/summary.json')['status'], 'passed')

    @unittest.skipUnless(SOURCE.exists(), 'Completed study not installed')
    def test_all_14_cells_all_41_heads_source_coverage_without_model_loading(self):
        original = control.load_core(SOURCE)
        total, controls = 0, 0
        with patch.object(original, 'prepare_model', side_effect=AssertionError('Must not load')):
            for task_id in range(14):
                _, task, jobs, _ = control.inspect_task(SOURCE, task_id, original)
                self.assertEqual(sum(len(g['sender_pairs']) for g in task['task']['groups']), 41)
                for job in jobs:
                    total += job['original_rows']
                    controls += sum(len(c['plans']) for c in job['chunks'])
                    mapped = [m['row_index'] for c in job['chunks'] for mappings in c['source_rows'] for m in mappings]
                    self.assertEqual(sorted(mapped), list(range(job['original_rows'])))
        self.assertEqual(total, 621040)
        self.assertEqual(controls, 15400)
        print(f'Validated complete coverage: {controls} clean rows -> {total} original paths; 14 cells, 41 + 41 heads')

    @unittest.skipUnless(SOURCE.exists(), 'Completed study not installed')
    def test_dependency_drift_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(source_root=SOURCE, output_root=Path(tmp)/'control',
                                      task_id=0, preflight=False, tolerance=1e-4)
            original = control.load_core(SOURCE)
            with patch.object(control, 'load_core', return_value=original), \
                 patch.object(original, 'provenance', return_value={'versions': {}}), \
                 patch.object(original, 'prepare_model', side_effect=AssertionError('Must not load')):
                with self.assertRaisesRegex(ValueError, 'Dependency drift'):
                    control.execute(args)
            self.assertFalse(args.output_root.exists())


if __name__ == '__main__':
    unittest.main()
