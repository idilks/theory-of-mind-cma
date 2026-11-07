"""Combined steering tests use actual transformer hooks; no downloaded model."""
import tempfile
import unittest
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from stage2_steering import make_steer_fn
from steering_bridge import (bridge_forward, bridge_conditions, narrative_mask,
                             select_code_heads, validate_partition, write_identity,
                             save_raw_states, validate_checkpoint, atomic_json, answer_token)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = HookedTransformer(HookedTransformerConfig(
            n_layers=3, d_model=16, n_heads=4, n_key_value_heads=2,
            d_head=4, d_mlp=32, d_vocab=23, n_ctx=16, act_fn="relu",
            normalization_type="RMS", positional_embedding_type="rotary",
            rotary_dim=4, device="cpu", seed=42)).eval()
        cls.tokens = torch.tensor([[1, 2, 3, 4]])
        cls.site = "blocks.2.hook_resid_pre"
        cls.readers = [(2, 1), (2, 3)]
        cls.bases = {"rank1": torch.eye(16, dtype=torch.float64)[:1],
                     "subspace": torch.eye(16, dtype=torch.float64)[:3]}

    def test_combined_hooks_restore_coordinate_preserve_complement(self):
        clean = bridge_forward(self.model, self.tokens, {}, self.site, self.readers)
        deltas = {(0, 0): torch.tensor([.3, -.1, .2, .4]),
                  (1, 2): torch.tensor([-.2, .2, .1, .3])}
        results = bridge_conditions(self.model, self.tokens, deltas, self.site,
            self.readers, self.bases, clean, make_steer_fn, control_draws=2, seed=7)
        steer = results["head_only"]
        self.assertGreater(float((steer["residual"] - clean["residual"]).norm()), 0)
        for name, basis in self.bases.items():
            restored = results[f"{name}/restore"]["residual"]
            delta = steer["residual"] - clean["residual"]
            component = (delta @ basis.T) @ basis
            torch.testing.assert_close(restored, steer["residual"] - component,
                                       atol=2e-7, rtol=0)
            torch.testing.assert_close(results[f"{name}/component_only"]["residual"],
                                       clean["residual"] + component, atol=2e-7, rtol=0)
            self.assertEqual(results[f"{name}/restore"]["attention"].shape, (2, 4))
            for draw in range(2):
                shift = results[f"{name}/control_{draw}"]["residual"] - steer["residual"]
                self.assertAlmostEqual(float(shift.norm()), float(component.norm()), places=6)
                torch.testing.assert_close(shift @ basis.T, torch.zeros(basis.shape[0],
                    dtype=torch.float64), atol=2e-7, rtol=0)
        print("Actual-hook bridge: restoration, component-only, controls and attention passed")

    def test_zero_steering_and_restoration_are_identity(self):
        clean = bridge_forward(self.model, self.tokens, {}, self.site, self.readers)
        results = bridge_conditions(self.model, self.tokens, {(0, 0): torch.zeros(4)},
            self.site, self.readers, self.bases, clean, make_steer_fn,
            control_draws=1, seed=5)
        for result in results.values():
            torch.testing.assert_close(result["logits"], clean["logits"], atol=0, rtol=0)
            torch.testing.assert_close(result["residual"], clean["residual"], atol=0, rtol=0)

    def test_direct_write_uses_actual_selected_head_delta(self):
        delta = torch.tensor([.1, -.2, .3, -.4])
        result = bridge_forward(self.model, self.tokens, {(0, 2): delta},
                                self.site, self.readers)
        actual = result["head_deltas"][(0, 2)]
        expected = actual @ self.model.blocks[0].attn.W_O[2].detach().double().cpu()
        torch.testing.assert_close(result["direct_write"], expected)
        self.assertLess(float((actual - delta.double()).abs().max()), 1e-6)

    def test_bfloat16_rotated_basis_restoration(self):
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        q, _ = torch.linalg.qr(torch.randn(16, 3, dtype=torch.float64,
                                          generator=torch.Generator().manual_seed(12)))
        bases = {"rank1": q.T[:1], "subspace": q.T}
        clean = bridge_forward(model, self.tokens, {}, self.site, self.readers)
        results = bridge_conditions(model, self.tokens,
            {(0, 0): torch.tensor([.3, -.1, .2, .4])}, self.site,
            self.readers, bases, clean, make_steer_fn, control_draws=2, seed=7)
        for name, basis in bases.items():
            restored = results[f"{name}/restore"]
            check = restored["validation"]
            self.assertEqual(check["max_error_from_rounded_target"], 0)
            self.assertLessEqual(check["coordinate_error"], check["rounding_error_norm"] + 1e-12)
            delta = results["head_only"]["residual"] - clean["residual"]
            component = (delta @ basis.T) @ basis
            expected_component = (clean["residual"] + component).bfloat16().double()
            torch.testing.assert_close(results[f"{name}/component_only"]["residual"],
                                       expected_component, atol=0, rtol=0)

    def test_raw_checkpoint_preserves_states_and_rejects_corruption(self):
        clean = bridge_forward(self.model, self.tokens, {}, self.site, self.readers)
        steered = bridge_forward(self.model, self.tokens, {(0, 2): torch.ones(4)},
                                 self.site, self.readers)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt_0000.json"
            artifact = save_raw_states(path.with_suffix(".npz"), [clean, steered], [(0, 2)])
            atomic_json(path, {"prompt_sha256": hashlib.sha256(b"test").hexdigest(),
                               "raw_states": artifact})
            validate_checkpoint(path, "test")
            with np.load(path.with_suffix(".npz")) as saved:
                np.testing.assert_array_equal(saved["residual"][0], clean["residual"].numpy())
                np.testing.assert_array_equal(saved["head_deltas"][0], np.zeros((1, 4)))
                np.testing.assert_array_equal(saved["head_deltas"][1, 0],
                                              steered["head_deltas"][(0, 2)].numpy())
                np.testing.assert_array_equal(saved["logits"][1], steered["logits"].numpy())
            with self.assertRaisesRegex(ValueError, "prompt mismatch"):
                validate_checkpoint(path, "different")
            path.with_suffix(".npz").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "raw states"):
                validate_checkpoint(path, "test")

    def test_answer_tokens_follow_capture_spacing_convention(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [ord(text[0])]
        self.assertEqual(answer_token(Tokenizer(), "park", "go to "), ord(" "))
        self.assertEqual(answer_token(Tokenizer(), "park", "go to"), ord("p"))

    def test_sparse_head_map_rejects_missing_not_silent_reshape(self):
        capture = {"z": torch.arange(24).reshape(3, 2, 4),
                   "heads": [{"layer": 1, "head": 3}, {"layer": 0, "head": 2}]}
        selected = select_code_heads(capture, [(0, 2), (1, 3)])
        torch.testing.assert_close(selected[:, 0], capture["z"][:, 1].double())
        with self.assertRaisesRegex(ValueError, "missing"):
            select_code_heads(capture, [(0, 0)])

    def test_dense_head_selection(self):
        z = torch.arange(3 * 2 * 4 * 4).reshape(3, 2, 4, 4)
        selected = select_code_heads({"z": z}, [(1, 3), (0, 2)])
        torch.testing.assert_close(selected[:, 0], z[:, 1, 3].double())

    def test_fit_hold_partition(self):
        validate_partition(np.array([0, 2]), np.array([1, 3]), 4)
        for fit, hold in [([0, 1], [1, 2]), ([0], [1]), ([0, 0], [1, 2, 3])]:
            with self.assertRaises(ValueError):
                validate_partition(np.array(fit), np.array(hold), 4)

    def test_manifest_protects_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_identity(path, {"source": "a", "inputs": "x"})
            write_identity(path, {"source": "a", "inputs": "x"})
            with self.assertRaisesRegex(ValueError, "identity"):
                write_identity(path, {"source": "b", "inputs": "x"})

    def test_masks_exclude_special_tokens_and_question(self):
        class CharTokenizer:
            def __call__(self, text, **kwargs):
                return {"offset_mapping": [(i, i + 1) for i in range(len(text))],
                        "input_ids": list(range(len(text)))}
        prompt = "<|im_start|>user\nAlice left art for history.\nWhere? Answer art.<|im_end|>"
        mask = narrative_mask(CharTokenizer(), prompt, "art", len(prompt))
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual(np.where(mask)[0].tolist(), list(range(prompt.index("art for"),
                                                            prompt.index("art for") + 3)))
        with self.assertRaises(ValueError):
            narrative_mask(CharTokenizer(), prompt, "missing", len(prompt))


if __name__ == "__main__":
    unittest.main()
