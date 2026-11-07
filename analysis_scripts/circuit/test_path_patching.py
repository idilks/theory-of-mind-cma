"""CPU integration tests using actual TransformerLens hooks, no model downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel
from argparse import Namespace
from typing import ClassVar

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from path_patching import (capture, digest, freeze_writers, load_pinned_model, margin,
                           pair_provenance, patch_receivers,
                           route_spec, run_pair, validate_resume)


class PathPatchingTests(unittest.TestCase):
    model: ClassVar[HookedTransformer]
    recipient: ClassVar[torch.Tensor]
    donor: ClassVar[torch.Tensor]

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cfg = HookedTransformerConfig(n_layers=3, d_model=16, n_heads=4,
            n_key_value_heads=2, d_head=4, d_mlp=32, d_vocab=23, n_ctx=12,
            act_fn="relu", normalization_type="RMS", positional_embedding_type="rotary",
            rotary_dim=4, device="cpu", seed=42)
        cls.model = HookedTransformer(cfg).eval()
        cls.recipient = torch.tensor([[1, 2, 3, 4]])
        cls.donor = torch.tensor([[1, 6, 8, 7, 9]])

    def test_self_donor_zero_and_controls(self):
        rows = run_pair(self.model, self.recipient, self.recipient, 5, 6,
                        [[(0, 0)]], [[(2, 1)]], ["q", "k", "v"], True)
        for row in rows:
            self.assertAlmostEqual(row["path_effect"], 0, places=6)
            self.assertAlmostEqual(row["whole_sender_effect"], 0, places=6)
            self.assertEqual(row["sender_self_max_logit_error"], 0)
            self.assertEqual(row["self_path_max_logit_error"], 0)
            self.assertEqual(row["receiver_noop_max_logit_error"], 0)

    def test_nontrivial_actual_hooks_variable_length(self):
        rows = run_pair(self.model, self.recipient, self.donor, 5, 6,
                        [[(0, 0)], [(0, 0), (1, 1)]], [[(2, 1)]], ["q", "k", "v"], True)
        self.assertEqual(len(rows), 6)
        self.assertTrue(any(abs(row["path_effect"]) > 1e-5 for row in rows))
        self.assertTrue(all(row["receiver_delta_l2"][0] > 0 for row in rows))
        self.assertTrue(all(row["self_path_max_logit_error"] == 0 for row in rows))
        print("Actual-hook path effects:", [(r["senders"], r["component"], r["path_effect"]) for r in rows])

    def test_batched_effects_match_one_plan_per_forward(self):
        senders, receivers = [[(0, 0)], [(1, 1)]], [[(2, 0)], [(2, 1)], [(2, 3)]]
        unbatched = run_pair(self.model, self.recipient, self.donor, 5, 6, senders, receivers,
                             ["q", "k", "v"], True, batch_size=1)
        for size in (2, 3, 8):
            batched = run_pair(self.model, self.recipient, self.donor, 5, 6, senders, receivers,
                               ["q", "k", "v"], True, batch_size=size)
            self.assertEqual([(r["senders"], r["receivers"], r["component"]) for r in batched],
                             [(r["senders"], r["receivers"], r["component"]) for r in unbatched])
            for a, b in zip(batched, unbatched):
                self.assertAlmostEqual(a["path_effect"], b["path_effect"], places=5)
        # Independent reference: the original single-row protocol.
        names = {f"blocks.{l}.{suffix}" for l in range(3) for suffix in ("attn.hook_z", "hook_mlp_out")}
        names |= {f"blocks.2.attn.hook_{c}" for c in "qkv"}
        with torch.inference_mode():
            clean_logits, clean = capture(self.model, self.recipient, names)
            _, donor = capture(self.model, self.donor, names)
            for row in unbatched:
                sender = [tuple(h) for h in [(int(row["senders"][1]), int(row["senders"][3]))]]
                _, route_cache = capture(self.model, self.recipient, names,
                                         freeze_writers(clean, donor, sender))
                routed = self.model.run_with_hooks(self.recipient,
                    fwd_hooks=patch_receivers(route_cache, row["routes"]))[:, -1]
                expected = margin(routed, 5, 6) - margin(clean_logits, 5, 6)
                self.assertAlmostEqual(row["path_effect"], expected, places=5)
        self.assertTrue(any(abs(r["path_effect"]) > 1e-4 for r in unbatched))

    def test_frozen_writers_and_only_final_sender(self):
        names = {f"blocks.{l}.{suffix}" for l in range(3)
                 for suffix in ("attn.hook_z", "hook_mlp_out")}
        with torch.inference_mode():
            _, clean = capture(self.model, self.recipient, names)
            _, donor = capture(self.model, self.donor, names)
            _, frozen = capture(self.model, self.recipient, names,
                                 freeze_writers(clean, donor, [(0, 0)]))
        for name in names:
            expected = clean[name].clone()
            if name == "blocks.0.attn.hook_z":
                expected[:, -1, 0] = donor[name][:, -1, 0]
            torch.testing.assert_close(frozen[name], expected, rtol=0, atol=0)

    def test_gqa_requires_acknowledgment_and_records_group(self):
        with self.assertRaisesRegex(ValueError, "GQA group"):
            route_spec(self.model, [(0, 0)], [(2, 1)], "k")
        routes = route_spec(self.model, [(0, 0)], [(2, 1), (2, 0)], "v", True)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["query_heads_affected"], [0, 1])
        self.assertEqual(routes[0]["hook_index"], 0)
        self.assertTrue(routes[0]["native_kv_group"])
        q = route_spec(self.model, [(0, 0)], [(2, 1)], "q")
        self.assertEqual(q[0]["query_heads_affected"], [1])

    def test_receiver_group_edit_preserves_other_groups_and_tokens(self):
        routes = route_spec(self.model, [(0, 0)], [(2, 1)], "v", True)
        name = "blocks.2.attn.hook_v"
        value = torch.zeros(1, 4, 2, 4)
        hooks = patch_receivers({name: torch.ones_like(value)}, routes)
        edited = hooks[0][1](value, None)
        self.assertEqual(float(edited[:, -1, 0].sum()), 4)
        self.assertEqual(float(edited[:, :-1].sum()), 0)
        self.assertEqual(float(edited[:, :, 1].sum()), 0)
        self.assertEqual(float(value.sum()), 0)

    def test_invalid_paths_and_token_collision(self):
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            route_spec(self.model, [(2, 0)], [(1, 0)], "q")
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            route_spec(self.model, [(0, 4)], [(2, 0)], "q")
        with self.assertRaisesRegex(ValueError, "zero-signal"):
            run_pair(self.model, self.recipient, self.donor, 5, 5,
                     [[(0, 0)]], [[(2, 1)]], ["q"])

    def test_signed_causal_original_margin(self):
        logits = torch.tensor([[0., 2., -3.]])
        self.assertEqual(margin(logits, 2, 1), -5.)

    def test_shared_kv_interventions_run_once_with_receiver_aliases(self):
        rows = run_pair(self.model, self.recipient, self.donor, 5, 6,
                        [[(0, 0)]], [[(2, 0)], [(2, 1)]], ["q", "k", "v"], True)
        self.assertEqual(len(rows), 4)  # two Q routes, one K group, one V group
        for row in rows:
            if row["component"] in ("k", "v"):
                self.assertEqual(row["receiver_aliases"], ["L2H0", "L2H1"])

    def test_model_and_tokenizer_both_use_pinned_offline_revision(self):
        args = Namespace(model="Qwen/Qwen2.5-14B-Instruct", revision="a" * 40,
                         allow_downloads=False, device="cpu", n_devices=1, dtype="float32")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=sentinel.tokenizer) as tok, \
             patch.object(HookedTransformer, "from_pretrained", return_value=sentinel.model) as model:
            self.assertIs(load_pinned_model(args), sentinel.model)
        tok.assert_called_once_with(args.model, revision=args.revision, local_files_only=True)
        self.assertEqual(model.call_args.args, (args.model,))
        self.assertEqual(model.call_args.kwargs["revision"], args.revision)
        self.assertTrue(model.call_args.kwargs["local_files_only"])
        self.assertIs(model.call_args.kwargs["tokenizer"], sentinel.tokenizer)

    def test_export_provenance_rejects_revision_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            pairs = Path(temp) / "pairs.jsonl"
            pairs.write_text('{"pair_id":"test"}\n')
            model = "Qwen/Qwen2.5-14B-Instruct"
            revision = "a" * 40
            self.assertEqual(pair_provenance(pairs, model, revision)["status"],
                             "unverified_manual_pairs")
            sidecar = pairs.with_suffix(".meta.json")
            metadata = {"pairs_sha256": digest(pairs), "command": ["export", "--revision", revision],
                        "tokenizer_revision": None, "config": {"model": {"model_type": model}}}
            sidecar.write_text(json.dumps(metadata))
            checked = pair_provenance(pairs, model, revision)
            self.assertEqual(checked["status"], "validated_export")
            self.assertEqual(checked["sidecar_sha256"], digest(sidecar))
            with self.assertRaisesRegex(ValueError, "revision does not match"):
                pair_provenance(pairs, model, "b" * 40)
            pairs.write_text('{"pair_id":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "pair hash mismatch"):
                pair_provenance(pairs, model, revision)

    def test_resume_refuses_changed_input_or_orphan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            # Avoid executing git/environment provenance in this unit test.
            (path / "manifest.json").write_text(json.dumps({"identity": {"hash": "a"}}))
            validate_resume(path, {"hash": "a"})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_resume(path, {"hash": "b"})
            (path / "manifest.json").unlink()
            (path / "pair_000000.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Orphan"):
                validate_resume(path, {"hash": "a"})


if __name__ == "__main__":
    unittest.main()
