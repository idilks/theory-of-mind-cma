"""Tests for the belief-free generic collateral leg of stage2_steering.

No downloaded model: a tiny HookedTransformer and a stub tokenizer.
Run: .venv/bin/python -m pytest analysis_scripts/circuit/test_generic_steering.py -q
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from generic_prompts import (GENERIC_VARIANT, build_generic_prompts,
                             generic_user_messages)
from stage2_steering import eval_config, make_steer_fn, render_figures


class StubTokenizer:
    """Deterministic token ids from the prompt text; chat template = concat."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [1 + (ord(c) % 20) for c in text[-8:]]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


class GenericPromptTests(unittest.TestCase):
    def test_pool_size_unique_and_deterministic(self):
        a = generic_user_messages(225, 99)
        b = generic_user_messages(225, 99)
        c = generic_user_messages(225, 7)
        self.assertEqual(len(a), 225)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({m for _, m in a}), 225)
        families = {f for f, _ in a}
        self.assertIn("capital", families)
        self.assertIn("arithmetic", families)
        self.assertIn("open", families)

    def test_rows_are_label_free(self):
        rows = build_generic_prompts(StubTokenizer(), 10, 99)
        self.assertEqual(len(rows), 10)
        for r in rows:
            self.assertEqual(r["variant"], GENERIC_VARIANT)
            self.assertEqual(r["correct"], "")
            self.assertEqual(r["alt"], "")
            self.assertFalse(r["believed"])
            self.assertTrue(r["prompt"].endswith("<assistant>"))
            self.assertTrue(r["template"].startswith("generic:"))

    def test_no_tom_vocabulary(self):
        text = " ".join(m for _, m in generic_user_messages(244, 99)).lower()
        for banned in ("belief", "believe", "alice", "bob", "truck", "salon",
                       "library", "moved", "thinks"):
            self.assertNotIn(banned, text)


class LabelFreeEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = HookedTransformer(HookedTransformerConfig(
            n_layers=3, d_model=16, n_heads=4, d_head=4, d_mlp=32,
            d_vocab=23, n_ctx=16, act_fn="relu", normalization_type="RMS",
            positional_embedding_type="rotary", rotary_dim=4, device="cpu",
            seed=42)).eval()
        cls.tok = StubTokenizer()
        cls.prompts = build_generic_prompts(cls.tok, 6, 99)
        cls.hook = "blocks.2.hook_resid_pre"

    def _run(self, hooks, clean, norms=None):
        # the steer fn and the eval loop must share ONE applied_norms list,
        # as main() does; the loop clears it per prompt and records the mean
        norms = [] if norms is None else norms
        rows, attn, lp = eval_config(self.model, self.tok, self.prompts, hooks,
                                     norms, clean, False, [], set(), "cpu")
        return rows, attn, lp

    def test_clean_pass_rows(self):
        rows, attn, lp = self._run([], None)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(lp), 6)
        self.assertEqual(attn, [])  # no believed/current -> no attention rows
        for r in rows:
            self.assertTrue(np.isnan(r["p_correct"]))
            self.assertTrue(np.isnan(r["flip_score"]))
            self.assertEqual(r["kl_from_clean"], 0.0)
            self.assertEqual(r["top1_agree"], 1.0)
            self.assertGreater(r["p_top1"], 0.0)

    def test_steered_pass_tracks_clean(self):
        _, _, clean = self._run([], None)
        vec = torch.zeros(16, dtype=torch.float64)
        vec[0] = 1.0
        norms = []
        # zero dose must reproduce clean: KL 0, top-1 agreement 1
        fn0 = make_steer_fn("add", 0.0, norms, vec=vec, scale=5.0, sign=1.0)
        rows0, _, _ = self._run([(self.hook, fn0)], clean, norms)
        for r in rows0:
            self.assertAlmostEqual(r["kl_from_clean"], 0.0, places=5)
            self.assertEqual(r["top1_agree"], 1.0)
        # large dose: KL positive, agreement in {0, 1}, applied norm = dose*scale
        fn = make_steer_fn("add", 40.0, norms, vec=vec, scale=5.0, sign=-1.0)
        rows, _, _ = self._run([(self.hook, fn)], clean, norms)
        self.assertTrue(all(r["kl_from_clean"] > 0 for r in rows))
        self.assertTrue(all(r["top1_agree"] in (0.0, 1.0) for r in rows))
        self.assertTrue(all(abs(r["applied_norm"] - 200.0) < 1e-6 for r in rows))


class SummaryTests(unittest.TestCase):
    def test_render_handles_label_free_rows(self):
        rows = []
        for direction, dose in (("clean", 0.0), ("target_rank1", 4.0),
                                ("target_rank1", 8.0), ("random", 4.0)):
            for i in range(5):
                rows.append(dict(prompt_set="generic", mode="inject",
                                 direction=direction, dose=dose,
                                 template="generic:capital", pair_idx=i,
                                 variant="generic", p_correct=np.nan,
                                 p_alt=np.nan, flip_score=np.nan,
                                 kl_from_clean=0.0 if direction == "clean" else dose / 10,
                                 top1_str="x", top1_agree=1.0 if dose < 8 else 0.0,
                                 p_top1=0.9, applied_norm=dose * 50))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            pd.DataFrame(rows).to_csv(out / "per_prompt.csv", index=False)
            render_figures(out)
            summary = pd.read_csv(out / "summary.csv")
            self.assertIn("top1_agree", summary.columns)
            self.assertEqual(int(summary["n"].iloc[0]), 5)  # NaN flip must not shrink n
            r8 = summary[(summary.direction == "target_rank1") & (summary.dose == 8.0)]
            self.assertEqual(float(r8["top1_agree"].iloc[0]), 0.0)
            self.assertTrue((out / "figures" / "dose_kl.png").exists())
            self.assertTrue((out / "figures" / "dose_top1_agree.png").exists())
            self.assertFalse((out / "figures" / "dose_flip_score.png").exists())

    def test_render_backfills_old_runs(self):
        # L41_main-style per_prompt.csv without the new columns
        rows = [dict(prompt_set="true_belief", mode="inject", direction=d,
                     dose=dose, template="food_truck", pair_idx=i,
                     variant="true_belief", p_correct=0.9, p_alt=0.05,
                     flip_score=0.05, kl_from_clean=0.1, top1_str="x",
                     applied_norm=50.0)
                for d, dose in (("clean", 0.0), ("random", 1.0)) for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            pd.DataFrame(rows).to_csv(out / "per_prompt.csv", index=False)
            render_figures(out)
            summary = pd.read_csv(out / "summary.csv")
            self.assertTrue(summary["top1_agree"].isna().all())
            self.assertTrue((out / "figures" / "dose_flip_score.png").exists())


if __name__ == "__main__":
    unittest.main()
