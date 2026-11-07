"""Reader-restoration hooks on a tiny GQA transformer; no downloaded model."""
import unittest

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from stage2_steering import make_steer_fn
from stage2_reader_restoration import (
    by_layer, expected_runs, forward, kv_groups, layer_matched_control_sets,
    make_restore_hooks, readouts, run_prompt,
)


class ReaderRestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = HookedTransformer(HookedTransformerConfig(
            n_layers=3, d_model=16, n_heads=4, n_key_value_heads=2,
            d_head=4, d_mlp=32, d_vocab=23, n_ctx=16, act_fn="relu",
            normalization_type="RMS", positional_embedding_type="rotary",
            rotary_dim=4, device="cpu", seed=42)).eval()
        cls.tokens = torch.tensor([[1, 2, 3, 4, 5]])
        cls.readers = [(1, 0), (2, 1), (2, 3)]
        cls.layers = [1, 2]
        cls.sites = ["q", "k", "v", "pattern"]
        vec = torch.zeros(16, dtype=torch.float64)
        vec[0] = 1.0
        cls.norms = []
        cls.steer = ("blocks.1.hook_resid_pre",
                     make_steer_fn("add", 4.0, cls.norms, vec=vec, scale=0.5, sign=1.0))

    def test_restore_from_own_run_is_identity(self):
        clean_lg, clean_st = forward(self.model, self.tokens, [], self.layers, self.sites)
        steer_lg, steer_st = forward(self.model, self.tokens, [self.steer], self.layers, self.sites)
        self.assertGreater(float((steer_lg - clean_lg).abs().max()), 1e-4)
        for site in self.sites:
            err = []
            lg, st = forward(self.model, self.tokens,
                             make_restore_hooks(site, self.readers, clean_st, 4, 2, err),
                             self.layers, self.sites)
            torch.testing.assert_close(lg, clean_lg, atol=1e-6, rtol=0)
            self.assertEqual(max(err), 0.0)
            err = []
            lg, st = forward(self.model, self.tokens,
                             [self.steer] + make_restore_hooks(site, self.readers, steer_st, 4, 2, err),
                             self.layers, self.sites)
            torch.testing.assert_close(lg, steer_lg, atol=1e-6, rtol=0)
            self.assertEqual(max(err), 0.0)

    def test_restored_activation_equals_target_and_others_untouched(self):
        clean_lg, clean_st = forward(self.model, self.tokens, [], self.layers, self.sites)
        _, steer_st = forward(self.model, self.tokens, [self.steer], self.layers, self.sites)
        for site in ["q", "pattern"]:
            err = []
            _, st = forward(self.model, self.tokens,
                            [self.steer] + make_restore_hooks(site, self.readers, clean_st, 4, 2, err),
                            self.layers, self.sites)
            for L, hs in by_layer(self.readers).items():
                others = [h for h in range(4) if h not in hs]
                torch.testing.assert_close(st[(L, site)][hs], clean_st[(L, site)][hs], atol=0, rtol=0)
                # non-restored heads in the same layer keep their steered value
                # (layer 1: the steer enters at resid_pre L1, so its q differs
                # from clean; downstream layers differ through the residual)
                self.assertGreater(float((st[(L, site)][others] - clean_st[(L, site)][others]).abs().max()), 1e-7)
            if site == "pattern":
                for L in self.layers:
                    torch.testing.assert_close(st[(L, "pattern")].float().sum(-1),
                                               torch.ones(4), atol=1e-5, rtol=0)

    def test_kv_restoration_acts_on_shared_groups(self):
        self.assertEqual(kv_groups([0, 1], 4, 2), [0])
        self.assertEqual(kv_groups([1, 3], 4, 2), [0, 1])
        self.assertEqual(kv_groups([2], 4, 1), [0])
        self.assertEqual(kv_groups([2], 4, 4), [2])
        _, clean_st = forward(self.model, self.tokens, [], self.layers, self.sites)
        err = []
        _, st = forward(self.model, self.tokens,
                        [self.steer] + make_restore_hooks("k", [(1, 0)], clean_st, 4, 2, err),
                        self.layers, self.sites)
        torch.testing.assert_close(st[(1, "k")][0], clean_st[(1, "k")][0], atol=0, rtol=0)
        self.assertEqual(st[(1, "k")].shape[0], 2)

    def test_controls_layer_matched_and_disjoint(self):
        draws = layer_matched_control_sets(self.readers, 4, 3, seed=1)
        self.assertEqual(len(draws), 3)
        for ctl in draws:
            self.assertEqual({L: len(h) for L, h in by_layer(ctl).items()},
                             {L: len(h) for L, h in by_layer(self.readers).items()})
            self.assertFalse(set(ctl) & set(self.readers))
            self.assertEqual(len(set(ctl)), len(ctl))
        with self.assertRaises(ValueError):
            layer_matched_control_sets([(1, 0), (1, 1), (1, 2)], 4, 1, seed=0)

    def test_run_prompt_rows_and_margin(self):
        controls = layer_matched_control_sets(self.readers, 4, 2, seed=3)
        m_bel = torch.zeros(5, dtype=torch.bool).numpy()
        m_cur = m_bel.copy()
        m_bel[1], m_cur[3] = True, True
        rows, attn, ident = run_prompt(self.model, self.tokens, self.steer, self.readers,
                                       controls, ["q", "pattern"], 4, 2, cid=7, aid=9,
                                       m_bel=m_bel, m_cur=m_cur, applied_norms=self.norms,
                                       do_identity=True)
        self.assertEqual({r["run"] for r in rows}, expected_runs(["q", "pattern"], 2))
        self.assertEqual(len(attn), len(rows) * len(self.readers))
        self.assertEqual(len(ident), 4)
        for r in ident:
            self.assertLess(r["max_abs_logit_diff"], 1e-6)
        by_run = {r["run"]: r for r in rows}
        self.assertEqual(by_run["clean"]["kl_from_clean"], 0.0)
        self.assertGreater(by_run["steered"]["applied_norm"], 0.0)
        self.assertEqual(by_run["transplant_q"]["applied_norm"], 0.0)
        lg, _ = forward(self.model, self.tokens, [], self.layers, self.sites)
        self.assertAlmostEqual(by_run["clean"]["margin"], float(lg[9] - lg[7]), places=6)
        self.assertAlmostEqual(readouts(lg, None, 7, 9)["flip_score"],
                               by_run["clean"]["flip_score"], places=9)
        for r in rows:
            if r["head_set"] != "none":
                self.assertEqual(r["restore_max_err"], 0.0)


if __name__ == "__main__":
    unittest.main()
