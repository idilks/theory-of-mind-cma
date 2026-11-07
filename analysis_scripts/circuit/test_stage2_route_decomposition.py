"""Route-decomposition hooks on a tiny transformer; no downloaded model."""
import unittest

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from stage2_steering import make_steer_fn
from stage2_route_decomposition import (
    expected_runs, forward, pin_hooks, route_names, run_prompt,
)
from analyze_route_decomposition import shapley_three


class RouteDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = HookedTransformer(HookedTransformerConfig(
            n_layers=3, d_model=16, n_heads=4, n_key_value_heads=2,
            d_head=4, d_mlp=32, d_vocab=23, n_ctx=16, act_fn="relu",
            normalization_type="RMS", positional_embedding_type="rotary",
            rotary_dim=4, device="cpu", seed=42)).eval()
        cls.tokens = torch.tensor([[1, 2, 3, 4, 5]])
        cls.layers = [1, 2]
        cls.readers = [(1, 0), (2, 1), (2, 3)]
        cls.site = "blocks.1.hook_resid_pre"
        cls.final = "blocks.2.hook_resid_post"
        vec = torch.zeros(16, dtype=torch.float64)
        vec[0] = 1.0
        cls.steer = (cls.site, make_steer_fn("add", 4.0, [], vec=vec, scale=0.5, sign=1.0))

    def test_factorial_pins_are_exact_compositions(self):
        extra = [self.site, self.final]
        _, clean, _ = forward(self.model, self.tokens, [], self.layers, [1, 2], extra)
        steer_lg, steer, _ = forward(self.model, self.tokens, [self.steer], self.layers, [1, 2], extra)
        delta = steer[self.site].double() - clean[self.site].double()
        self.assertGreater(float(delta.norm()), 0)
        # direct only: steered resid_pre, clean attn+mlp
        err = []
        _, scc, _ = forward(self.model, self.tokens,
                            [self.steer] + pin_hooks(route_names("attn", self.layers), clean, err)
                            + pin_hooks(route_names("mlp", self.layers), clean, err),
                            self.layers, [1, 2], extra)
        torch.testing.assert_close(scc[self.final].double(), clean[self.final].double() + delta,
                                   atol=1e-5, rtol=0)
        self.assertEqual(max(err), 0.0)
        # everything but direct: clean resid_pre, steered attn+mlp
        err = []
        _, css, _ = forward(self.model, self.tokens,
                            pin_hooks(route_names("attn", self.layers), steer, err)
                            + pin_hooks(route_names("mlp", self.layers), steer, err),
                            self.layers, [1, 2], extra)
        torch.testing.assert_close(css[self.final].double(), steer[self.final].double() - delta,
                                   atol=1e-5, rtol=0)
        # pinning the steered run to its own outputs reproduces it
        err = []
        lg, sss, _ = forward(self.model, self.tokens,
                             [self.steer] + pin_hooks(route_names("attn", self.layers), steer, err)
                             + pin_hooks(route_names("mlp", self.layers), steer, err),
                             self.layers, [1, 2], extra)
        torch.testing.assert_close(lg, steer_lg, atol=1e-6, rtol=0)

    def test_run_prompt_rows_and_exactness(self):
        m_bel = torch.zeros(5, dtype=torch.bool).numpy()
        m_cur = m_bel.copy()
        m_bel[1], m_cur[3] = True, True
        rows, attn, exact = run_prompt(self.model, self.tokens, self.steer, self.layers,
                                       self.readers, 4, 7, 9, m_bel, m_cur, self.site, self.final)
        self.assertEqual({r["run"] for r in rows}, expected_runs(self.layers))
        self.assertEqual(len(rows), 8 + 2 + 4)
        self.assertEqual(len(attn), len(rows) * len(self.readers))
        self.assertLess(exact["scc_max_abs_err"], 1e-5)
        self.assertGreater(exact["delta_norm"], 0)
        by = {r["run"]: r for r in rows}
        self.assertEqual(by["fac_ccc"]["kl_from_clean"], 0.0)
        for r in rows:
            if r["run"] not in ("fac_ccc", "fac_sss"):
                self.assertEqual(r["pin_max_err"], 0.0)
        # pattern_all restoration must put reader attention back to clean
        clean_att = {(a["layer"], a["head"]): a["mass_believed"] for a in attn if a["run"] == "fac_ccc"}
        pat_att = {(a["layer"], a["head"]): a["mass_believed"] for a in attn if a["run"] == "pattern_all"}
        for k in clean_att:
            self.assertAlmostEqual(clean_att[k], pat_att[k], places=6)

    def test_shapley_sums_to_effect_and_is_exact_for_additive_game(self):
        v = {"ccc": 0.0, "scc": 1.0, "csc": 2.0, "ccs": 4.0, "ssc": 3.0, "scs": 5.0,
             "css": 6.0, "sss": 7.0}
        phi = shapley_three(v)
        self.assertAlmostEqual(phi["direct"], 1.0)
        self.assertAlmostEqual(phi["attn"], 2.0)
        self.assertAlmostEqual(phi["mlp"], 4.0)
        v["sss"] = 10.0  # interaction: shares still sum to the total
        phi = shapley_three(v)
        self.assertAlmostEqual(sum(phi.values()), 10.0)


if __name__ == "__main__":
    unittest.main()
