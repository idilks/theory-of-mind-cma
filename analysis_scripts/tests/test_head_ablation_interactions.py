"""Small CPU checks for pairing, endpoint alignment, and resampling units."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_head_ablation_interactions import (  # noqa: E402
    bootstrap_means, contrasts, endpoint_values, template_of, verify_pairing,
)


class AblationInteractionsTest(unittest.TestCase):
    def test_pairing_rejects_reordered_or_modified_answer_records(self):
        for other in (["b", "a"], ["a", "changed answer"]):
            with self.assertRaises(ValueError):
                verify_pairing([["a", "b"], other])

    def test_contrast_uses_each_runs_own_baseline(self):
        full = np.array([[0.9, 0.3]])
        shared = np.array([[0.8, 0.4]])
        remainder = np.array([[0.7, 0.6]])
        np.testing.assert_allclose(contrasts(full, shared, remainder), [[0.6, 0.4, 0.1, 0.2, 0.1]])
        with self.assertRaises(ValueError):
            contrasts(full, shared, np.ones((2, 2)))

    def test_bootstrap_preserves_template_balance_and_metric_pairing(self):
        # Unequal group sizes should still get equal weight, not pooled weight.
        values = np.array([[1., 2.], [1., 2.], [3., 6.]])
        labels = np.array(["a", "a", "b"])
        actual = bootstrap_means(values, labels, 100, np.random.default_rng(1))
        np.testing.assert_allclose(actual, np.tile([2., 4.], (100, 1)))

    def test_endpoint_rejects_noncontiguous_grid_and_summary_mismatch(self):
        payload = {"target_heads": [(1, 2)], "target": {"v": {"p_correct": np.array([[0.9, 0.4], [0.7, 0.2]])}}}
        summary = pd.DataFrame({"k": [0, 1], "curve": ["target"] * 2,
                                "eval_variant": ["v"] * 2, "mean_p_correct": [0.8, 0.3]})
        np.testing.assert_allclose(endpoint_values(payload, summary, "v"), [[0.9, 0.4], [0.7, 0.2]])
        summary.loc[1, "k"] = 2
        with self.assertRaises(ValueError):
            endpoint_values(payload, summary, "v")
        summary.loc[1, "k"] = 1
        summary.loc[1, "mean_p_correct"] = 0.4
        with self.assertRaises(ValueError):
            endpoint_values(payload, summary, "v")

    def test_template_uses_only_user_content_and_fails_ambiguous(self):
        self.assertEqual(template_of("book\n<|im_start|>user\nA food truck moves.<|im_end|>"), "food_truck")
        with self.assertRaises(ValueError):
            template_of("<|im_start|>user\nA book in a food truck.<|im_end|>")


if __name__ == "__main__":
    unittest.main()
