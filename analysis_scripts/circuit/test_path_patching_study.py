"""Coverage, matching and restart safety for the full direct-path study."""
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from path_patching_study import build_plan, execute_task, freeze_plan, task_spec

ROOT = Path(__file__).resolve().parents[2]


class StudyTests(unittest.TestCase):
    def plan(self):
        return build_plan(ROOT / "docs/head_evidence_inventory/head_membership.csv",
                          ROOT / "results/aggregated/circuit/reader_heads.csv", seed=20261022)

    def test_full_union_and_one_to_one_same_layer_controls(self):
        plan = self.plan()
        selected = {tuple(p["target"]) for p in plan["sender_pairs"]}
        controls = {tuple(p["control"]) for p in plan["sender_pairs"]}
        self.assertEqual(len(selected), 41)
        self.assertEqual(len(controls), 41)
        self.assertFalse(selected & controls)
        self.assertEqual(sum(p["membership"] == "belief + photo + code"
                             for p in plan["sender_pairs"]), 9)
        self.assertTrue(all(p["target"][0] == p["control"][0] for p in plan["sender_pairs"]))
        self.assertEqual(plan, self.plan())

    def test_every_sender_and_control_in_every_cell_with_forward_paths(self):
        plan = self.plan()
        tasks = [task_spec(plan, i) for i in range(14)]
        self.assertEqual(len({t["cell"]["id"] for t in tasks}), 14)
        self.assertEqual({t["cell"]["base_rule"] for t in tasks}, {"ABA", "ABB"})
        all_heads = sorted(h for p in plan["sender_pairs"] for h in (p["target"], p["control"]))
        for task in tasks:
            senders = sorted(h for g in task["groups"] for h in g["senders"])
            self.assertEqual(senders, all_heads)
            for group in task["groups"]:
                self.assertTrue(all(r[0] > max(l for l, _ in group["senders"]) for r in group["receivers"]))
                if group["scope"] == "exploratory_layer47":
                    self.assertEqual(len(group["receivers"]), 40)
                    self.assertTrue(all(h[0] == 46 for h in group["senders"]))
                else:
                    self.assertEqual(len(group["receivers"]), 26)
        for bad in (-1, 14):
            with self.assertRaises(ValueError):
                task_spec(plan, bad)

    def test_frozen_plan_cannot_be_overwritten_by_different_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan = self.plan()
            freeze_plan(path, plan)
            freeze_plan(path, plan)
            modified = json.loads(json.dumps(plan))
            modified["control_seed"] += 1
            with self.assertRaisesRegex(ValueError, "different"):
                freeze_plan(path, modified)
            self.assertEqual(json.loads(path.read_text()), plan)

    def test_study_guard_written_before_export_and_rejects_mixed_sample_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "plan.json"
            freeze_plan(plan_path, self.plan())
            args = Namespace(plan=str(plan_path), task_id=0, pairs_per_cell=10, batch_size=8,
                             dry_run=False, output_root=str(root / "study"))
            # Stop at the external tokenizer/export boundary; exercise real guards.
            with patch("path_patching_study.export_pairs", side_effect=RuntimeError("export boundary")):
                with self.assertRaisesRegex(RuntimeError, "export boundary"):
                    execute_task(args)
                saved = json.loads((root / "study/study.json").read_text())
                self.assertEqual(saved["pairs_per_cell"], 10)
                args.task_id = 1
                args.pairs_per_cell = 11
                with self.assertRaisesRegex(ValueError, "different frozen settings"):
                    execute_task(args)


if __name__ == "__main__":
    unittest.main()
