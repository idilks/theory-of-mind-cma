"""Checks for the significance-count head selection (no model, saved data only).

Run:  python scripts/rsa_head_outputs/test_head_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import dims  # noqa: E402
import head_sets_source as hss  # noqa: E402
from build_significance_head_sets import derived_sets, select  # noqa: E402

FP_DIR = ROOT / "results/aggregated/fingerprints"


def test_select_threshold_is_inclusive():
    counts = np.array([0, 1, 2, 3, 4, 5, 6])
    assert select(counts, 4).tolist() == [4, 5, 6]
    assert select(counts, 3).tolist() == [3, 4, 5, 6]
    assert select(counts, 7).tolist() == []


def test_derived_sets_are_set_algebra():
    by = {"C1_abstract_belief": {1, 2, 3}, "C5_answer_changes_location_only": {3, 4},
          "C2_abstract_photo": {5, 6}, "C6_answer_changes_location_only_photo": {6}}
    out = derived_sets(by)
    assert out == {"div_unique": [1, 2], "div_shared": [3], "photo_C2": [5]}


def test_top32_unchanged():
    """The default selection must still be the paper's fingerprint rule."""
    s = hss.build("Qwen2.5-14B-Instruct", dims.QWEN, FP_DIR)
    fp1 = np.load(FP_DIR / "C1_abstract_belief_clamped.npy")
    fp5 = np.load(FP_DIR / "C5_answer_changes_location_only_clamped.npy")
    t1 = set(np.argsort(fp1)[-32:].tolist())
    t5 = set(np.argsort(fp5)[-22:].tolist())
    assert s["div_unique"] == sorted(t1 - t5) and len(s["div_unique"]) == 22
    assert s["div_shared"] == sorted(t1 & t5) and len(s["div_shared"]) == 10


def test_significance_sets_match_json_and_csv_counts():
    for sel in ("sig4of6", "sig3of6"):
        min_cells = int(sel[3])
        s = hss.build("Qwen2.5-14B-Instruct", dims.QWEN, FP_DIR, selection=sel)
        j = json.loads((hss.SELECTION_DIR / sel / "sets_qwen.json").read_text())
        assert s["div_unique"] == j["div_unique"], sel
        assert s["div_shared"] == j["div_shared"], sel
        assert s["photo_C2"] == j["photo_C2"], sel
        for cond in j["conditions"]:
            t = pd.read_csv(hss.SELECTION_DIR / sel / f"{cond}.csv")
            assert (t.n_sig_cells >= min_cells).all(), (sel, cond)
            assert sorted(t.flat.tolist()) == j["conditions"][cond], (sel, cond)
        # a significance set never contains a head that is significant nowhere
        assert not set(s["div_unique"]) & set(s["div_shared"])
    four = hss.build("Qwen2.5-14B-Instruct", dims.QWEN, FP_DIR, selection="sig4of6")
    three = hss.build("Qwen2.5-14B-Instruct", dims.QWEN, FP_DIR, selection="sig3of6")
    c1_four = set(four["div_unique"]) | set(four["div_shared"])
    c1_three = set(three["div_unique"]) | set(three["div_shared"])
    assert c1_four <= c1_three, "4/6 set must nest inside 3/6 set"


def test_random32_excludes_selected():
    for sel in hss.SELECTIONS:
        s = hss.build("Qwen2.5-14B-Instruct", dims.QWEN, FP_DIR, selection=sel)
        claimed = set(s["div_unique"]) | set(s["div_shared"])
        assert not claimed & set(s["random32"]), sel


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
