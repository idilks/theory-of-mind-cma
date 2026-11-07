"""Head-set construction, parameterised by model.

The Qwen pipeline selects heads by `argsort`-ing 1920-long CMA fingerprint
arrays (`fp_C1` top-32, `fp_C5` top-22). Gemma has no fingerprint arrays on
this machine — its CMA shipped as *significance-thresholded head lists*
(`gemma_c{1,2,3,5}_head_list.csv`, columns layer/head/n_sig_cells/n_cells/
mean_score).

These are NOT the same operation and the difference must be stated wherever
the two models are compared:

  Qwen   top-k by clamped fingerprint mass  -> fixed sizes (32 / 22)
  Gemma  heads passing the CMA significance test -> data-determined sizes
         (C1=21, C2=32, C3=50, C5=37)

So Gemma's div_unique is "C1-significant and not C5-significant" (14 heads),
while Qwen's is "top-32 by fp_C1 minus top-22 by fp_C5" (22 heads). Comparable
in spirit and in density (0.95% vs 1.15% of all heads), not identical in
construction.

Returns flat indices under the model's own head grid (see dims.py) — Qwen
layer*40+head, Gemma layer*32+head. Mixing the two conventions silently
misassigns every head, so the dims object is a required argument.

`selection` (Qwen only) picks the rule:
  "top32"    fingerprint top-32 / top-22 (the paper's original pipeline; default)
  "sig4of6"  FWER-significant in >= 4 of the condition's 6 end-position cells
  "sig3of6"  >= 3 of 6 (sensitivity)
The sig* lists are read from results/aggregated/head_selection/{rule}/, built
by analysis_scripts/build_significance_head_sets.py, so the steering runner and the
figures consume one file rather than re-deriving the rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SELECTIONS = ("top32", "sig4of6", "sig3of6")
SELECTION_DIR = Path(__file__).resolve().parents[2] / "results/aggregated/head_selection"


def _flat_from_csv(path: Path, d) -> set[int]:
    t = pd.read_csv(path)
    assert {"layer", "head"} <= set(t.columns), f"{path} lacks layer/head"
    bad = t[(t.layer >= d.n_layers) | (t["head"] >= d.n_heads)]
    assert bad.empty, (
        f"{path}: {len(bad)} rows exceed the {d} grid — wrong model dims or "
        f"a grouped/ungrouped GQA mismatch. First: {bad.iloc[0].to_dict()}")
    return {d.flat(r.layer, r["head"]) for _, r in t.iterrows()}


def significance_set(cond: str, selection: str, d,
                     sel_dir: Path = SELECTION_DIR) -> set[int]:
    """Flat indices from the prebuilt significance-count CSV for one condition."""
    assert selection in SELECTIONS and selection != "top32", selection
    path = Path(sel_dir) / selection / f"{cond}.csv"
    assert path.exists(), (f"missing {path}; run analysis_scripts/build_significance_head_sets.py "
                           "(and push results/aggregated/ to the HPC)")
    t = pd.read_csv(path)
    flat = _flat_from_csv(path, d)
    assert flat == set(t["flat"].astype(int).tolist()), f"{path}: flat column disagrees with layer/head"
    return flat


def qwen_sets(d, fp_dir: Path, geom_csv: Path | None = None,
              code_sig_csv: Path | None = None, seed: int = 0,
              selection: str = "top32") -> dict:
    assert selection in SELECTIONS, f"unknown selection {selection!r}; choose from {SELECTIONS}"
    fp1 = np.load(fp_dir / "C1_abstract_belief_clamped.npy")
    fp5 = np.load(fp_dir / "C5_answer_changes_location_only_clamped.npy")
    assert len(fp1) == d.nh, f"fingerprint length {len(fp1)} != NH {d.nh}"
    if selection == "top32":
        t1 = set(np.argsort(fp1)[-32:].tolist())
        t5 = set(np.argsort(fp5)[-22:].tolist())
    else:
        t1 = significance_set("C1_abstract_belief", selection, d)
        t5 = significance_set("C5_answer_changes_location_only", selection, d)
    sets = {"div_unique": sorted(t1 - t5), "div_shared": sorted(t1 & t5)}
    if code_sig_csv and Path(code_sig_csv).exists():
        dual = sorted(_flat_from_csv(Path(code_sig_csv), d) & t1)
        if dual:
            sets["dual_sig"] = dual
    if geom_csv and Path(geom_csv).exists():
        g = pd.read_csv(geom_csv)
        sets["address"] = sorted(d.flat(r.layer, r.head)
                                 for r in g.nlargest(32, "queried_slot_rel").itertuples())
    # ORDER MATTERS. The original head_sets() drew random32 from a pool that
    # excluded div_unique/div_shared/dual_sig/address only; photo_C2 was added
    # afterwards by state_geometry_threeway. Adding photo_C2 before the draw
    # shrinks the pool and yields a DIFFERENT random32, silently breaking
    # comparability with every result reported in G/G2/G3/SG4.
    sets = _add_random(sets, d, seed)
    c2p = fp_dir / "C2_abstract_photo_clamped.npy"
    c6p = fp_dir / "C6_answer_changes_location_only_photo_clamped.npy"
    if selection != "top32":
        sets["photo_C2"] = sorted(
            significance_set("C2_abstract_photo", selection, d)
            - significance_set("C6_answer_changes_location_only_photo", selection, d))
    elif c2p.exists() and c6p.exists():
        f2, f6 = np.load(c2p), np.load(c6p)
        sets["photo_C2"] = sorted(set(np.argsort(f2)[-26:].tolist())
                                  - set(np.argsort(f6)[-13:].tolist()))
    return sets


def gemma_sets(d, gemma_dir: Path, seed: int = 0) -> dict:
    """From significance-thresholded CMA head lists (no fingerprint arrays)."""
    need = {t: Path(gemma_dir) / f"gemma_{t}_head_list.csv"
            for t in ("c1", "c2", "c5")}
    missing = [str(p) for p in need.values() if not p.exists()]
    assert not missing, f"missing Gemma head lists: {missing}"
    c1 = _flat_from_csv(need["c1"], d)
    c2 = _flat_from_csv(need["c2"], d)
    c5 = _flat_from_csv(need["c5"], d)
    # photo_C2: Qwen uses C2-top26 minus C6-top13 (photo-state minus photo-
    # retrieval). Gemma has no C6 list, so the nearest analog is C2 minus C5
    # (photo-state minus retrieval). Same intent, different construction —
    # flag it in any Qwen/Gemma comparison of this set specifically.
    sets = {"div_unique": sorted(c1 - c5),
            "div_shared": sorted(c1 & c5),
            "photo_C2": sorted(c2 - c5)}
    # no `address` set: it derives from the code-position RSA, and no Gemma
    # code-domain CMA exists. Callers must tolerate its absence.
    return _add_random(sets, d, seed)


def _add_random(sets: dict, d, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    claimed = set().union(*[set(v) for v in sets.values()]) if sets else set()
    pool = [h for h in range(d.nh) if h not in claimed]
    k = min(32, len(pool))
    sets["random32"] = sorted(rng.choice(pool, k, replace=False).tolist())
    return sets


def build(model_type: str, d, fp_dir: Path, gemma_dir: Path | None = None,
          geom_csv: Path | None = None, code_sig_csv: Path | None = None,
          seed: int = 0, selection: str = "top32") -> dict:
    if "gemma" in model_type.lower():
        assert gemma_dir is not None, "gemma_dir required for Gemma head sets"
        assert selection == "top32", "Gemma sets are already significance-selected; no selection switch"
        return gemma_sets(d, gemma_dir, seed)
    return qwen_sets(d, fp_dir, geom_csv, code_sig_csv, seed, selection)
