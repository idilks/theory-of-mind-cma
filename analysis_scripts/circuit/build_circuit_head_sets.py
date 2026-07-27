"""Bake the writer and reader head sets for the layered-lookback circuit.

Writers (candidate wire authors, design table row "C1 head set"):
  primary     — T_C1, the prefill-protocol significance-based belief set
                (other_results/ablation/belief_only_sysprompt_prefill/
                head_list.csv, K=32), restricted to layer <= 40. The two
                L46 members (46,10 / 46,13) sit inside the reader band and
                are dropped: a writer downstream of the measurement layer
                is incoherent.
  sensitivity — heads significant in >= half of best-source C1 cells (the
                scatter-pack rule, ~27 heads), same layer restriction.
  NOTE: tom_c1_c5_heads.csv is NOT used here. Its tom_c1 rows actually
  match T_C1 (verified July 24 2026), but the design pins the ablation
  head_list.csv as the source of truth — do not mix lists silently.

Readers (who could read the L40/41 wire):
  - DLA writer band: union over templates of top-20 DLA heads with layer
    in 42..46 (results/rsa_head_outputs/dla/{t}_dla_top20.csv; both signs
    kept — suppressor heads read the wire too).
  - T_C5 heads at layer >= 41 (other_results/ablation/loc_belief_heads).

Outputs (results/aggregated/circuit/ — synced to HPC by sync_to_hpc.sh):
  writer_heads.csv  columns: set, layer, head, mean_score, source
  reader_heads.csv  columns: layer, head, source, dla_mean_logit_diff,
                             tc5_mean_score

Run locally (needs other_results/, which HPC doesn't have).
"""

from __future__ import annotations

import csv as csvmod
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_common import (  # noqa: E402
    AGG_DIR, DLA_DIR, READER_CSV, READER_DLA_LAYERS, READER_TC5_MIN_LAYER,
    REPO, T_C1_PREFILL_CSV, T_C5_CSV, TEMPLATES, WRITER_CSV, WRITER_MAX_LAYER,
)

sys.path.insert(0, str(REPO / "scripts"))
from condition_registry import resolve_best_source  # noqa: E402
from h2_additivity_robustness import CROSS_RULES, RESULTS_ROOT  # noqa: E402


def build_writers() -> pd.DataFrame:
    rows = []

    t_c1 = pd.read_csv(T_C1_PREFILL_CSV)
    assert len(t_c1) == 32, f"expected 32 T_C1 heads, got {len(t_c1)}"
    dropped = t_c1[t_c1["layer"] > WRITER_MAX_LAYER]
    kept = t_c1[t_c1["layer"] <= WRITER_MAX_LAYER]
    print(f"T_C1 primary: {len(kept)} writers (dropped "
          f"{[(int(r.layer), int(r.head)) for r in dropped.itertuples()]} "
          f"as reader-band members)")
    for r in kept.itertuples():
        rows.append(dict(set="primary", layer=int(r.layer), head=int(r.head),
                         mean_score=float(r.mean_score),
                         source="belief_only_sysprompt_prefill/head_list.csv"))

    # sensitivity: >= half-of-cells significance rule (scatter-pack style)
    best = resolve_best_source(RESULTS_ROOT, ["abstract_belief"], CROSS_RULES)
    counts: dict[tuple[int, int], int] = {}
    scores: dict[tuple[int, int], list] = {}
    n_cells = 0
    for (_tmpl, _rule), entry in sorted(best.items()):
        p = Path(entry["csv"])
        if not p.exists():
            continue
        n_cells += 1
        with open(p) as f:
            for row in csvmod.DictReader(f):
                if row["significant"].strip().lower() == "true":
                    key = (int(row["layer"]), int(row["head"]))
                    counts[key] = counts.get(key, 0) + 1
                    scores.setdefault(key, []).append(float(row["observed_score"]))
    need = int(np.ceil(n_cells / 2))
    sens = sorted(k for k, v in counts.items() if v >= need)
    sens_kept = [k for k in sens if k[0] <= WRITER_MAX_LAYER]
    print(f"sensitivity set: {len(sens)} heads sig in >= {need}/{n_cells} C1 "
          f"cells; {len(sens_kept)} after layer<= {WRITER_MAX_LAYER} filter")
    for (layer, head) in sens_kept:
        rows.append(dict(set="sensitivity", layer=layer, head=head,
                         mean_score=float(np.mean(scores[(layer, head)])),
                         source=f"sig>=half of {n_cells} C1 best-source cells"))

    # code-domain writers: the published code_abstract significant set
    # (union of positive sig heads over both directions), layer <= 40.
    code_csv = REPO / "results/aggregated/code_head_sets/code_abstract_head_list.csv"
    code = pd.read_csv(code_csv)
    code_kept = code[code["layer"] <= WRITER_MAX_LAYER]
    dropped = code[code["layer"] > WRITER_MAX_LAYER]
    print(f"code_abstract: {len(code_kept)} writers (dropped "
          f"{[(int(r.layer), int(r.head)) for r in dropped.itertuples()]})")
    for r in code_kept.itertuples():
        rows.append(dict(set="code_abstract", layer=int(r.layer),
                         head=int(r.head), mean_score=float(r.mean_score),
                         source="code_abstract_head_list.csv"))

    return pd.DataFrame(rows)


def build_readers() -> pd.DataFrame:
    lo, hi = READER_DLA_LAYERS
    dla_rows: dict[tuple[int, int], float] = {}
    for template in TEMPLATES:
        df = pd.read_csv(DLA_DIR / f"{template}_dla_top20.csv")
        band = df[(df["layer"] >= lo) & (df["layer"] <= hi)]
        for r in band.itertuples():
            key = (int(r.layer), int(r.head))
            prev = dla_rows.get(key)
            val = float(r.mean_logit_diff)
            if prev is None or abs(val) > abs(prev):
                dla_rows[key] = val

    t_c5 = pd.read_csv(T_C5_CSV)
    tc5_rows = {(int(r.layer), int(r.head)): float(r.mean_score)
                for r in t_c5.itertuples() if r.layer >= READER_TC5_MIN_LAYER}

    keys = sorted(set(dla_rows) | set(tc5_rows))
    rows = []
    for key in keys:
        in_dla, in_tc5 = key in dla_rows, key in tc5_rows
        rows.append(dict(
            layer=key[0], head=key[1],
            source="both" if (in_dla and in_tc5)
                   else ("dla_band" if in_dla else "t_c5_l41plus"),
            dla_mean_logit_diff=dla_rows.get(key, np.nan),
            tc5_mean_score=tc5_rows.get(key, np.nan),
        ))
    df = pd.DataFrame(rows)
    assert (46, 13) in keys, "L46H13 (DLA peak) missing from reader set"
    print(f"readers: {len(df)} heads "
          f"(dla_band={sum(df.source == 'dla_band')}, "
          f"t_c5_l41plus={sum(df.source == 't_c5_l41plus')}, "
          f"both={sum(df.source == 'both')})")
    return df


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    writers = build_writers()
    readers = build_readers()
    writers.to_csv(WRITER_CSV, index=False)
    readers.to_csv(READER_CSV, index=False)
    print(f"wrote {WRITER_CSV}")
    print(f"wrote {READER_CSV}")
    print("\nwriters by set:")
    print(writers.groupby("set").size().to_string())
    print("\nreader heads:")
    print(readers.to_string(index=False))


if __name__ == "__main__":
    main()
