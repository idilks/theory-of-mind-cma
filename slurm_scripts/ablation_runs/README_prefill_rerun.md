# C5-C7 + deconfounding re-runs under the matched-prompt + prefill protocol

Fills the gap flagged in `[[ablation-protocols]]` memory: T_C1 (abstract_belief)
and T_C2 (abstract_photo) were already re-run under the **primary** eval
protocol (`baseline_systemprompt_prefill.sh`, `cross_frame_photo_heads.sh`),
but the C5-C7 controls and the T_C1/T_C5 deconfounding partitions were only
ever ablated under the **original** protocol (neutral system prompt, free
generation, false_belief baseline flip rate ≈0.37 — a known confound). This
batch re-runs those same head sets under the protocol that collapses the
baseline to ≈0.

## Protocol (identical across all new scripts)

- `--system_prompt "You are a helpful assistant reasoning about agents' beliefs."`
  (matched to what CMA used when selecting heads)
- `--assistant_prefill per_variant` (model's first generated token is the
  answer)
- `--seed 1 --prompt_num 50` (target ablation runs) — identical to
  `baseline_systemprompt_prefill.sh` (Run B, T_C1) and
  `cross_frame_photo_heads.sh` (T_C2), so all results land in the same
  comparable table
- `--eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba`
  (full 6-variant set)
- Verified in `cma_scripts/tasks/causal_analysis/ablation_experiment.py:786-798`
  (`--system_prompt` / `--assistant_prefill` argparse definitions) and by
  diffing against `slurm_scripts/ablation_runs/baseline_systemprompt_prefill.sh`
  and `cross_frame_photo_heads.sh`, which already use this exact flag
  combination for T_C1/T_C2.

## Head-set CSVs — pinned, not re-derived

Head sets are pinned to the **exact same CSV + condition/pair/set** the
original-protocol scripts used, so only the eval protocol changes:

| New script | Mirrors | head_source | CSV | condition / pair+set | heads |
|---|---|---|---|---|---|
| `c5_loc_tom_heads_matched_prefill.sh` | (no prior C5-only run) | per_condition | `top20_by_condition_end.csv` | `answer_changes_location_only` | 22 |
| `c6_loc_photo_heads_matched_prefill.sh` | (no prior C6-only run) | per_condition | `top20_by_condition_end.csv` | `answer_changes_location_only_photo` | 13 |
| `c7_state_only_heads_matched_prefill.sh` | (no prior C7-only run) | per_condition | `top20_by_condition_end.csv` | `state_only_no_belief` | 17 |
| `run5_belief_minus_loc_matched_prefill.sh` | `run5_belief_minus_loc.sh` | overlap_set | `condition_overlap_heads_end.csv` | pair `abstract_belief,answer_changes_location_only`, set `a_only` | 21 |
| `run6_loc_minus_belief_matched_prefill.sh` | `run6_loc_minus_belief.sh` | overlap_set | `condition_overlap_heads_end.csv` | pair `abstract_belief,answer_changes_location_only`, set `b_only` | 11 |
| `run7_shared_belief_loc_matched_prefill.sh` | `run7_shared_belief_loc.sh` | overlap_set | `condition_overlap_heads_end.csv` | pair `abstract_belief,answer_changes_location_only`, set `intersection` | 11 |

All CSV paths are `results/lookback_analysis/_summary/<file>` (same
convention as `run1_belief.sh`...`run7_shared_belief_loc.sh`).

**CSV caveat (please read before submitting):** the copy of these CSVs
currently synced to this machine under `results/lookback_analysis/_summary/`
is a stale/partial pull — it only contains 4 conditions
(`abstract_belief`, `abstract_photo`, `answer_changes_belief`,
`answer_changes_photo`) and no `answer_changes_location_only*` /
`state_only_no_belief` rows or pairs. I did **not** invent these paths: I
confirmed the condition names, the pair, and the exact head counts above
(22/13/17/21/11/11) against the fuller copy at
`results/lookback_analysis/_summary/{top20_by_condition_end.csv,
condition_overlap_heads_end.csv}`, which is a more complete pull (per the
project's `results/ vs results/` split note). Since `run5_belief_minus_loc.sh`
/ `run6_loc_minus_belief.sh` / `run7_shared_belief_loc.sh` already reference
`results/lookback_analysis/_summary/condition_overlap_heads_end.csv` under
this same relative path and are documented as having actually run
successfully on HPC, the canonical HPC copy of `results/` should already
contain the full CSVs — but **confirm the row/head counts on HPC** (e.g.
`grep -c answer_changes_location_only results/lookback_analysis/_summary/top20_by_condition_end.csv`)
before trusting `head_list.csv` from any of the three C5/C6/C7 analysis_scripts. If
the full CSVs are missing on HPC, they'll need to be rebuilt there or scp'd
over from wherever `results/lookback_analysis/_summary/` was produced.

## What each script does

**Target ablation runs** (cumulative zero-out + eval, `ablation_experiment.py`):
- `c5_loc_tom_heads_matched_prefill.sh` — C5 control set (loc, ToM frame)
- `c6_loc_photo_heads_matched_prefill.sh` — C6 control set (loc, photo frame)
- `c7_state_only_heads_matched_prefill.sh` — C7 control set (location
  retrieval, no belief)
- `run5_belief_minus_loc_matched_prefill.sh` — T_C1 \ T_C5, belief-unique (21
  heads) — **highest priority**
- `run6_loc_minus_belief_matched_prefill.sh` — T_C5 \ T_C1, loc-unique (11
  heads) — **highest priority**
- `run7_shared_belief_loc_matched_prefill.sh` — T_C1 ∩ T_C5, shared (11
  heads) — **highest priority**

Each writes `results/ablation/<run_label>/` containing `head_list.csv`,
`causal_scores`/cumulative-ablation curves, and the ablation plot (same
output shape as existing `results/ablation/*` runs).

**Random-baseline nulls** (`analysis_scripts/random_baseline_ablation.py`, layer-matched
random head sets, same eval protocol as the paired target run):
- `random_baseline_c5_loc_tom_matched_prefill.sh` (k_eval 11, 22)
- `random_baseline_c6_loc_photo_matched_prefill.sh` (k_eval 11, 13)
- `random_baseline_c7_state_only_matched_prefill.sh` (k_eval 11, 17)
- `random_baseline_belief_minus_loc_matched_prefill.sh` (k_eval 11, 21) —
  companion to the existing original-protocol
  `random_baseline_belief_minus_loc.sh`, same k values so the two protocols'
  nulls sit on the same x-axis
- `random_baseline_loc_minus_belief_matched_prefill.sh` (k_eval 11) — no
  original-protocol null previously existed for this partition; this fills
  both gaps
- `random_baseline_shared_belief_loc_matched_prefill.sh` (k_eval 11) — same,
  no prior null existed for this partition

Each writes `results/ablation/random_baselines/<name>/null_summary.csv`
(mirrors `random_baseline_belief_only_matched_prefill.sh`'s output shape).
Random baselines use `--eval_variants false_belief` only (cheapest,
headline metric) and `--seed 42` for the random-draw RNG, matching every
existing `random_baseline_*.sh` script's convention — this is intentionally
a different seed from the target run's `--seed 1` (prompt sampling), exactly
as in `random_baseline_belief_only_matched_prefill.sh` vs
`baseline_systemprompt_prefill.sh`. If a null on `true_belief`/`true_photo`
is also wanted, add a `_true` companion following
`random_baseline_belief_only_matched_prefill_true.sh`'s pattern (not created
here to avoid speculative scope).

## Submission order

Each random-baseline script reads its paired target run's `head_list.csv`,
so submit target runs first (or chain with `--dependency=afterok:<jobid>`).
Nothing here has been submitted.

```bash
cd /dartfs/rc/lab/F/FranklandS/tom   # or wherever this repo lives on HPC

# 1. Deconfounding partitions (highest priority)
jid5=$(sbatch --parsable slurm_scripts/ablation_runs/run5_belief_minus_loc_matched_prefill.sh)
jid6=$(sbatch --parsable slurm_scripts/ablation_runs/run6_loc_minus_belief_matched_prefill.sh)
jid7=$(sbatch --parsable slurm_scripts/ablation_runs/run7_shared_belief_loc_matched_prefill.sh)

sbatch --dependency=afterok:$jid5 slurm_scripts/ablation_runs/random_baseline_belief_minus_loc_matched_prefill.sh
sbatch --dependency=afterok:$jid6 slurm_scripts/ablation_runs/random_baseline_loc_minus_belief_matched_prefill.sh
sbatch --dependency=afterok:$jid7 slurm_scripts/ablation_runs/random_baseline_shared_belief_loc_matched_prefill.sh

# 2. C5-C7 single-condition control sets
jidc5=$(sbatch --parsable slurm_scripts/ablation_runs/c5_loc_tom_heads_matched_prefill.sh)
jidc6=$(sbatch --parsable slurm_scripts/ablation_runs/c6_loc_photo_heads_matched_prefill.sh)
jidc7=$(sbatch --parsable slurm_scripts/ablation_runs/c7_state_only_heads_matched_prefill.sh)

sbatch --dependency=afterok:$jidc5 slurm_scripts/ablation_runs/random_baseline_c5_loc_tom_matched_prefill.sh
sbatch --dependency=afterok:$jidc6 slurm_scripts/ablation_runs/random_baseline_c6_loc_photo_matched_prefill.sh
sbatch --dependency=afterok:$jidc7 slurm_scripts/ablation_runs/random_baseline_c7_state_only_matched_prefill.sh
```

Each `sbatch` above requires the corresponding CSV row-count check (see CSV
caveat) to pass first — do not submit until that's confirmed on HPC.

## Expected outputs

```
results/ablation/loc_tom_heads_matched_prefill/{head_list.csv, ...}
results/ablation/loc_photo_heads_matched_prefill/{head_list.csv, ...}
results/ablation/state_only_heads_matched_prefill/{head_list.csv, ...}
results/ablation/belief_minus_loc_matched_prefill/{head_list.csv, ...}
results/ablation/loc_minus_belief_matched_prefill/{head_list.csv, ...}
results/ablation/shared_belief_loc_matched_prefill/{head_list.csv, ...}
results/ablation/random_baselines/loc_tom_heads_matched_prefill/null_summary.csv
results/ablation/random_baselines/loc_photo_heads_matched_prefill/null_summary.csv
results/ablation/random_baselines/state_only_heads_matched_prefill/null_summary.csv
results/ablation/random_baselines/belief_minus_loc_matched_prefill/null_summary.csv
results/ablation/random_baselines/loc_minus_belief_matched_prefill/null_summary.csv
results/ablation/random_baselines/shared_belief_loc_matched_prefill/null_summary.csv
```

Together with the existing `belief_only_sysprompt_prefill/` (T_C1) and
`photo_heads_sysprompt_prefill/` (T_C2), these 6 target runs complete the
matched-prompt+prefill row for every cell of the thesis deconfounding table
(T_C1, T_C2, T_C5, T_C6, T_C7, and the T_C1×T_C5 partition).
