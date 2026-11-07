# SLURM launchers

The batch scripts we ran on our cluster, kept as the record of the exact arguments
behind each result: prompt counts, seeds, time limits, head lists, patch positions.
They are not portable as-is: `PROJECT_DIR`, conda hooks, model caches and partition
names point at our cluster. Treat each one as a worked command line for the script
it calls (`cma_scripts/tasks/causal_analysis/cma.py`, `ablation_experiment.py`, or an
`analysis_scripts/` module) and adapt the header.

| family | scripts |
|---|---|
| CMA cells | `end_patch_seed1.sh`, `patch.sh`, `photo.sh`, `photo_new.sh`, `c5_seed1.sh`, `c7_seed1.sh`, `run_c6_bf.sh`, `missing_conditions.sh`, `resid_*.sh`, `gemma/` |
| code domain | `run_code_cma.sh`, `run_code_abstract_smoke.sh`, `run_code_index_cma.sh`, `run_code_pos_pilot.sh` |
| ablation | `ablation*.sh`, `ablation_runs/` (one file per head set and protocol; see `ablation_runs/README_prefill_rerun.md`) |
| circuit and steering | `circuit_*.sh`, `head_residual_bridge.sh`, `steer_*.sh`, `padtwins_deconfound.sh` |
| path patching | `path_patching.sh`, `path_patching_clean_control.sh`, `path_patching_*_h200_repair.sh` |
| RSA / DLA captures | `rsa_*.sh` |
| attention | `lookback_capture*.sh`, `capture_code_attention_rows.sh`, `prompt_attention_bars.sh`, `region_attention_shares.sh`, `reanalyze_c4_lookback.sh` |

Timing that matters: code CMA at n=150 takes about 8.4 h on an A100, and `cma.py` saves
only after all prompts, so a timeout loses the run. `ablation_experiment.py` checkpoints
per step and resumes.

`../slurm_logs/` holds the six job logs that `analysis_scripts/analyze_head_ablation_interactions.py`
checks before trusting a saved ablation run (model id, dtype, prefill mode).
