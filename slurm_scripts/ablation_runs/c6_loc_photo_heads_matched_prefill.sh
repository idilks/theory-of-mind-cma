#!/bin/bash
#SBATCH --job-name=abl_c6_locp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_c6_locp_%j.out
#SBATCH --error=logs/abl_c6_locp_%j.err

# Matched-prompt + answer-prefill re-run of the C6 control head set (control,
# loc / photo frame). Condition name in the lookback CSVs is
# "answer_changes_location_only_photo" (13 heads, n_templates==3).
#
# Same rationale/protocol as c5_loc_tom_heads_matched_prefill.sh: C5-C7 were
# never re-run under the primary (matched sysprompt + per_variant prefill)
# protocol used for T_C1/T_C2. Identical seed=1/prompt_num=50/eval_variants
# to baseline_systemprompt_prefill.sh and cross_frame_photo_heads.sh for
# direct comparability.
#
# Head-set CSV/condition pinned to the same convention run1/run2 used
# (top20_by_condition_end.csv, rank_by max_observed_score) -- only the eval
# protocol changes, not which heads get ablated.
#
# CSV caveat: same as c5_loc_tom_heads_matched_prefill.sh -- the locally
# synced results/lookback_analysis/_summary/top20_by_condition_end.csv is a
# stale partial pull without this condition. Verified "answer_changes_
# location_only_photo" exists with 13 rows in other_results/lookback_analysis/
# _summary/top20_by_condition_end.csv. Confirm the row count on HPC before
# trusting head_list.csv from this run.

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis
export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
mkdir -p logs

python codebase/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv results/lookback_analysis/_summary/top20_by_condition_end.csv \
  --condition answer_changes_location_only_photo \
  --rank_by max_observed_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label loc_photo_heads_matched_prefill \
  --output_dir results/ablation/loc_photo_heads_matched_prefill
