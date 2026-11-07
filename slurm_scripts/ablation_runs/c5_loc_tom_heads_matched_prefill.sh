#!/bin/bash
#SBATCH --job-name=abl_c5_loc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_c5_loc_%j.out
#SBATCH --error=logs/abl_c5_loc_%j.err

# Matched-prompt + answer-prefill re-run of the C5 control head set (control,
# loc / ToM frame). Condition name in the lookback CSVs is
# "answer_changes_location_only" (22 heads, n_templates==3).
#
# Per project_ablation_protocols memory: C5-C7 have so far only been ablated
# under the ORIGINAL protocol (neutral sysprompt, free generation, baseline
# false_belief flip rate ~0.37 confound). This re-runs C5 under the PRIMARY
# protocol (matched sysprompt + per_variant prefill, baseline collapses to
# ~0), using the identical seed/prompt_num/eval_variants as
# baseline_systemprompt_prefill.sh (Run B, T_C1) and cross_frame_photo_heads.sh
# (T_C2), so all three land in the same comparable table.
#
# Head-set CSV/condition name is pinned to what run1_belief.sh / run2_photo.sh
# used for single-condition head sets (top20_by_condition_end.csv,
# rank_by max_observed_score) -- only the eval protocol changes here, not
# which heads get ablated.
#
# CSV caveat: the locally-synced results/lookback_analysis/_summary/
# top20_by_condition_end.csv only carries 4 conditions (abstract_belief,
# abstract_photo, answer_changes_belief, answer_changes_photo) -- a stale/
# partial local pull. The condition "answer_changes_location_only" (22 rows)
# was verified to exist in the fuller results/lookback_analysis/_summary/
# top20_by_condition_end.csv copy pulled from HPC. run5/6/7 already pin the
# sibling condition_overlap_heads_end.csv under this same results/ path and
# run successfully on HPC, so the canonical HPC results/ tree should have the
# full CSV -- but confirm row count for this condition on HPC before trusting
# head_list.csv from this run.

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

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv results/lookback_analysis/_summary/top20_by_condition_end.csv \
  --condition answer_changes_location_only \
  --rank_by max_observed_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label loc_tom_heads_matched_prefill \
  --output_dir results/ablation/loc_tom_heads_matched_prefill
