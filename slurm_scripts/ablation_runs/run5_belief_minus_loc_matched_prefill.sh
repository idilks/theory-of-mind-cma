#!/bin/bash
#SBATCH --job-name=abl_b_minus_loc_mp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_b_minus_loc_mp_%j.out
#SBATCH --error=logs/abl_b_minus_loc_mp_%j.err

# Matched-prompt + answer-prefill re-run of run5_belief_minus_loc.sh.
# T_C1 \ T_C5 = abstract_belief heads NOT in answer_changes_location_only
# (21 belief-unique heads). Highest-priority deconfounding partition: under
# the ORIGINAL protocol this set showed phi(false_belief)=0.72 at k=11 and
# 0.73 at k=21 (confounded by the 0.37 free-generation baseline). This
# re-run checks whether the belief-unique effect survives once the baseline
# is pinned near 0 by matched sysprompt + per_variant prefill.
#
# Pair/CSV/set are pinned to exactly what run5_belief_minus_loc.sh used
# (condition_overlap_heads_end.csv, pair "abstract_belief,
# answer_changes_location_only", set a_only) -- only the eval protocol
# changes, not which heads get ablated. Verified via other_results/
# lookback_analysis/_summary/condition_overlap_heads_end.csv (the fuller,
# non-stale local copy of this CSV) that this pair+set yields 21 heads,
# matching run5's original head count.
#
# seed=1, prompt_num=50, full eval_variants set -- identical to
# baseline_systemprompt_prefill.sh (Run B, T_C1) and cross_frame_photo_heads.sh
# (T_C2) so all land in the same comparable deconfounding table.

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
  --head_source overlap_set \
  --head_csv results/lookback_analysis/_summary/condition_overlap_heads_end.csv \
  --pair "abstract_belief,answer_changes_location_only" \
  --set a_only \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label belief_minus_loc_matched_prefill \
  --output_dir results/ablation/belief_minus_loc_matched_prefill
