#!/bin/bash
#SBATCH --job-name=abl_b_inter_loc_mp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_b_inter_loc_mp_%j.out
#SBATCH --error=logs/abl_b_inter_loc_mp_%j.err

# Matched-prompt + answer-prefill re-run of run7_shared_belief_loc.sh.
# T_C1 ∩ T_C5 = 11 heads in BOTH abstract_belief and
# answer_changes_location_only. Highest-priority deconfounding partition:
# these are the heads driving the C1/C5 overlap concern. Re-run under the
# primary protocol so whichever share of the effect these heads carry isn't
# attributable to the ~0.37 free-generation baseline confound.
#
# Pair/CSV/set pinned to exactly what run7_shared_belief_loc.sh used
# (condition_overlap_heads_end.csv, pair "abstract_belief,
# answer_changes_location_only", set intersection) -- only the eval protocol
# changes. Verified via results/lookback_analysis/_summary/
# condition_overlap_heads_end.csv that this pair+set yields 11 heads,
# matching run7's original head count.
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

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
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
  --set intersection \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label shared_belief_loc_matched_prefill \
  --output_dir results/ablation/shared_belief_loc_matched_prefill
