#!/bin/bash
#SBATCH --job-name=abl_rand_lmb_mp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_rand_lmb_mp_%j.out
#SBATCH --error=logs/abl_rand_lmb_mp_%j.err

# Random layer-matched null for T_C5 \ T_C1 (loc-unique, 11 heads), under the
# matched-prompt + per_variant-prefill protocol. No original-protocol
# companion currently exists for this specific partition (only
# random_baseline_belief_minus_loc.sh was built pre-existing) -- this fills
# both the matched-protocol AND the original-protocol gap for this cell of
# the deconfounding table. k_eval=11 is the full set size (also the
# recurring T_C1 ∩ T_C5 checkpoint used across this table).
#
# MUST run after run6_loc_minus_belief_matched_prefill.sh completes --
# --target_csv reads that run's head_list.csv output.

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

python scripts/random_baseline_ablation.py \
  --target_csv results/ablation/loc_minus_belief_matched_prefill/head_list.csv \
  --n_random 100 \
  --k_eval 11 \
  --eval_variants false_belief \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --seed 42 \
  --n_devices 1 \
  --device_map cuda \
  --output_dir results/ablation/random_baselines/loc_minus_belief_matched_prefill
