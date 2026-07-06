#!/bin/bash
#SBATCH --job-name=abl_rand_blu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_rand_blu_%j.out
#SBATCH --error=logs/abl_rand_blu_%j.err

# Random layer-matched null for T_C1 \ T_C5 (belief-unique, 21 heads).
# Tests whether the thesis-reported phi=0.72 at k=11 (and 0.73 at k=21) on
# false_belief is above what random layer-matched ablations of the same size
# would produce. Settings match the original thesis ablation run for
# belief_minus_loc (neutral system prompt, no prefill, seed=42, prompt_num=50).

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
  --target_csv results/ablation/belief_minus_loc/head_list.csv \
  --n_random 100 \
  --k_eval 11 21 \
  --eval_variants false_belief \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 42 \
  --n_devices 1 \
  --device_map cuda \
  --output_dir results/ablation/random_baselines/belief_minus_loc
