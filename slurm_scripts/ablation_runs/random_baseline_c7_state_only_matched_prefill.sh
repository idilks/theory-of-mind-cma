#!/bin/bash
#SBATCH --job-name=abl_rand_c7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_rand_c7_%j.out
#SBATCH --error=logs/abl_rand_c7_%j.err

# Random layer-matched null for the C7 (location retrieval control, no
# belief) head set (state_only_no_belief, 17 heads), under the SAME eval
# protocol as c7_state_only_heads_matched_prefill.sh (matched sysprompt +
# per_variant prefill). Mirrors random_baseline_belief_only_matched_prefill.sh's
# convention (--seed 42 for the random-draw RNG, distinct from the target
# run's --seed 1 for prompt sampling).
#
# k_eval = 11 (the recurring T_C1 ∩ T_C5 intersection-size checkpoint used
# throughout this deconfounding table) and 17 (the full C7 set).
#
# MUST run after c7_state_only_heads_matched_prefill.sh completes --
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

python analysis_scripts/random_baseline_ablation.py \
  --target_csv results/ablation/state_only_heads_matched_prefill/head_list.csv \
  --n_random 100 \
  --k_eval 11 17 \
  --eval_variants false_belief \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --seed 42 \
  --n_devices 1 \
  --device_map cuda \
  --output_dir results/ablation/random_baselines/state_only_heads_matched_prefill
