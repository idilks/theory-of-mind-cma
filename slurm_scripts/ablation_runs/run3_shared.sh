#!/bin/bash
#SBATCH --job-name=abl_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:59:00
#SBATCH --partition=h200_preemptable
#SBATCH --mem=200GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_shared_%j.out
#SBATCH --error=logs/abl_shared_%j.err

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

mkdir -p logs

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 42 \
  --n_devices 1 \
  --device_map cuda \
  --head_source overlap_set \
  --head_csv results/lookback_analysis/_summary/condition_overlap_heads_end.csv \
  --pair "abstract_belief,abstract_photo" \
  --set intersection \
  --eval_conditions abstract_belief abstract_photo state_only_no_belief \
  --run_label shared_belief_photo \
  --output_dir results/ablation/shared_belief_photo
