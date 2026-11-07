#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:59:00
#SBATCH --partition=h200_preemptable
#SBATCH --mem=200GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

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
  --device_map cuda


python cma_scripts/tasks/causal_analysis/ablation_experiment.py 
  --model_type Qwen2.5-14B-Instruct 
  --prompt_num 50 
  --max_new_tokens 5 
  --templates food_truck hair_styling library_book 
  --seed 42 
  --n_devices 1 
  --device_map cuda