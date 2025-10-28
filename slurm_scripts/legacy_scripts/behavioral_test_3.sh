#!/bin/bash
#SBATCH --job-name=tom_behavioral_mini        # create a short name for your job
#SBATCH --nodes=1           # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G         # total memory
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=16:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/behavioral_mini_%j.out
#SBATCH --error=slurm_logs/behavioral_mini_%j.err
#SBATCH --partition=gpuq
#SBATCH --mail-type=BEGIN,END,FAIL

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

python behavioral/behavioral_eval.py \
    --config_type quick_test \
    --prompt_num 10 \
    --vignette_types false_belief \
    --models "Qwen2.5-14B-Instruct" \
    --samples_per_condition 2 \