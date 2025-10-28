#!/bin/bash
#SBATCH --job-name=tom_cma         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=60G         # total memory
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=1:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/minimal_%j.out
#SBATCH --error=slurm_logs/minimal_%j.err
#SBATCH --partition=gpuq
#SBATCH --mail-type=BEGIN,END,FAIL


source /optnfs/common/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"


cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis
cd codebase/tasks/identity_rules

sample_size=4
prompt_num=50
low_prob_threshold=0.65
max_new_tokens=40
activation_name='resid_post'
model_type='Qwen2.5-14B-Instruct'
base_rule='ABA'
context_type='abstract'


## 1. eval generation
# python cma.py --use_tom_prompts --ask_world --context_type abstract --base_rule ABA --n_devices 1 --min_valid_sample_num 1 --max_new_tokens 40 --sample_size 10 --verbose --temperature 0.3 --low_prob_threshold 0.8



python cma.py --use_tom_prompts --ask_world --base_rule $base_rule --context_type $context_type --model_type $model_type --activation_name $activation_name --prompt_num $prompt_num --low_prob_threshold $low_prob_threshold --max_new_tokens $max_new_tokens --sample_size $sample_size --n_devices 1