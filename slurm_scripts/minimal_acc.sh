#!/bin/bash
#SBATCH --job-name=tom_acc        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=60G         # total memory
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=1:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/acc_%j.out
#SBATCH --error=slurm_logs/acc_%j.err
#SBATCH --partition=gpuq
#SBATCH --mail-type=BEGIN,END,FAIL

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis
cd codebase/tasks/identity_rules

## 1. eval generation
python cma.py --generate --use_tom_prompts --ask_world --context_type abstract --base_rule ABA --n_devices 1 --min_valid_sample_num 1 --max_new_tokens 40 --sample_size 10 --verbose --temperature 0.3 --low_prob_threshold 0.8 --activation_name resid_post

