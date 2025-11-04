#!/bin/bash
#SBATCH --job-name=original        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=5:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/original%j.out
#SBATCH --error=slurm_logs/original%j.err
#SBATCH --partition=h200_preemptable
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-3
#SBATCH --output=logs/cma_%A_%a.out

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



RULES=("ABA" "ABB" "ABA" "ABB")
TEMPLATES=("food_truck" "food_truck" "basic_object_move_detailed" "basic_object_move_detailed")

python codebase/tasks/identity_rules/cma.py \
  --use_behavioral_tom \
  --context_type abstract \
  --base_rule ${RULES[$SLURM_ARRAY_TASK_ID]} \
  --template_names ${TEMPLATES[$SLURM_ARRAY_TASK_ID]} \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --better_cma 
