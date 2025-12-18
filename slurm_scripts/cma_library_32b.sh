#!/bin/bash
#SBATCH --job-name=lib_32b        # create a short name for your job
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=1         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1              # number of gpus per node
#SBATCH --time=10:59:00           # total run time limit (HH:MM:SS)
#SBATCH --partition=h200_preemptable
#SBATCH --mem=500GB               # total memory
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-3
#SBATCH --output=logs/lib_32b_%A_%a.out
#SBATCH --error=logs/lib_32b_%A_%a.err

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

# Arrays for different experimental conditions  
RULES=("ABA" "ABA" "ABA" "ABA")
TEMPLATES=("library_book" "food_truck" "food_truck" "food_truck")
CONTEXTS=("abstract" "abstract" "abstract" "abstract") 
ACTIVATIONS=("z" "resid_post" "resid_post" "resid_post")
STYLES=("completion" "completion" "generation" "completion")

python codebase/tasks/identity_rules/cma.py \
  --use_behavioral_tom \
  --context_type ${CONTEXTS[$SLURM_ARRAY_TASK_ID]} \
  --base_rule ${RULES[$SLURM_ARRAY_TASK_ID]} \
  --template_names ${TEMPLATES[$SLURM_ARRAY_TASK_ID]} \
  --prompt_num 20 \
  --max_new_tokens 15 \
  --activation_name ${ACTIVATIONS[$SLURM_ARRAY_TASK_ID]} \
  --model_type Qwen2.5-32B \
  --question_style ${STYLES[$SLURM_ARRAY_TASK_ID]} \
  --samples_per_condition 1