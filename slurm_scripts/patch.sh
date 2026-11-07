#!/bin/bash
#SBATCH --job-name=patch_cma      # create a short name for your job
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=1         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1              # number of gpus per node
#SBATCH --time=10:59:00           # total run time limit (HH:MM:SS)
#SBATCH --partition=h200_preemptable
#SBATCH --mem=100GB               # total memory
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-5
#SBATCH --output=logs/patch_cma_%A_%a.out
#SBATCH --error=logs/patch_cma_%A_%a.err

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

# 3 templates x 2 contexts = 6 jobs
TEMPLATES=("library_book" "food_truck" "hair_styling")
CONTEXTS=("abstract" "control")

TEMPLATE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
CONTEXT_IDX=$((SLURM_ARRAY_TASK_ID % 2))

TEMPLATE=${TEMPLATES[$TEMPLATE_IDX]}
CONTEXT=${CONTEXTS[$CONTEXT_IDX]}

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE context=$CONTEXT"

python cma_scripts/tasks/causal_analysis/cma.py \
  --use_template_system \
  --context_type $CONTEXT \
  --base_rule ABA \
  --template_names $TEMPLATE \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --patch_after_movement \
  --run_statistical_tests \
  --n_permutations 1000
