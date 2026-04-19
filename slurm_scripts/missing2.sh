#!/bin/bash
#SBATCH --job-name=cma_missing    # fill the 6-CMA grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-5%2
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/missing_cma_%A_%a.out
#SBATCH --error=logs/missing_cma_%A_%a.err

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

# C3 (answer_changes_belief) × bf × {ABA, ABB} × 3 templates = 6 jobs
# indices 0-2: ABA × [food_truck, library_book, hair_styling]
# indices 3-5: ABB × [food_truck, library_book, hair_styling]

TEMPLATES=("food_truck" "library_book" "hair_styling")

RULES=(
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
)

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${TEMPLATES[$((IDX % 3))]}
RULE=${RULES[$IDX]}

echo "Job $IDX: condition=answer_changes_belief template=$TEMPLATE rule=$RULE patch_bf=1"

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition answer_changes_belief \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000 \
  --patch_after_movement
