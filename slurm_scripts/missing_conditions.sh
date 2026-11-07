#!/bin/bash
#SBATCH --job-name=cma_missing    # fill the 20-CMA grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-23%2
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

# 8 missing condition/position/direction combos × 3 templates = 24 jobs
#
# Missing cells from the 20-CMA grid:
#   C1 (abstract_belief)        × bf  × ABA   [0-2]
#   C1 (abstract_belief)        × bf  × ABB   [3-5]
#   C3 (answer_changes_belief)  × end × ABA   [6-8]
#   C3 (answer_changes_belief)  × end × ABB   [9-11]
#   C4 (answer_changes_photo)   × bf  × ABA   [12-14]
#   C4 (answer_changes_photo)   × bf  × ABB   [15-17]
#   C4 (answer_changes_photo)   × end × ABA   [18-20]
#   C4 (answer_changes_photo)   × end × ABB   [21-23]

TEMPLATES=("food_truck" "library_book" "hair_styling")

CONDITIONS=(
  "abstract_belief"        "abstract_belief"        "abstract_belief"
  "abstract_belief"        "abstract_belief"        "abstract_belief"
  "answer_changes_belief"  "answer_changes_belief"  "answer_changes_belief"
  "answer_changes_belief"  "answer_changes_belief"  "answer_changes_belief"
  "answer_changes_photo"   "answer_changes_photo"   "answer_changes_photo"
  "answer_changes_photo"   "answer_changes_photo"   "answer_changes_photo"
  "answer_changes_photo"   "answer_changes_photo"   "answer_changes_photo"
  "answer_changes_photo"   "answer_changes_photo"   "answer_changes_photo"
)

RULES=(
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
)

# 1 = use --patch_after_movement (bf position), 0 = end position
PATCH_BF=(
  1 1 1
  1 1 1
  0 0 0
  0 0 0
  1 1 1
  1 1 1
  0 0 0
  0 0 0
)

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${TEMPLATES[$((IDX % 3))]}
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}
USE_BF=${PATCH_BF[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE patch_bf=$USE_BF"

PATCH_FLAG=""
if [ "$USE_BF" -eq 1 ]; then
  PATCH_FLAG="--patch_after_movement"
fi

python cma_scripts/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000 \
  $PATCH_FLAG
