#!/bin/bash
#SBATCH --job-name=cma_resid       # residual-stream CMA, C1+C2, bf vs end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-23%4
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/resid_cma_%A_%a.out
#SBATCH --error=logs/resid_cma_%A_%a.err

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

# Residual-stream CMA at the bf vs end token positions for C1 (abstract_belief)
# and C2 (abstract_photo). Tests whether belief-veracity / state-tracking signal
# localizes spatially. 48 layers patched per cell, 5000 permutations, seed=1
# matches the canonical 7-condition end-position seed.
#
# Layout: 24 jobs = 2 patch positions x 2 conditions x 2 directions x 3 templates
#   [0-2]   bf  C1 ABA [ft, lb, hs]
#   [3-5]   bf  C1 ABB [ft, lb, hs]
#   [6-8]   bf  C2 ABA [ft, lb, hs]
#   [9-11]  bf  C2 ABB [ft, lb, hs]
#   [12-14] end C1 ABA [ft, lb, hs]
#   [15-17] end C1 ABB [ft, lb, hs]
#   [18-20] end C2 ABA [ft, lb, hs]
#   [21-23] end C2 ABB [ft, lb, hs]
#
# Output dir suffix: resid_post_seed_1_shuffle_False/
# Patch position folder: patch_after_movement (bf) or patch_before_movement (end).

TEMPLATES=("food_truck" "library_book" "hair_styling")

CONDITIONS=(
  "abstract_belief" "abstract_belief" "abstract_belief"
  "abstract_belief" "abstract_belief" "abstract_belief"
  "abstract_photo"  "abstract_photo"  "abstract_photo"
  "abstract_photo"  "abstract_photo"  "abstract_photo"
  "abstract_belief" "abstract_belief" "abstract_belief"
  "abstract_belief" "abstract_belief" "abstract_belief"
  "abstract_photo"  "abstract_photo"  "abstract_photo"
  "abstract_photo"  "abstract_photo"  "abstract_photo"
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

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${TEMPLATES[$((IDX % 3))]}
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}

# First 12 indices: bf-position (--patch_after_movement). Last 12: end-position.
if [ $IDX -lt 12 ]; then
  PATCH_FLAG="--patch_after_movement"
  PATCH_LABEL="bf"
else
  PATCH_FLAG=""
  PATCH_LABEL="end"
fi

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE position=$PATCH_LABEL seed=1 prompt_num=150 activation=resid_post"

python cma_scripts/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --seed 1 \
  --prompt_num 150 \
  --min_valid_sample_num 80 \
  --max_new_tokens 5 \
  --activation_name resid_post \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000 \
  $PATCH_FLAG
