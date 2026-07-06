#!/bin/bash
#SBATCH --job-name=cma_pmove_ft     # post-move vs end residual-stream CMA, food_truck only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-7%5
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/resid_pmove_%A_%a.out
#SBATCH --error=logs/resid_pmove_%A_%a.err

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

# Residual-stream CMA on food_truck only, C3 (answer_changes_belief) and
# C4 (answer_changes_photo). Two patch positions: post-movement (string-anchored
# at "drives the truck to <X>.", same in base and exp; world state matched) and
# end-token (existing behavior, baseline curve). Within-template manipulation:
# base and exp share substitutions, differ only in sentence order (false vs true
# context template). Layer-wise scan across all 48 layers, FWER permutation
# correction at 5000 perms.
#
# Layout (8 jobs):
#   [0] post-move C3 ABA   [1] post-move C3 ABB
#   [2] post-move C4 ABA   [3] post-move C4 ABB
#   [4] end       C3 ABA   [5] end       C3 ABB
#   [6] end       C4 ABA   [7] end       C4 ABB
#
# Output dir layout:
#   results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct/
#     {answer_changes_belief, answer_changes_photo}/template_food_truck/
#       base_rule_{ABA,ABB}_exp_rule_*/
#         {patch_at_world_change, patch_before_movement}/
#           resid_post_seed_1_shuffle_False/

CONDITIONS=(
  "answer_changes_belief" "answer_changes_belief"
  "answer_changes_photo"  "answer_changes_photo"
  "answer_changes_belief" "answer_changes_belief"
  "answer_changes_photo"  "answer_changes_photo"
)

RULES=(
  "ABA" "ABB"
  "ABA" "ABB"
  "ABA" "ABB"
  "ABA" "ABB"
)

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}

# First 4 indices: post-movement patch. Last 4: end-token (no patch flag).
if [ $IDX -lt 4 ]; then
  PATCH_FLAG="--patch_at_world_change"
  PATCH_LABEL="post_move"
else
  PATCH_FLAG=""
  PATCH_LABEL="end"
fi

echo "Job $IDX: condition=$CONDITION template=food_truck rule=$RULE position=$PATCH_LABEL seed=1 prompt_num=150 activation=resid_post"

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names food_truck \
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
