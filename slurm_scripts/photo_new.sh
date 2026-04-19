#!/bin/bash
#SBATCH --job-name=photo_new_cma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:59:00
#SBATCH --partition=h200_preemptable
#SBATCH --mem=200GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-11
#SBATCH --output=logs/photo_new_cma_%A_%a.out
#SBATCH --error=logs/photo_new_cma_%A_%a.err

# Photo CMA patching at the PHOTO-TAKING EVENT (epistemic access moment)
# instead of the movement event.
#
# Only runs bf (patch_after_movement) — end position (-1) is identical
# to the original abstract_photo / answer_changes_photo runs.
#
# 12 jobs = 2 conditions x 2 directions x 3 templates
#   0-2:   abstract_photo_new,         ABA (primary), bf
#   3-5:   abstract_photo_new,         ABB (reverse),  bf
#   6-8:   answer_changes_photo_new,   ABA (primary), bf
#   9-11:  answer_changes_photo_new,   ABB (reverse),  bf

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

TEMPLATES=("food_truck" "library_book" "hair_styling")

TEMPLATE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
GROUP=$((SLURM_ARRAY_TASK_ID / 3))

case $GROUP in
  0) CONDITION="abstract_photo_new";         BASE_RULE="ABA" ;;
  1) CONDITION="abstract_photo_new";         BASE_RULE="ABB" ;;
  2) CONDITION="answer_changes_photo_new";   BASE_RULE="ABA" ;;
  3) CONDITION="answer_changes_photo_new";   BASE_RULE="ABB" ;;
esac

echo "Job $SLURM_ARRAY_TASK_ID: condition=$CONDITION template=${TEMPLATES[$TEMPLATE_IDX]} base_rule=$BASE_RULE"

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $BASE_RULE \
  --template_names ${TEMPLATES[$TEMPLATE_IDX]} \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000 \
  --patch_after_movement
