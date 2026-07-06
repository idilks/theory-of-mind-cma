#!/bin/bash -l
#SBATCH --job-name=gemma_cma      # full 20-CMA grid on Gemma-2-27B-it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --array=0-35%6
#SBATCH --mem=100GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/gemma_cma_%A_%a.out
#SBATCH --error=logs/gemma_cma_%A_%a.err

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="***REMOVED_HF_TOKEN***"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# Gemma-2-27B-it CMA grid: 7 conditions × 2 rules × 3 templates = 42 cells,
# minus 6 already completed locally = 36 remaining jobs.
# End patch only (no --patch_after_movement).
# Cluster cap: 6 concurrent jobs (set via %6 in --array directive above).
#
# Completed (skipped here): abstract_belief×{food_truck/ABA, library_book/ABA,
#   library_book/ABB, hair_styling/ABB}; abstract_photo×{food_truck/ABA,
#   library_book/ABA}.

JOB_CONDS=(
  abstract_belief abstract_belief
  abstract_photo abstract_photo abstract_photo abstract_photo
  answer_changes_belief answer_changes_belief answer_changes_belief answer_changes_belief answer_changes_belief answer_changes_belief
  answer_changes_photo answer_changes_photo answer_changes_photo answer_changes_photo answer_changes_photo answer_changes_photo
  state_only_no_belief state_only_no_belief state_only_no_belief state_only_no_belief state_only_no_belief state_only_no_belief
  answer_changes_location_only answer_changes_location_only answer_changes_location_only answer_changes_location_only answer_changes_location_only answer_changes_location_only
  answer_changes_location_only_photo answer_changes_location_only_photo answer_changes_location_only_photo answer_changes_location_only_photo answer_changes_location_only_photo answer_changes_location_only_photo
)
JOB_TEMPLATES=(
  hair_styling food_truck
  hair_styling food_truck library_book hair_styling
  food_truck library_book hair_styling food_truck library_book hair_styling
  food_truck library_book hair_styling food_truck library_book hair_styling
  food_truck library_book hair_styling food_truck library_book hair_styling
  food_truck library_book hair_styling food_truck library_book hair_styling
  food_truck library_book hair_styling food_truck library_book hair_styling
)
JOB_RULES=(
  ABA ABB
  ABA ABB ABB ABB
  ABA ABA ABA ABB ABB ABB
  ABA ABA ABA ABB ABB ABB
  ABA ABA ABA ABB ABB ABB
  ABA ABA ABA ABB ABB ABB
  ABA ABA ABA ABB ABB ABB
)

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${JOB_CONDS[$IDX]}
TEMPLATE=${JOB_TEMPLATES[$IDX]}
RULE=${JOB_RULES[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE position=end"

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --prompt_num 150 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type gemma-2-27b-it \
  --ungroup_grouped_query_attention \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000 \
