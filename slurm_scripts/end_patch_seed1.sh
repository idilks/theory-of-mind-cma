#!/bin/bash
#SBATCH --job-name=cma_end_s1     # end-patch reruns, seed=1, 150 prompts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=21:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-29%2
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/end_s1_cma_%A_%a.out
#SBATCH --error=logs/end_s1_cma_%A_%a.err

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

# end-patch reruns with seed=1 to add ~100 new valid prompts to existing seed=0 results.
# 5 conditions x 2 directions x 3 templates = 30 jobs.
# combine afterwards with: python scripts/combine_per_sample_runs.py --seeds 0 1 --patch_position patch_before_movement
#
# layout: every block of 6 = one condition (ABA-ft, ABA-lb, ABA-hs, ABB-ft, ABB-lb, ABB-hs)
#   [0-5]   abstract_belief
#   [6-11]  abstract_photo
#   [12-17] answer_changes_belief
#   [18-23] answer_changes_photo
#   [24-29] state_only_no_belief (no observer/belief/photo, BasicTemplate). End-patch only.

TEMPLATES=("food_truck" "library_book" "hair_styling")

CONDITIONS=(
  "abstract_belief"        "abstract_belief"        "abstract_belief"
  "abstract_belief"        "abstract_belief"        "abstract_belief"
  "abstract_photo"         "abstract_photo"         "abstract_photo"
  "abstract_photo"         "abstract_photo"         "abstract_photo"
  "answer_changes_belief"  "answer_changes_belief"  "answer_changes_belief"
  "answer_changes_belief"  "answer_changes_belief"  "answer_changes_belief"
  "answer_changes_photo"   "answer_changes_photo"   "answer_changes_photo"
  "answer_changes_photo"   "answer_changes_photo"   "answer_changes_photo"
  "state_only_no_belief"   "state_only_no_belief"   "state_only_no_belief"
  "state_only_no_belief"   "state_only_no_belief"   "state_only_no_belief"
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
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
)

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${TEMPLATES[$((IDX % 3))]}
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE seed=1 prompt_num=150"

# no --patch_after_movement => end-patch position
python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --seed 1 \
  --prompt_num 150 \
  --min_valid_sample_num 80 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000
