#!/bin/bash
#SBATCH --job-name=c5_s1          # C5 control (FB scaffold + location swap), seed=1, end-patch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=21:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-5%6
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/c5_s1_cma_%A_%a.out
#SBATCH --error=logs/c5_s1_cma_%A_%a.err

# C5 control re-run with seed=1 to bump samples beyond the legacy seed=0 n=26-50.
# Canonical condition is answer_changes_location_only (LOCATION_ONLY_MATCHED_BELIEF):
# both prompts false-belief, locations swapped, answer differs.
#
# Output lands at: results/causal_analysis/cma/.../answer_changes_location_only/...
# (NOT in the legacy control_context/ folder — those used a now-deprecated flag path.)
#
# Layout: 6 jobs = 3 templates x 2 base_rules
#   [0-2] base_rule=ABA: food_truck, library_book, hair_styling
#   [3-5] base_rule=ABB: food_truck, library_book, hair_styling

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
RULES=("ABA" "ABA" "ABA" "ABB" "ABB" "ABB")

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${TEMPLATES[$((IDX % 3))]}
RULE=${RULES[$IDX]}

echo "Job $IDX: condition=answer_changes_location_only template=$TEMPLATE rule=$RULE seed=1 prompt_num=150"

# end-patch only (no --patch_after_movement)
python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition answer_changes_location_only \
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
