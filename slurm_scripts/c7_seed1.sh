#!/bin/bash
#SBATCH --job-name=c7_s1          # C7 photo control (matched false-photo, location swap), seed=1, end-patch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=21:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-5%6
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/c7_s1_cma_%A_%a.out
#SBATCH --error=logs/c7_s1_cma_%A_%a.err

# C7 photo control — analogue of C8 (LOCATION_ONLY_MATCHED_BELIEF) on the photo side.
# Both prompts use false-photo context, locations swapped, answer differs.
# Adds the photo-side cell needed for the H1 / decomposition tests:
# tests whether r(C3, C1+C5) ~= r(C3, C2+C7) etc., and rounds out the design.
#
# Within-context patch: ABA->ABA and ABB->ABB (no false<->true flip).
#
# Output lands at: results/causal_analysis/cma/.../answer_changes_location_only_photo/...
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

echo "Job $IDX: condition=answer_changes_location_only_photo template=$TEMPLATE rule=$RULE seed=1 prompt_num=150"

# end-patch only (no --patch_after_movement)
python cma_scripts/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition answer_changes_location_only_photo \
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
