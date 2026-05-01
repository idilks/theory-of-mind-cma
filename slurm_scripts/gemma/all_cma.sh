#!/bin/bash -l
#SBATCH --job-name=gemma_cma      # full 20-CMA grid on Gemma-2-27B-it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:59:00
#SBATCH --partition=h200_preemptable
# Note: H200-only (141GB). A100-40/48GB partitions OOM on Gemma-2-27B in bf16
# (model weights are ~54GB and shared GPUs leave <40GB free). See diag from
# job 8352412 where tasks 2-5 OOM'd while tasks 0-1 succeeded only because they
# landed on H200s. Slower queue, no OOM.
#SBATCH --array=0-29%6
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

# Full 20-CMA grid on Gemma-2-27B-it:
#   5 conditions × 2 directions (ABA/ABB) x 3 templates = 30 jobs
#
# Cluster cap: 6 concurrent jobs (set via %6 in --array directive above).
#
# Launch options:
#   sbatch --array=0-2  slurm_scripts/gemma/all_cma.sh   # smoke (3 jobs):
#                                                          abstract_belief × ABA × {food_truck,library,hair}
#   sbatch --array=0-5  slurm_scripts/gemma/all_cma.sh   # overnight (6 jobs):
#                                                          abstract_belief × {ABA,ABB} × all templates
#   sbatch              slurm_scripts/gemma/all_cma.sh   # full grid (30 jobs, 6 at a time)
#
# Verify after smoke: heatmap shape (46, 32), no OOM, scores finite.

TEMPLATES=("food_truck" "library_book" "hair_styling")
CONDITIONS=("abstract_belief" "abstract_photo" "answer_changes_belief" "answer_changes_photo" "control")
RULES=("ABA" "ABB")

IDX=$SLURM_ARRAY_TASK_ID

# Index decomposition: 30 = 5 conditions * 2 rules * 3 templates
# Iterate fastest over templates so a smoke test of --array=0-2 covers all 3 templates
# of the first (condition, rule) cell = (abstract_belief, ABA).
# End patch is the default (no --patch_after_movement flag).
TEMPLATE_IDX=$((IDX % 3))
RULE_IDX=$(( (IDX / 3) % 2 ))
COND_IDX=$(( IDX / 6 ))

TEMPLATE=${TEMPLATES[$TEMPLATE_IDX]}
CONDITION=${CONDITIONS[$COND_IDX]}
RULE=${RULES[$RULE_IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE position=end"

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --prompt_num 200 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type gemma-2-27b-it \
  --ungroup_grouped_query_attention \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000 \
