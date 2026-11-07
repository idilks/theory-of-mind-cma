#!/bin/bash -l
#SBATCH --job-name=gemma_cma      # remaining rebuttal cells on Gemma-2-27B-it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --array=0-13%6
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
: "${HF_TOKEN:?HF_TOKEN not set - add it to ~/.bashrc on the HPC (see README setup)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# NeurIPS-rebuttal Gemma grid (July 24 2026). End patch only, seed 0
# (matches all existing Gemma cells). 14 jobs:
#   [0-1]   C2 abstract_photo — the 2 missing reverse cells
#   [2-7]   C3 answer_changes_belief — all 6 (needed for H2 replication)
#   [8-13]  C5 answer_changes_location_only — all 6 (needed for H1+H2)
# C4/C6/C7 deliberately skipped — no headline claim needs them on Gemma.
#
# Already on disk (results/causal_analysis/cma/google/gemma-2-27b-it_ungroup_gqa):
#   abstract_belief  6/6 cells
#   abstract_photo   food_truck ABA+ABB, hair_styling ABA, library_book ABA

JOB_CONDS=(
  abstract_photo abstract_photo
  answer_changes_belief answer_changes_belief answer_changes_belief
  answer_changes_belief answer_changes_belief answer_changes_belief
  answer_changes_location_only answer_changes_location_only answer_changes_location_only
  answer_changes_location_only answer_changes_location_only answer_changes_location_only
)
JOB_TEMPLATES=(
  hair_styling library_book
  food_truck library_book hair_styling
  food_truck library_book hair_styling
  food_truck library_book hair_styling
  food_truck library_book hair_styling
)
JOB_RULES=(
  ABB ABB
  ABA ABA ABA
  ABB ABB ABB
  ABA ABA ABA
  ABB ABB ABB
)

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${JOB_CONDS[$IDX]}
TEMPLATE=${JOB_TEMPLATES[$IDX]}
RULE=${JOB_RULES[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE position=end seed=0"

python cma_scripts/tasks/causal_analysis/cma.py \
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
  --n_permutations 5000
