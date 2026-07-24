#!/bin/bash -l
#SBATCH --job-name=phi4_cma       # C1 grid on microsoft/phi-4 (14B, scale-matched to Qwen)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --array=0-5%4
#SBATCH --mem=60GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/phi4_cma_%A_%a.out
#SBATCH --error=logs/phi4_cma_%A_%a.err

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

# phi-4 C1 grid (July 2026) — submit ONLY after smoke_cma.sh passes.
# abstract_belief x 3 templates x 2 directions = 6 cells, end patch, seed 0.
# Rationale: phi-4 is 14B dense (40L x 40H, GQA 10 KV) — an exact scale match
# to Qwen2.5-14B-Instruct that isolates the model-FAMILY variable
# (Gemma-2-27B varies family and scale simultaneously).
# If C1 replicates the mid-late-layer divergence band, extend to C5 next
# (copy this array with --condition answer_changes_location_only).

JOB_TEMPLATES=(
  food_truck library_book hair_styling
  food_truck library_book hair_styling
)
JOB_RULES=(
  ABA ABA ABA
  ABB ABB ABB
)

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${JOB_TEMPLATES[$IDX]}
RULE=${JOB_RULES[$IDX]}

echo "Job $IDX: condition=abstract_belief template=$TEMPLATE rule=$RULE position=end seed=0"

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition abstract_belief \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --prompt_num 150 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type phi-4 \
  --ungroup_grouped_query_attention \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 5000
