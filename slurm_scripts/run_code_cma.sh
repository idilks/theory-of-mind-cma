#!/bin/bash
#SBATCH --job-name=cma_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-11%2
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/code_cma_%A_%a.out
#SBATCH --error=logs/code_cma_%A_%a.err

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

# Code-condition CMA, end-patch only.
# 2 conditions x 2 directions x 3 templates = 12 jobs.
#   [0-2]   code_answer_changes  ABA  (list_append, dict_update, list_extend)
#   [3-5]   code_answer_changes  ABB
#   [6-8]   code_control         ABA
#   [9-11]  code_control         ABB

TEMPLATES=("list_append" "dict_update" "list_extend")

CONDITIONS=(
  "code_answer_changes" "code_answer_changes" "code_answer_changes"
  "code_answer_changes" "code_answer_changes" "code_answer_changes"
  "code_control"        "code_control"        "code_control"
  "code_control"        "code_control"        "code_control"
)

RULES=(
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
  "ABA" "ABA" "ABA"
  "ABB" "ABB" "ABB"
)

IDX=$SLURM_ARRAY_TASK_ID
TEMPLATE=${TEMPLATES[$((IDX % 3))]}
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE seed=0 prompt_num=50"

# no --patch_after_movement => end-patch position (-1)
python codebase/tasks/causal_analysis/cma.py \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --seed 0 \
  --prompt_num 50 \
  --min_valid_sample_num 10 \
  --max_new_tokens 30 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --run_statistical_tests \
  --n_permutations 5000
