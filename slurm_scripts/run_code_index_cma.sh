#!/bin/bash
#SBATCH --job-name=cma_code_index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-14%15
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/code_index_cma_%A_%a.out
#SBATCH --error=logs/code_index_cma_%A_%a.err

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"

# wandb: uses WANDB_API_KEY from the submitting environment if present
# (sbatch exports it by default); otherwise runs with --no_wandb.
WANDB_FLAG=""
if [ -z "$WANDB_API_KEY" ]; then WANDB_FLAG="--no_wandb"; fi

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# Index-vs-veracity dissociation (Sept 10 2026 meeting, item 2). End-patch, z.
# Seed 0 = DISCOVERY (defines the index head set), seed 1 = HELD-OUT evaluation
# of all frozen head sets (scripts/analyze_code_index_dissociation.py).
# code_abstract seed 0 already exists (the discovery run for its head list).
# ABA = copy side as base, ABB = alias side as base; code_index* keep the kind
# on both sides (within_context), code_abstract* flip it.
#   [0-1]  code_index             seed 0   ABA, ABB   discovery
#   [2-3]  code_index             seed 1   ABA, ABB   held-out
#   [4-5]  code_index_flip        seed 1   ABA, ABB   held-out (key vs line position; alias only is diagnostic)
#   [6-7]  code_abstract_two_mut  seed 1   ABA, ABB   held-out, format-matched copy/alias contrast
#   [8-9]  code_abstract          seed 1   ABA, ABB   held-out, original format
#   [10-11] code_index            seed 1   ABA, ABB   resid_post layer sweep = POSITIVE CONTROL: shows the
#           index patch can redirect retrieval at some depth (expected mid-layers; ~0 late, where the
#           residual already carries the donor's answer = exp's answer). Not a normalizer.
#   [12]    code_veracity_only    seed 1   ABA        held-out, mutate-b: copy status differs, read site + answer same
#   [13]    code_site_only        seed 1   ABA        held-out, mutate-b: both copy, read site list->mutation line
#   [14]    code_site_only_rev    seed 1   ABA        held-out, mutate-b: both copy, read site mutation line->list
#           (mutate-b conditions are defined with a copy base only: ABB raises)
# cma.py saves only after ALL prompts: a timeout/preemption loses the job.

TEMPLATE="indexed_list"

CONDITIONS=(
  "code_index"            "code_index"
  "code_index"            "code_index"
  "code_index_flip"       "code_index_flip"
  "code_abstract_two_mut" "code_abstract_two_mut"
  "code_abstract"         "code_abstract"
  "code_index"            "code_index"
  "code_veracity_only" "code_site_only" "code_site_only_rev"
)
RULES=(ABA ABB ABA ABB ABA ABB ABA ABB ABA ABB ABA ABB ABA ABA ABA)
SEEDS=(0 0 1 1 1 1 1 1 1 1 1 1 1 1 1)
ACTS=(z z z z z z z z z z resid_post resid_post z z z)

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}
SEED=${SEEDS[$IDX]}
ACT=${ACTS[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE seed=$SEED act=$ACT prompt_num=150"
# log the node + GPU so a bad node can be excluded on resubmission
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || echo "nvidia-smi FAILED on $(hostname)"

# no --patch_after_movement => end-patch position (-1)
python codebase/tasks/causal_analysis/cma.py \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --seed $SEED \
  --prompt_num 150 \
  --min_valid_sample_num 10 \
  --max_new_tokens 30 \
  --activation_name $ACT \
  --model_type Qwen2.5-14B-Instruct \
  --run_statistical_tests \
  --n_permutations 5000 \
  $WANDB_FLAG
