#!/bin/bash -l
#SBATCH --job-name=gemma_code_cma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --array=0-5%6
#SBATCH --mem=100GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/gemma_code_cma_%A_%a.out
#SBATCH --error=logs/gemma_code_cma_%A_%a.err

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

# Gemma-2-27B code-condition CMA (July 26 2026) — the first Gemma code CMA.
# Purpose: code-NATIVE Gemma head sets for the pad-twins/RDM arm (current
# gemma_c1..c5 are story-CMA only), and the second-model test of the
# code<->story head overlap (Qwen: code_abstract <-> C1 fingerprint r=0.867).
# Mirrors slurm_scripts/run_code_cma.sh (Qwen) with the Gemma flags from
# slurm_scripts/gemma/all_cma.sh. End patch, seed 0, indexed_list only.
# 6 jobs; priority if the queue is tight: [0-1] code_abstract (essential),
# [4-5] code_control (retrieval contrast), [2-3] code_answer_changes (H2
# analog, optional).
# Timing: Qwen code CMA n=150 = 4.3 h on h200; Gemma story CMA cells needed
# up to ~20 h — hence the 20:59 walltime. cma.py saves only after ALL
# prompts: a timeout loses the cell.

TEMPLATE="indexed_list"

CONDITIONS=(
  "code_abstract"       "code_abstract"
  "code_answer_changes" "code_answer_changes"
  "code_control"        "code_control"
)

RULES=(
  "ABA" "ABB"
  "ABA" "ABB"
  "ABA" "ABB"
)

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE seed=0 prompt_num=150 model=gemma-2-27b-it"

# no --patch_after_movement => end-patch position (-1)
python cma_scripts/tasks/causal_analysis/cma.py \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --seed 0 \
  --prompt_num 150 \
  --min_valid_sample_num 10 \
  --max_new_tokens 30 \
  --activation_name z \
  --model_type gemma-2-27b-it \
  --ungroup_grouped_query_attention \
  --run_statistical_tests \
  --n_permutations 5000
