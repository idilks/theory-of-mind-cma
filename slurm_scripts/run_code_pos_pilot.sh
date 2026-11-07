#!/bin/bash
#SBATCH --job-name=cma_code_pos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-3%4
#SBATCH --exclude=a02,l40sx8
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/code_pos_pilot_%A_%a.out
#SBATCH --error=logs/code_pos_pilot_%A_%a.err

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

# Position-matched pilot (Q3): does the copy/alias effect survive when the
# answer's token position and distance to the end are matched? One-element
# lists, comment-line fillers, n=40 per cell (~1.6 h/cell at 3 s/layer-step).
# Matched cells first so a partial run still yields the primary comparison;
# unmatched is the same-format calibration baseline (frozen-set effects).
#   [0-1]  code_pos_matched    ABA (copy->alias), ABB (alias->copy)  answer token index + length equal
#   [2-3]  code_pos_unmatched  ABA, ABB                               baseline: same fillers, positions differ
# Read-out: scripts/analyze_code_index_dissociation.py (frozen sets; T_C1
# split into the 6 index-shared heads vs remaining 26; L29H31, L35H30).
# cma.py saves only after ALL prompts: a timeout/preemption loses the job.

TEMPLATE="indexed_list"
CONDITIONS=("code_pos_matched" "code_pos_matched" "code_pos_unmatched" "code_pos_unmatched")
RULES=(ABA ABB ABA ABB)

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${CONDITIONS[$IDX]}
RULE=${RULES[$IDX]}

echo "Job $IDX: condition=$CONDITION template=$TEMPLATE rule=$RULE seed=1 act=z prompt_num=40"
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || echo "nvidia-smi FAILED on $(hostname)"

python codebase/tasks/causal_analysis/cma.py \
  --condition $CONDITION \
  --base_rule $RULE \
  --template_names $TEMPLATE \
  --seed 1 \
  --prompt_num 40 \
  --min_valid_sample_num 10 \
  --max_new_tokens 30 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --run_statistical_tests \
  --n_permutations 5000 \
  $WANDB_FLAG
