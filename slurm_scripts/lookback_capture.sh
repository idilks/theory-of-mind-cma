#!/bin/bash
#SBATCH --job-name=lookback_capture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-2
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lookback_%A_%a.out
#SBATCH --error=logs/lookback_%A_%a.err

# Capture attention patterns for top-50 CMA-significant heads (end / patch_before_movement)
# on the abstract_belief condition, ABA -> ABB direction, across 3 templates.
#
# Each array index = one template. Forward-pass-only analysis: no patching, no CMA.
# Roughly 50 prompts × 1 fwd pass on Qwen2.5-14B-Instruct.
#
# Array indices:
#   [0] food_truck
#   [1] hair_styling
#   [2] library_book

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

TEMPLATES=("food_truck" "hair_styling" "library_book")
TEMPLATE=${TEMPLATES[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_ARRAY_TASK_ID: lookback capture template=$TEMPLATE"

python scripts/lookback_analysis/capture_attention_patterns.py \
  --template $TEMPLATE \
  --condition abstract_belief \
  --base_rule ABA \
  --exp_rule ABB \
  --patch_position patch_before_movement \
  --top_k 50 \
  --prompt_num 50 \
  --seed 0 \
  --results_root codebase/results \
  --out_root results/lookback_analysis

# Auto-run analysis after capture (CPU-only; cheap)
CAP_DIR=results/lookback_analysis/abstract_belief/template_${TEMPLATE}/base_rule_ABA_exp_rule_ABB/patch_before_movement
python scripts/lookback_analysis/analyze_lookback_attention.py --capture_dir $CAP_DIR
