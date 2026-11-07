#!/bin/bash -l
#SBATCH --job-name=ablation_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --array=0-4%5
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

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
export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH

mkdir -p logs

# 5-run flip-matrix ablation. One row per CMA condition (head set).
# Each task ablates the top-K FWER-significant heads for that condition
# cumulatively and evaluates on ALL six prompt variants (full flip matrix):
# false/true belief, false/true photo, basic A->B, basic B->A.

JOB_CONDITIONS=(
  abstract_belief
  abstract_photo
  answer_changes_location_only
  answer_changes_location_only_photo
  state_only_no_belief
)
JOB_LABELS=(
  belief_heads
  photo_heads
  loc_belief_heads
  loc_photo_heads
  state_only_heads
)
# All five jobs now evaluate on the full 6-variant set so the 5x6 flip
# matrix has no missing cells.
ALL_VARIANTS="false_belief true_belief false_photo true_photo basic_ab basic_ba"
JOB_EVAL_VARIANTS=(
  "$ALL_VARIANTS"
  "$ALL_VARIANTS"
  "$ALL_VARIANTS"
  "$ALL_VARIANTS"
  "$ALL_VARIANTS"
)

IDX=$SLURM_ARRAY_TASK_ID
COND=${JOB_CONDITIONS[$IDX]}
LABEL=${JOB_LABELS[$IDX]}
EVAL_VARIANTS=${JOB_EVAL_VARIANTS[$IDX]}

echo "Job $IDX: head_condition=$COND label=$LABEL eval=$EVAL_VARIANTS"

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 150 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv results/lookback_analysis/_summary/top20_by_condition_end.csv \
  --condition $COND \
  --rank_by max_observed_score \
  --eval_variants $EVAL_VARIANTS \
  --run_label $LABEL \
  --output_dir results/ablation/$LABEL
