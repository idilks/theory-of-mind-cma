#!/bin/bash -l
#SBATCH --job-name=abl_overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --array=0-2%3
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_overlap_%A_%a.out
#SBATCH --error=logs/abl_overlap_%A_%a.err

# C1 vs C5 overlap diagnostic. Decomposes the T_C5-flips-false_belief
# result by ablating each subset of T_C1 ∪ T_C5 separately:
#   IDX=0  C1 \ C5  (a_only,        21 heads)  belief-unique
#   IDX=1  C5 \ C1  (b_only,        11 heads)  loc-unique
#   IDX=2  C1 ∩ C5  (intersection,  11 heads)  shared
# Override concurrency at submit time with e.g. `sbatch --array=0-2%2 ...`.

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

PAIR="abstract_belief,answer_changes_location_only"

JOB_SETS=(a_only b_only intersection)
JOB_LABELS=(belief_minus_loc loc_minus_belief shared_belief_loc)

ALL_VARIANTS="false_belief true_belief false_photo true_photo basic_ab basic_ba"

IDX=$SLURM_ARRAY_TASK_ID
SET=${JOB_SETS[$IDX]}
LABEL=${JOB_LABELS[$IDX]}

echo "Job $IDX: pair=$PAIR set=$SET label=$LABEL"

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 150 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source overlap_set \
  --head_csv results/lookback_analysis/_summary/condition_overlap_heads_end.csv \
  --pair "$PAIR" \
  --set $SET \
  --eval_variants $ALL_VARIANTS \
  --run_label $LABEL \
  --output_dir results/ablation/$LABEL
