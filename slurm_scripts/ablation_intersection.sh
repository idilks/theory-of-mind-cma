#!/bin/bash -l
#SBATCH --job-name=ablation_intersection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/ablation_intersection_%A.out
#SBATCH --error=logs/ablation_intersection_%A.err

# Ablate the abstract_belief n abstract_photo intersection heads (~19 heads
# at end-position) and evaluate on the full 6-variant flip-matrix set.
# Heads ranked by geometric mean of belief and photo max_observed_score.
#
# Prereq: run scripts/build_intersection_head_csv.py first to generate
#   results/lookback_analysis/_summary/top20_with_intersection_end.csv

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

# Build the intersection-augmented head CSV if missing.
if [ ! -f results/lookback_analysis/_summary/top20_with_intersection_end.csv ]; then
  echo "Building intersection head CSV..."
  python scripts/build_intersection_head_csv.py
fi

LABEL=intersection_belief_photo

python codebase/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 150 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv results/lookback_analysis/_summary/top20_with_intersection_end.csv \
  --condition $LABEL \
  --rank_by max_observed_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --run_label $LABEL \
  --output_dir results/ablation/$LABEL
