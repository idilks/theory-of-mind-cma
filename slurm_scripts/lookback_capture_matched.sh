#!/bin/bash
#SBATCH --job-name=lookback_matched
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-11%6
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lookback_matched_%A_%a.out
#SBATCH --error=logs/lookback_matched_%A_%a.err

# Matched-stimulus attention capture: the divergence AND retrieval head groups
# (head_groups.py: CMA-significant in >= half of the condition's cells) measured
# on the SAME prompts, for both prompt types. Forward passes only.
#
#   belief: heads C1 (abstract_belief) + C5 (answer_changes_location_only)
#           on C1 prompts, base ABA (false-belief) and base ABB (true-belief)
#   photo:  heads C2 (abstract_photo) + C6 (answer_changes_location_only_photo)
#           on C2 prompts, base ABA (outdated photo) and base ABB (current photo)
#
# --seed 1 --prompt_num 150 regenerate the canonical CMA prompts (as lookback_capture_full.sh).
# Array index -> (domain, prompt type, template):
#   0-2  belief FB  food_truck, hair_styling, library_book
#   3-5  belief TB  ...
#   6-8  photo  FB  ...
#   9-11 photo  TB  ...
# Output: results/lookback_analysis/_matched/{abstract_belief|abstract_photo}/template_*/base_rule_*/patch_before_movement/attention.pt
# Then run slurm_scripts/region_attention_shares.sh with MATCHED=1.

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
mkdir -p logs

TEMPLATES=("food_truck" "hair_styling" "library_book")
BLOCK=$(( SLURM_ARRAY_TASK_ID / 3 ))
TEMPLATE=${TEMPLATES[$(( SLURM_ARRAY_TASK_ID % 3 ))]}

case $BLOCK in
  0) COND=abstract_belief; BASE=ABA; EXP=ABB; HEADS="abstract_belief answer_changes_location_only" ;;
  1) COND=abstract_belief; BASE=ABB; EXP=ABA; HEADS="abstract_belief answer_changes_location_only" ;;
  2) COND=abstract_photo;  BASE=ABA; EXP=ABB; HEADS="abstract_photo answer_changes_location_only_photo" ;;
  3) COND=abstract_photo;  BASE=ABB; EXP=ABA; HEADS="abstract_photo answer_changes_location_only_photo" ;;
esac
echo "task $SLURM_ARRAY_TASK_ID: prompts=$COND base=$BASE template=$TEMPLATE heads=[$HEADS]"

python scripts/lookback_analysis/capture_attention_patterns.py \
  --template $TEMPLATE \
  --condition $COND \
  --base_rule $BASE \
  --exp_rule $EXP \
  --patch_position patch_before_movement \
  --head_conditions $HEADS \
  --prompt_num 150 \
  --seed 1 \
  --results_root codebase/results \
  --out_root results/lookback_analysis/_matched
