#!/bin/bash
#SBATCH --job-name=lookback_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-41%6
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lookback_full_%A_%a.out
#SBATCH --error=logs/lookback_full_%A_%a.err

# Full lookback capture: 7 conditions x (1 or 2) directions x 3 templates = 42 jobs.
# Forward-pass only on top-50 CMA-significant heads from end-position (patch_before_movement),
# now ranked by signed positive observed_score (not |score|).
#
# IMPORTANT: --seed 1 and --prompt_num 150 to match the canonical CMA runs
# (z_seed_1_shuffle_False, sample_num up to 150). Same RNG => same regenerated prompts.
#
# Condition labelling (post 2026-05 rename):
#   C1 = abstract_belief                       (cross-rule)
#   C2 = abstract_photo                        (cross-rule)
#   C3 = answer_changes_belief                 (cross-rule)
#   C4 = answer_changes_photo                  (cross-rule)
#   C5 = answer_changes_location_only          (ToM-frame location control, same-rule)
#   C6 = answer_changes_location_only_photo    (photo-frame location control, same-rule, NEW)
#   C7 = state_only_no_belief                  (state-only no-agent control, same-rule)
#
# Cell layout (14 cells x 3 templates = array indices 0-41):
#   cell  CONDITION                          BASE EXP   array indices   label
#   0     abstract_belief                    ABA  ABB   0,1,2           C1
#   1     abstract_belief                    ABB  ABA   3,4,5           C1 (reverse)
#   2     abstract_photo                     ABA  ABB   6,7,8           C2
#   3     abstract_photo                     ABB  ABA   9,10,11         C2 (reverse)
#   4     answer_changes_belief              ABA  ABB   12,13,14        C3
#   5     answer_changes_belief              ABB  ABA   15,16,17        C3 (reverse)
#   6     answer_changes_photo               ABA  ABB   18,19,20        C4
#   7     answer_changes_photo               ABB  ABA   21,22,23        C4 (reverse)
#   8     answer_changes_location_only       ABA  ABA   24,25,26        C5
#   9     answer_changes_location_only       ABB  ABB   27,28,29        C5 (reverse)
#  10     answer_changes_location_only_photo ABA  ABA   30,31,32        C6 (NEW)
#  11     answer_changes_location_only_photo ABB  ABB   33,34,35        C6 (reverse, NEW)
#  12     state_only_no_belief               ABA  ABA   36,37,38        C7
#  13     state_only_no_belief               ABB  ABB   39,40,41        C7 (reverse)
# templates per cell: food_truck, hair_styling, library_book

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

# 14 cells, ordered: (condition, base_rule, exp_rule)
CONDITIONS=( \
  "abstract_belief"                    "abstract_belief" \
  "abstract_photo"                     "abstract_photo" \
  "answer_changes_belief"              "answer_changes_belief" \
  "answer_changes_photo"               "answer_changes_photo" \
  "answer_changes_location_only"       "answer_changes_location_only" \
  "answer_changes_location_only_photo" "answer_changes_location_only_photo" \
  "state_only_no_belief"               "state_only_no_belief" \
)
BASE_RULES=("ABA" "ABB" "ABA" "ABB" "ABA" "ABB" "ABA" "ABB" "ABA" "ABB" "ABA" "ABB" "ABA" "ABB")
EXP_RULES=( "ABB" "ABA" "ABB" "ABA" "ABB" "ABA" "ABB" "ABA" "ABA" "ABB" "ABA" "ABB" "ABA" "ABB")

CELL=$(( SLURM_ARRAY_TASK_ID / 3 ))
TIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

CONDITION=${CONDITIONS[$CELL]}
BASE_RULE=${BASE_RULES[$CELL]}
EXP_RULE=${EXP_RULES[$CELL]}
TEMPLATE=${TEMPLATES[$TIDX]}

echo "Job $SLURM_ARRAY_TASK_ID: condition=$CONDITION base=$BASE_RULE exp=$EXP_RULE template=$TEMPLATE"

python analysis_scripts/lookback_analysis/capture_attention_patterns.py \
  --template $TEMPLATE \
  --condition $CONDITION \
  --base_rule $BASE_RULE \
  --exp_rule $EXP_RULE \
  --patch_position patch_before_movement \
  --top_k 50 \
  --prompt_num 150 \
  --seed 1 \
  --results_root cma_scripts/results \
  --out_root results/lookback_analysis

CAP_DIR=results/lookback_analysis/${CONDITION}/template_${TEMPLATE}/base_rule_${BASE_RULE}_exp_rule_${EXP_RULE}/patch_before_movement
python analysis_scripts/lookback_analysis/analyze_lookback_attention.py --capture_dir $CAP_DIR
python analysis_scripts/lookback_analysis/per_token_profile.py --capture_dir $CAP_DIR --top_k 5

# After the LAST array task completes, aggregate unannotated peaks across all 42 cells.
# (Each task does this; the last-running one wins. Cheap, idempotent.)
python analysis_scripts/lookback_analysis/per_token_profile.py --root results/lookback_analysis || true
