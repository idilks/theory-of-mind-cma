#!/bin/bash
#SBATCH --job-name=abl_rand_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-3%4
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_rand_code_%A_%a.out
#SBATCH --error=logs/abl_rand_code_%A_%a.err

# Experiment 3: random-head nulls for the code-domain ablation arm.
# k random heads (matched k), sampled UNIFORMLY from all 48x40=1920 heads
# excluding the targeted set (--sampling uniform; the ToM nulls used
# layer_matched — flag added for this arm), same evals as the targets:
#   [0] null vs code_abstract    (k=12) on code_copy+code_alias, quote prefill
#   [1] null vs tom_intersection (k=19) on code_copy+code_alias, quote prefill
#   [2] null vs code_control     (k=14) on code_copy+code_alias, quote prefill
#   [3] null vs code_abstract    (k=12) on ToM false_belief, matched
#       sysprompt + prefill (companion null for experiment 2b)
#
# Prereq: results/aggregated/code_head_sets/*_head_list.csv on HPC
# (build locally with `python analysis_scripts/build_code_head_csvs.py`, scp over).

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

HEAD_DIR=results/aggregated/code_head_sets

TARGETS=(
  "$HEAD_DIR/code_abstract_head_list.csv"
  "$HEAD_DIR/tom_intersection_head_list.csv"
  "$HEAD_DIR/code_control_head_list.csv"
  "$HEAD_DIR/code_abstract_head_list.csv"
)
KS=( 12 19 14 12 )
VARIANTS=( "code_copy code_alias" "code_copy code_alias" "code_copy code_alias" "false_belief" )
SYSPROMPTS=(
  "You are a helpful assistant. Answer concisely."
  "You are a helpful assistant. Answer concisely."
  "You are a helpful assistant. Answer concisely."
  "You are a helpful assistant reasoning about agents' beliefs."
)
LABELS=( "code_abstract_on_code" "tom_intersection_on_code" "code_control_on_code" "code_abstract_on_tom" )

IDX=$SLURM_ARRAY_TASK_ID
TARGET=${TARGETS[$IDX]}
K=${KS[$IDX]}
LABEL=${LABELS[$IDX]}

if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET not present on HPC. Build it locally and scp it over." >&2
  exit 1
fi

echo "Job $IDX: null vs $TARGET (k=$K) on '${VARIANTS[$IDX]}'"

# NB: --system_prompt is ignored for code variants (they always use
# CODE_SYSTEM_PROMPT, matching the code CMA protocol).
python analysis_scripts/random_baseline_ablation.py \
  --target_csv $TARGET \
  --n_random 100 \
  --k_eval $K \
  --sampling uniform \
  --eval_variants ${VARIANTS[$IDX]} \
  --prompt_num 50 \
  --max_new_tokens 8 \
  --templates food_truck hair_styling library_book \
  --system_prompt "${SYSPROMPTS[$IDX]}" \
  --assistant_prefill per_variant \
  --seed 42 \
  --n_devices 1 \
  --device_map cuda \
  --output_dir results/ablation/random_baselines/$LABEL
