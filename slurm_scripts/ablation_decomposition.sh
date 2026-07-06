#!/bin/bash -l
#SBATCH --job-name=abl_decomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --array=0-3%4
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_decomp_%A_%a.out
#SBATCH --error=logs/abl_decomp_%A_%a.err

# Cross-frame ablation decomposition: tests whether the published cross-frame
# flips are driven by the B∩P intersection or by selection-only heads.
#
# 4 jobs run in parallel:
#   0 intersection_photo_rank  (K=19) -> eval false_belief  (cf. published top row, K=26)
#   1 photo_only               (K=7)  -> eval false_belief  (photo-selection heads NOT in belief set)
#   2 intersection_belief_rank (K=19) -> eval false_photo   (cf. published bottom row, K=32)
#   3 belief_only              (K=13) -> eval false_photo   (belief-selection heads NOT in photo set)
#
# Protocol matches scripts/render_cross_frame_ablation.py "corrected" rows
# (matched sysprompt + per_variant prefill) so curves are directly comparable.
#
# Prereq:  python scripts/build_decomposition_head_csvs.py
#          (produces results/lookback_analysis/_summary/decomposition_heads_end.csv)

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

HEAD_CSV=results/lookback_analysis/_summary/decomposition_heads_end.csv
# NOTE: the build script's inputs (other_results/ablation/*/head_list.csv)
# are excluded from sync_to_hpc.sh, so we cannot build on HPC. Build locally
# with `python scripts/build_decomposition_head_csvs.py` and scp the CSV to
# /dartfs/rc/lab/F/FranklandS/tom/results/lookback_analysis/_summary/.
if [ ! -f "$HEAD_CSV" ]; then
  echo "ERROR: $HEAD_CSV not present on HPC. Build it locally and scp it over." >&2
  exit 1
fi

JOB_CONDITIONS=(
  intersection_photo_rank
  photo_only
  intersection_belief_rank
  belief_only
)
JOB_LABELS=(
  decomp_intersection_photo_rank
  decomp_photo_only
  decomp_intersection_belief_rank
  decomp_belief_only
)

IDX=$SLURM_ARRAY_TASK_ID
COND=${JOB_CONDITIONS[$IDX]}
LABEL=${JOB_LABELS[$IDX]}

echo "Job $IDX: condition=$COND label=$LABEL"

python codebase/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv $HEAD_CSV \
  --condition $COND \
  --rank_by mean_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label $LABEL \
  --output_dir results/ablation/$LABEL
