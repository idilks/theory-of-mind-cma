#!/bin/bash
#SBATCH --job-name=rsa_head_capture_content
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --array=0-2%6
#SBATCH --mem=120GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_head_capture_content_%A_%a.out
#SBATCH --error=logs/rsa_head_capture_content_%A_%a.err

# Capture per-head hook_z outputs (all 1920 heads, final token) over the
# CONTENT-separable ToM stimulus set (Sept 2026, Sept 3 meeting: Bio et al.
# mirror -- believed-location identity decorrelated from veracity and
# scenario). 120 stimuli per template (6 locations x 5 partners x 2 belief x
# 2 replicate draws), vs 48 in activations_full; ~2.5x the runtime of
# rsa_head_capture_full.sh. --decode_answers adds a greedy 8-token generation
# per prompt for the baseline-accuracy gate (roughly doubles runtime).
#
# Outputs:
#   results/rsa_head_outputs/activations_content/{template}.pt
#     z: (120, 48, 40, 128) float16 (~180 MB per template), stim, generations
#
# Downstream (local, CPU):
#   python analysis_scripts/rsa_head_outputs/analyze_content_rdms.py
#   python analysis_scripts/rsa_head_outputs/mvpa_veracity.py \
#       --act_dir results/rsa_head_outputs/activations_content --target believed

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

STIM_DIR=results/rsa_head_outputs/stimuli_content
ACT_DIR=results/rsa_head_outputs/activations_content

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE"

# Do NOT build stimuli inside the array: the builder writes all three CSVs,
# so three concurrent tasks race on the same files (job 9383566 left a
# fragment row in library_book.csv). Build locally, sync, then submit.
if [ ! -f $STIM_DIR/${TEMPLATE}.csv ]; then
  echo "missing $STIM_DIR/${TEMPLATE}.csv -- run build_content_stimulus_set.py locally and sync"; exit 1
fi

mkdir -p $ACT_DIR
python analysis_scripts/rsa_head_outputs/capture_head_outputs_full.py \
  --template $TEMPLATE \
  --stim_dir  $STIM_DIR \
  --out_dir   $ACT_DIR \
  --decode_answers || { echo "capture FAILED for $TEMPLATE"; exit 1; }

echo "done. Output: $ACT_DIR/${TEMPLATE}.pt"
