#!/bin/bash
#SBATCH --job-name=circuit_s1_capture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-2%6
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s1_capture_%A_%a.out
#SBATCH --error=logs/circuit_s1_capture_%A_%a.err

# Stage-1 capture for the layered-lookback circuit
# (docs/design_layered_lookback_circuit.md). 150 matched FB/TB pairs per
# template, forward passes only: hook_z + hook_mlp_out L28-40, resid_post
# L40/41 at position -1, under the matched-prompt+prefill protocol.
# Array task 0 additionally saves the weight-slice artifact (W_O band,
# reader W_Q/W_V/ln1.w) so the laptop screen needs no model load.
#
# Prerequisite (synced via results/aggregated): run locally first
#   python scripts/circuit/build_circuit_head_sets.py

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

EXTRA=""
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  EXTRA="--save_weights"
fi

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE $EXTRA"

python scripts/circuit/capture_circuit_activations.py \
  --template "$TEMPLATE" \
  --prompt_num 150 \
  --seed 0 \
  $EXTRA
