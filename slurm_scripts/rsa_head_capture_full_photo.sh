#!/bin/bash
#SBATCH --job-name=rsa_head_capture_full_photo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --array=0-2%6
#SBATCH --mem=100GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_head_capture_full_photo_%A_%a.out
#SBATCH --error=logs/rsa_head_capture_full_photo_%A_%a.err

# Photo-frame analog of rsa_head_capture_full.sh. Reads the photo-context
# stimulus set (built locally by build_stimulus_set.py --context_type photo)
# and captures per-head hook_z at the final input token for all 1920 heads.
#
# Purpose: feed DLA on photo-frame activations so we can plot a C2-condition
# scatter analogous to the belief-frame DLA scatter. Lets us check whether
# CMA-significant heads in C2 (abstract_photo) sit at |DLA| ~ 0 (routing)
# the same way C1 heads do.
#
# Outputs:
#   results/rsa_head_outputs/activations_full_photo/{template}.pt
#     z: (48, 48, 40, 128) float16, ~70 MB per template

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

STIM_DIR=results/rsa_head_outputs/stimuli_photo
ACT_DIR=results/rsa_head_outputs/activations_full_photo

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE"

if [ ! -f ${STIM_DIR}/${TEMPLATE}.csv ]; then
  echo "Building photo stimulus set..."
  python scripts/rsa_head_outputs/build_stimulus_set.py --context_type photo
fi

mkdir -p $ACT_DIR
python scripts/rsa_head_outputs/capture_head_outputs_full.py \
  --template $TEMPLATE \
  --stim_dir  $STIM_DIR \
  --out_dir   $ACT_DIR

echo "done. Output: $ACT_DIR/${TEMPLATE}.pt"
