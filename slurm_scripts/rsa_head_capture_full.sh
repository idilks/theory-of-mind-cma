#!/bin/bash
#SBATCH --job-name=rsa_head_capture_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --array=0-2%6
#SBATCH --mem=120GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_head_capture_full_%A_%a.out
#SBATCH --error=logs/rsa_head_capture_full_%A_%a.err

# Capture per-head hook_z outputs for ALL 1920 heads (48 layers x 40 heads)
# across the balanced RSA stimulus set, per template.
#
# Companion to rsa_head_capture_7cond.sh, which captures only the union
# top-20 (~43 heads). The full capture lets the downstream analysis
# CMA-weight every head — Taylor's "do it over all heads" suggestion.
#
# Outputs:
#   results/rsa_head_outputs/activations_full/{template}.pt
#     z: (n_stim, 48, 40, 128) float16, ~70 MB per template

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

ACT_DIR=results/rsa_head_outputs/activations_full

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE"

if [ ! -f results/rsa_head_outputs/stimuli/${TEMPLATE}.csv ]; then
  echo "Building stimulus set..."
  python analysis_scripts/rsa_head_outputs/build_stimulus_set.py
fi

mkdir -p $ACT_DIR
python analysis_scripts/rsa_head_outputs/capture_head_outputs_full.py \
  --template $TEMPLATE \
  --stim_dir  results/rsa_head_outputs/stimuli \
  --out_dir   $ACT_DIR

echo "done. Output: $ACT_DIR/${TEMPLATE}.pt"
