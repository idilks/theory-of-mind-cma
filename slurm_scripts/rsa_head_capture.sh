#!/bin/bash
#SBATCH --job-name=rsa_head_capture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-2%6
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_head_capture_%A_%a.out
#SBATCH --error=logs/rsa_head_capture_%A_%a.err

# Capture per-head hook_z outputs across the 4-cell balanced stimulus set.
# 3 templates x 48 stimuli x 34 heads (union of top-20 C1 + top-20 C5).
# Forward pass only — no patching — so this is much cheaper than CMA.

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

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE"

# Build stimulus CSVs and top-k head CSVs on the cluster if they don't exist.
# Cheap (CPU-only, seconds) and ensures reproducibility from a fresh checkout.
if [ ! -f results/rsa_head_outputs/stimuli/${TEMPLATE}.csv ]; then
  echo "Building stimulus set..."
  python scripts/rsa_head_outputs/build_stimulus_set.py
fi
if [ ! -f results/rsa_head_outputs/topk_heads/union_top20.csv ]; then
  echo "Selecting top-k heads..."
  python scripts/rsa_head_outputs/select_topk_heads.py
fi

python scripts/rsa_head_outputs/capture_head_outputs.py \
  --template $TEMPLATE \
  --heads_csv results/rsa_head_outputs/topk_heads/union_top20.csv \
  --stim_dir  results/rsa_head_outputs/stimuli \
  --out_dir   results/rsa_head_outputs/activations
