#!/bin/bash
#SBATCH --job-name=circuit_s2_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_code_%j.out
#SBATCH --error=logs/circuit_s2_code_%j.err

# Code-domain Stage-2 steering: inject/delete the code wire
# (stage1_code/directions_L41.npz, natural scale 89.0) on held-out
# code_alias/code_copy prompts with the full control battery, plus the
# pre-registered mirror leg (code direction -> held-out ToM true_belief
# prompts at the ToM natural scale; predicted null, wire cos = 0.0997).
# Checkpoints per config; resubmit on preemption.

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

for f in results/aggregated/circuit/stage1_code/directions_L41.npz \
         results/aggregated/circuit/stage1/directions_L41.npz; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f missing — run the stage-1 screens locally and sync." >&2
    exit 1
  fi
done

python analysis_scripts/circuit/stage2_steering.py \
  --domain code \
  --site L41
