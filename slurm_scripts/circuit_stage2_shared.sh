#!/bin/bash
#SBATCH --job-name=circuit_s2_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_shared_%j.out
#SBATCH --error=logs/circuit_s2_shared_%j.err

# Two follow-ups to the cross-domain steering results, one GPU job:
# 1. Shared-component steering (stage2_shared_direction.py): the mediation
#    test — does the code wire's projection into ToM's subspace reproduce
#    the mirror effect, and does any single direction steer both domains?
# 2. ToM->code injection RERUN at the code-native scale 89 (the original
#    null ran at the undersized ToM scale 53.6) — closes the asymmetry
#    question. Only the cross legs re-run, fresh run label.
#
# Prereq (synced via results/aggregated): shared_direction_L41.npz from
# scripts/circuit/extract_shared_direction.py. Both scripts checkpoint per
# config — resubmit on preemption.

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

if [ ! -f results/aggregated/circuit/shared_direction_L41.npz ]; then
  echo "ERROR: shared_direction_L41.npz missing — run extract_shared_direction.py locally and sync." >&2
  exit 1
fi

python scripts/circuit/stage2_shared_direction.py --site L41

python scripts/circuit/stage2_steering.py \
  --domain tom \
  --site L41 \
  --only_cross_domain \
  --cross_scale other \
  --run_label cross_rescaled
