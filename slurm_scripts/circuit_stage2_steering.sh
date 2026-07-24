#!/bin/bash
#SBATCH --job-name=circuit_s2_steer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_steer_%j.out
#SBATCH --error=logs/circuit_s2_steer_%j.err

# Stage-2 causal steering for the layered-lookback circuit
# (docs/design_layered_lookback_circuit.md). Injects/deletes the Stage-1
# wire directions at resid_pre L42 on held-out prompts, dose grid x full
# control battery + specificity + cross-domain (ToM direction -> code_alias).
# Forward passes only; ~15k prompts total. Checkpoints per config into
# per_prompt.csv and skips finished configs on restart, so preemption is
# recoverable — just resubmit.
#
# Prerequisites on the HPC tree:
#   results/circuit/stage1_capture/Qwen2.5-14B-Instruct/{template}.pt + weights.pt
#     (produced here by circuit_stage1_capture.sh)
#   results/aggregated/circuit/stage1/directions_L41.npz
#     (produced on the laptop by stage1_geometric_screen.py; synced via
#      sync_to_hpc.sh because it lives under results/aggregated/)

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

if [ ! -f results/aggregated/circuit/stage1/directions_L41.npz ]; then
  echo "ERROR: stage-1 directions missing — run stage1_geometric_screen.py locally and sync." >&2
  exit 1
fi

python scripts/circuit/stage2_steering.py \
  --site L41 \
  --run_label main
