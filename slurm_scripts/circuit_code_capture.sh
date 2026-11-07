#!/bin/bash
#SBATCH --job-name=circuit_code_capture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_code_capture_%j.out
#SBATCH --error=logs/circuit_code_capture_%j.err

# Code-domain Stage-1 capture for the layered-lookback circuit: 150 matched
# code_copy/code_alias pairs (single indexed_list template, CODE_SYSTEM_PROMPT,
# quote prefill), same hooks as the ToM capture. Reuses the existing
# weights.pt from the ToM capture dir — no --save_weights here.
#
# Downstream (laptop):
#   python analysis_scripts/circuit/stage1_geometric_screen.py --domain code
#   python analysis_scripts/circuit/compare_wires.py \
#     --a results/aggregated/circuit/stage1/directions_L41.npz \
#     --b results/aggregated/circuit/stage1_code/directions_L41.npz

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

python analysis_scripts/circuit/capture_circuit_activations.py \
  --domain code \
  --prompt_num 150 \
  --seed 0
