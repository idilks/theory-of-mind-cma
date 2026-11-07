#!/bin/bash
#SBATCH --job-name=code_attn_rows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/code_attn_rows_%j.out
#SBATCH --error=logs/code_attn_rows_%j.err

# Forward-only capture of full final-token attention rows for the code
# divergence (code_abstract / K1) and retrieval (code_control / K5) heads on the
# padtwins stimuli, so "rest of code" can be split into lines locally.
# Qwen2.5-14B-Instruct, all 864 padtwins stimuli (padded variants kept; filter later).
# No patching, no generation. Output (~5 MB, pulled by pull_results.sh block 4):
#   results/rsa_head_outputs/activations_code_attn_rows/code_attention_rows.pt

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
mkdir -p logs

STIM=results/rsa_head_outputs/stimuli_code_padtwins/code_padtwins.csv
if [ ! -f $STIM ]; then
  python scripts/rsa_head_outputs/build_code_padtwins_stimuli.py || exit 1
fi

python scripts/lookback_analysis/capture_code_attention_rows.py \
  --stim_csv $STIM \
  --model_type Qwen2.5-14B-Instruct
