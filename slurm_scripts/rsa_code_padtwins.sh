#!/bin/bash
#SBATCH --job-name=rsa_padtwins
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-1
#SBATCH --mem=100GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_padtwins_%A_%a.out
#SBATCH --error=logs/rsa_padtwins_%A_%a.err

# Full-head hook_z captures for the pad-twins grid (Exp 3 of the code RDM
# arm): 864 balanced stimuli (48 query cells x 2 content draws x 9 pad
# combos), veracity-tag vs lookback adjudication. Design in
# analysis_scripts/rsa_head_outputs/build_code_padtwins_stimuli.py; analysis in
# analyze_code_padtwins_rdms.py (run after pulling the .pt files).
# Array 0 = Qwen2.5-14B-Instruct, 1 = gemma-2-27b-it (GQA ungrouped).
# 864 prompts = 6x the code_ledger grid, which fit both grids in <3h.

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

# Builder runs its confound + pad-orthogonality checks; nonzero exit = bad grid.
if [ ! -f results/rsa_head_outputs/stimuli_code_padtwins/code_padtwins.csv ]; then
  python analysis_scripts/rsa_head_outputs/build_code_padtwins_stimuli.py || exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "Job 0: code_padtwins capture, Qwen2.5-14B-Instruct"
  python analysis_scripts/rsa_head_outputs/capture_head_outputs_full.py \
    --template code_padtwins \
    --stim_dir  results/rsa_head_outputs/stimuli_code_padtwins \
    --out_dir   results/rsa_head_outputs/activations_code_padtwins_full/Qwen2.5-14B-Instruct \
    --model_type Qwen2.5-14B-Instruct \
    --system_prompt code \
    --decode_answers
else
  echo "Job 1: code_padtwins capture, gemma-2-27b-it"
  python analysis_scripts/rsa_head_outputs/capture_head_outputs_full.py \
    --template code_padtwins \
    --stim_dir  results/rsa_head_outputs/stimuli_code_padtwins \
    --out_dir   results/rsa_head_outputs/activations_code_padtwins_full/gemma-2-27b-it \
    --model_type gemma-2-27b-it \
    --ungroup_grouped_query_attention \
    --system_prompt code \
    --decode_answers
fi
