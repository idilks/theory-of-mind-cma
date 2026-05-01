#!/bin/bash
#SBATCH --job-name=code_baseline_acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/code_baseline_%j.out
#SBATCH --error=logs/code_baseline_%j.err

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# No patching, just generation accuracy on code-tracing prompts.
# 50 prompts/template × 3 variants × 3 templates = 450 generations.
# At ~30 tokens each on H200, expect well under 30 minutes.

python scripts/code_baseline_acc.py \
    --model_type Qwen2.5-14B-Instruct \
    --prompt_num 50 \
    --templates list_append dict_update list_extend \
    --max_new_tokens 30 \
    --seed 0 \
    --output results/code_baseline/qwen14b_acc.json
